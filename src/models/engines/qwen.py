"""Qwen leg in one file: weights + KV-cache + attention + engine.

Default model class :class:`QwenEngine` (alias ``TinyLLM``), currently
Qwen3-0.6B. Qwen3 adds per-head QK-norm and drops attention biases vs
Qwen2.5; both are auto-detected from the state dict, so Qwen2.5
checkpoints keep working unchanged. All Triton kernels come from
:mod:`src.models.triton_kernels.qwen` (rmsnorm, rope, rope_batched,
swiglu, gqa_decode_attn, fused_qkv_gqa) with exact torch fallbacks, so this
module imports and runs on CPU too.
"""
import glob
import math
import os
import time

import torch
import torch.nn.functional as F

from src.models.triton_kernels.qwen import (
    HAVE_TRITON_KERNELS,
    build_cos_sin,
    fused_qkv_gqa,
    gqa_decode_attn,
    rmsnorm,
    rope,
    rope_batched,
    swiglu,
)

__all__ = ["MAXN", "MODEL_ID", "QwenEngine", "TinyLLM", "HAVE_TRITON_KERNELS"]

MODEL_ID = "Qwen/Qwen3-0.6B"
MAXN = 512

FALLBACK_DIMS = {
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "hidden_size": 1024,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
}


# -- weights ------------------------------------------------------------
def snapshot_path(repo_id=MODEL_ID, local_dir=None):
    """Local HF snapshot dir (lazy huggingface_hub), downloading if needed."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, allow_patterns=["*.safetensors"],
                             local_dir=local_dir)


def load_weights(weights_path=None, device="cuda:0", repo_id=MODEL_ID):
    """Load all ``*.safetensors`` from a dir (or HF snapshot) -> dict."""
    from safetensors.torch import load_file
    if weights_path is None:
        weights_path = snapshot_path(repo_id)
    if os.path.isdir(weights_path):
        files = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(
                "no .safetensors under %s" % weights_path)
        w = {}
        for f in files:
            w.update(load_file(f, device=device))
        return dict(w)
    return dict(load_file(weights_path, device=device))


def load_hf_config(repo_id=MODEL_ID):
    """Model dims via lazy transformers, else the Qwen3-0.6B fallback table."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo_id)
        hidden = cfg.hidden_size
        heads = cfg.num_attention_heads
        return {
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "hidden_size": hidden,
            "head_dim": int(getattr(cfg, "head_dim", None)
                            or hidden // heads),
            "rms_norm_eps": float(cfg.rms_norm_eps),
            "max_position_embeddings": cfg.max_position_embeddings,
            "rope_theta": float(getattr(cfg, "rope_theta", None)
                                or 1000000.0),
        }
    except Exception:
        return dict(FALLBACK_DIMS)


def lm_head_weight(w):
    """Return the output-projection matrix, honouring the tied LM head."""
    if "lm_head.weight" in w:
        return w["lm_head.weight"]
    return w["model.embed_tokens.weight"]


# -- KV cache -----------------------------------------------------------
class KVCache:
    """Per-layer key/value cache, shape [nlayers, Hk, max_len, dh] each."""

    def __init__(self, nlayers, n_kv_heads, max_len, head_dim,
                 device="cuda:0", dtype=None):
        self.nlayers = int(nlayers)
        self.n_kv_heads = int(n_kv_heads)
        self.max_len = int(max_len)
        self.head_dim = int(head_dim)
        self.device = device
        dev = torch.device(device)
        kw = {"device": dev}
        if dtype is not None:
            kw["dtype"] = dtype
        self.K = [torch.empty(n_kv_heads, max_len, head_dim, **kw)
                  for _ in range(nlayers)]
        self.V = [torch.empty(n_kv_heads, max_len, head_dim, **kw)
                  for _ in range(nlayers)]
        self.lengths = [0] * nlayers

    def store_prefill(self, layer, k, v):
        """Store prefill K/V [T, Hk, dh] at positions 0..T-1."""
        T = k.shape[0]
        self.K[layer][:, :T] = k.transpose(0, 1)
        self.V[layer][:, :T] = v.transpose(0, 1)
        self.lengths[layer] = T

    def store_step(self, layer, pos, k, v):
        """Store one decode step K/V [Hk, dh] at ``pos``."""
        self.K[layer][:, pos] = k
        self.V[layer][:, pos] = v
        self.lengths[layer] = max(self.lengths[layer], pos + 1)

    def get(self, layer, end=None):
        """(K, V) slices [Hk, N, dh] for ``N = end`` (default: filled)."""
        n = self.lengths[layer] if end is None else int(end)
        return self.K[layer][:, :n], self.V[layer][:, :n]

    def reset(self):
        self.lengths = [0] * self.nlayers


# -- blocks -------------------------------------------------------------
def mlp_forward(h, w, prefix):
    """Gated SwiGLU MLP output-projection result (caller adds residual)."""
    g = F.linear(h, w[prefix + "mlp.gate_proj.weight"], None)
    u = F.linear(h, w[prefix + "mlp.up_proj.weight"], None)
    return F.linear(swiglu(g, u), w[prefix + "mlp.down_proj.weight"], None)


def prefill_attention(x, w, prefix, cos, sin, n_heads, n_kv_heads, head_dim,
                       cache=None, layer=0, qk_norm=None):
    """Full-sequence prefill attention with RoPE positions 0..T-1.

    ``qk_norm`` is an optional (q_norm_w, k_norm_w) pair (Qwen3 QK-norm);
    biases are looked up tolerantly (Qwen3 drops attention biases).
    """
    T = x.shape[0]
    q = F.linear(x, w[prefix + "self_attn.q_proj.weight"],
                 w.get(prefix + "self_attn.q_proj.bias"))
    k = F.linear(x, w[prefix + "self_attn.k_proj.weight"],
                 w.get(prefix + "self_attn.k_proj.bias"))
    v = F.linear(x, w[prefix + "self_attn.v_proj.weight"],
                 w.get(prefix + "self_attn.v_proj.bias"))
    if qk_norm is not None:
        # Qwen3 order (verified vs HF logits): norm BEFORE rotary.
        qw, kw = qk_norm
        q = rmsnorm(q.reshape(T, n_heads, head_dim), qw).reshape(
            T, n_heads * head_dim)
        k = rmsnorm(k.reshape(T, n_kv_heads, head_dim), kw).reshape(
            T, n_kv_heads * head_dim)
    # Batched RoPE: ONE launch per q/k for the whole prefill.
    qr = rope_batched(q.reshape(T, n_heads, head_dim), cos, sin, 0).reshape(
        T, n_heads * head_dim)
    kr = rope_batched(k.reshape(T, n_kv_heads, head_dim), cos, sin, 0).reshape(
        T, n_kv_heads * head_dim)
    if cache is not None:
        cache.store_prefill(layer, kr.reshape(T, n_kv_heads, head_dim),
                             v.reshape(T, n_kv_heads, head_dim))
    q4 = qr.view(T, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k4 = kr.view(T, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k4 = k4.repeat_interleave(n_heads // n_kv_heads, dim=1)
    v4 = v.view(T, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    v4 = v4.repeat_interleave(n_heads // n_kv_heads, dim=1)
    o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)[0]
    o = o.transpose(0, 1).reshape(T, n_heads * head_dim)
    return F.linear(o, w[prefix + "self_attn.o_proj.weight"], None)


def decode_attention_step(x, w, prefix, cos, sin, pos, n_heads, n_kv_heads,
                           head_dim, scale, cache=None, layer=0,
                           Kcache=None, Vcache=None, qk_norm=None):
    """Single-token decode step with KV-cache write.

    Tries the fused Triton GQA-QKV path (only when all three biases exist;
    Qwen3 is bias-free and always takes the torch path), falls back to
    torch on failure.
    """
    d = n_heads * head_dim
    bq = w.get(prefix + "self_attn.q_proj.bias")
    bk = w.get(prefix + "self_attn.k_proj.bias")
    bv = w.get(prefix + "self_attn.v_proj.bias")
    # NOTE (measured in-engine): the fused Triton QKV path is FASTER here
    # (31ms/10tok) than eager 3xF.linear (83ms/10tok), so it stays primary
    # with the torch path as fallback.
    try:
        if (isinstance(x, torch.Tensor) and x.is_cuda
                and bq is not None and bk is not None and bv is not None):
            qf, kf, vf = fused_qkv_gqa(
                x.reshape(-1).contiguous(),
                w[prefix + "self_attn.q_proj.weight"],
                w[prefix + "self_attn.k_proj.weight"],
                w[prefix + "self_attn.v_proj.weight"],
                bq, bk, bv,
            )
            q = qf.reshape(n_heads, head_dim)
            k1 = kf.reshape(n_kv_heads, head_dim)
            v1 = vf.reshape(n_kv_heads, head_dim)
        else:
            raise RuntimeError("CPU/bias-free: use torch path")
    except Exception:
        q = F.linear(x, w[prefix + "self_attn.q_proj.weight"],
                     bq).reshape(n_heads, head_dim)
        k1 = F.linear(x, w[prefix + "self_attn.k_proj.weight"],
                      bk).reshape(n_kv_heads, head_dim)
        v1 = F.linear(x, w[prefix + "self_attn.v_proj.weight"],
                      bv).reshape(n_kv_heads, head_dim)
    if qk_norm is not None:
        # Qwen3 order (verified vs HF logits): norm BEFORE rotary.
        qw, kw = qk_norm
        q = rmsnorm(q, qw)
        k1 = rmsnorm(k1, kw)
    q = rope(q, cos, sin, pos)
    k1 = rope(k1, cos, sin, pos)
    if cache is not None:
        cache.store_step(layer, pos, k1, v1)
        K, V = cache.get(layer, pos + 1)
    else:
        Kcache[layer][:, pos] = k1
        Vcache[layer][:, pos] = v1
        K, V = Kcache[layer][:, :pos + 1], Vcache[layer][:, :pos + 1]
    o = gqa_decode_attn(q, K, V, scale)
    o = o.to(w[prefix + "self_attn.o_proj.weight"].dtype)
    return F.linear(o.reshape(d), w[prefix + "self_attn.o_proj.weight"], None)


# -- engine -------------------------------------------------------------
class QwenEngine:
    """Working Qwen greedy decoder with KV cache (Qwen3 + Qwen2.5).

    Takes explicit kwargs (``model``, ``dtype``, ``max_new_tokens``,
    ``max_seq``, ...); QK-norm (Qwen3) vs biases (Qwen2.5) is detected
    from the loaded state dict.
    """

    # End tokens (same ids in Qwen2.5 and Qwen3):
    # 151645 <|im_end|>, 151643 <|endoftext|>.
    STOP_IDS = frozenset((151645, 151643))

    def __init__(self, weights_path=None, device="cuda:0",
                 repo_id=None, model=None, max_len=None, max_seq=None,
                 max_new_tokens=None, dtype=None, kernels=None, **_ignored):
        repo_id = repo_id or model or MODEL_ID
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.max_len = int(max_len or max_seq or MAXN)
        if max_new_tokens is not None:
            self.default_max_tokens = int(max_new_tokens)
        cfg = load_hf_config(repo_id)
        self.nlayers = cfg["num_hidden_layers"]
        self.H = cfg["num_attention_heads"]
        self.Hk = cfg["num_key_value_heads"]
        self.hidden = cfg["hidden_size"]
        self.dh = int(cfg.get("head_dim") or self.hidden // self.H)
        self.scale = 1 / math.sqrt(self.dh)
        self.eps = cfg["rms_norm_eps"]
        self.w = load_weights(weights_path, device=device, repo_id=repo_id)
        # Qwen3 QK-norm (absent on Qwen2.5): detected from the weights.
        self.qk_norm = any(
            k.endswith("self_attn.q_norm.weight") for k in self.w)
        n = min(2048, cfg["max_position_embeddings"])
        theta = cfg.get("rope_theta") or 1000000.0
        self.cos, self.sin = build_cos_sin(n, self.dh, theta, device)
        print("llm layers=%d H=%d/%d dh=%d eps=%g%s" % (
            self.nlayers, self.H, self.Hk, self.dh, self.eps,
            " qk-norm" if self.qk_norm else ""), flush=True)

    def _qk(self, prefix):
        """(q_norm_w, k_norm_w) for a layer, or None (Qwen2.5)."""
        qw = self.w.get(prefix + "self_attn.q_norm.weight")
        kw = self.w.get(prefix + "self_attn.k_norm.weight")
        return (qw, kw) if qw is not None and kw is not None else None

    # -- layers -------------------------------------------------------
    def _new_cache(self):
        return KVCache(self.nlayers, self.Hk, self.max_len, self.dh,
                       self.device)

    def _layer_prefill(self, x, i, cache):
        p = "model.layers.%d." % i
        w = self.w
        h = rmsnorm(x, w[p + "input_layernorm.weight"], self.eps)
        x = x + prefill_attention(h, w, p, self.cos, self.sin, self.H,
                                  self.Hk, self.dh, cache=cache, layer=i,
                                  qk_norm=self._qk(p))
        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], self.eps)
        return x + mlp_forward(h, w, p)

    def _layer_decode(self, x, i, n, cache):
        p = "model.layers.%d." % i
        w = self.w
        h = rmsnorm(x, w[p + "input_layernorm.weight"], self.eps)
        x = x + decode_attention_step(h, w, p, self.cos, self.sin, n,
                                      self.H, self.Hk, self.dh, self.scale,
                                      cache=cache, layer=i,
                                      qk_norm=self._qk(p))
        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], self.eps)
        return x + mlp_forward(h, w, p)

    def _dec_layer_wrap(self, i, n, x, Kcache, Vcache):
        # Legacy wrapper: accepts either a KVCache or raw tensor lists.
        if isinstance(Kcache, KVCache):
            return self._layer_decode(x, i, n, Kcache)
        p = "model.layers.%d." % i
        w = self.w
        h = rmsnorm(x, w[p + "input_layernorm.weight"], self.eps)
        x = x + decode_attention_step(h, w, p, self.cos, self.sin, n,
                                      self.H, self.Hk, self.dh, self.scale,
                                      Kcache=Kcache, Vcache=Vcache, layer=i,
                                      qk_norm=self._qk(p))
        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], self.eps)
        return x + mlp_forward(h, w, p)

    def _logits(self, x):
        h = rmsnorm(x.reshape(1, -1), self.w["model.norm.weight"], self.eps)
        return F.linear(h.reshape(-1), lm_head_weight(self.w))

    # -- generation ---------------------------------------------------
    @torch.no_grad()
    def generate(self, ids, max_new_tokens=32):
        """Greedy decode -> dict(ids, ttft, tps_e2e, decode_tps)."""
        dev = self.device
        d = self.H * self.dh
        cache = self._new_cache()
        x = F.embedding(torch.tensor(ids, device=dev),
                        self.w["model.embed_tokens.weight"])
        t0 = time.perf_counter()
        for i in range(self.nlayers):
            x = self._layer_prefill(x, i, cache)
        n0 = len(ids)
        nxt = self._logits(x[-1]).argmax().item()
        ttft = time.perf_counter() - t0
        out = [] if nxt in self.STOP_IDS else [nxt]
        t1 = time.perf_counter()
        for n in range(n0, n0 + max_new_tokens - 1):
            if not out:
                break  # first token was EOS; nothing to continue from
            e = F.embedding(torch.tensor([nxt], device=dev),
                            self.w["model.embed_tokens.weight"]).reshape(-1)
            x = e
            for i in range(self.nlayers):
                x = self._layer_decode(x, i, n, cache)
            nxt = self._logits(x).argmax().item()
            if nxt in self.STOP_IDS:
                break
            out.append(nxt)
        dt = time.perf_counter() - t1
        return {"ids": out, "ttft": ttft,
                "tps_e2e": len(out) / max(ttft + dt, 1e-9),
                "decode_tps": (len(out) - 1) / max(dt, 1e-9)
                if len(out) > 1 else 0.0}

    @torch.no_grad()
    def generate_stream(self, ids, max_new_tokens=32):
        """Yields ``(token_id, ttft_or_None)`` as produced (for TTS)."""
        dev = self.device
        d = self.H * self.dh
        cache = self._new_cache()
        x = F.embedding(torch.tensor(ids, device=dev),
                        self.w["model.embed_tokens.weight"])
        for i in range(self.nlayers):
            x = self._layer_prefill(x, i, cache)
        n0 = len(ids)
        t0 = time.perf_counter()
        nxt = self._logits(x[-1]).argmax().item()
        ttft = time.perf_counter() - t0
        if nxt in self.STOP_IDS:
            return
        yield nxt, ttft
        for n in range(n0, n0 + max_new_tokens - 1):
            e = F.embedding(torch.tensor([nxt], device=dev),
                            self.w["model.embed_tokens.weight"]).reshape(-1)
            x = e
            for i in range(self.nlayers):
                x = self._layer_decode(x, i, n, cache)
            nxt = self._logits(x).argmax().item()
            if nxt in self.STOP_IDS:
                return
            yield nxt, None


# Backwards-compatible alias for the source-engine name.
TinyLLM = QwenEngine
QwenModel = QwenEngine
