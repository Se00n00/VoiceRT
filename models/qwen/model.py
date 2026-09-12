"""Qwen engine: KV-cached greedy decode with prefill + streaming generate.

Ported from ``VOICE/llm_engine.py`` (``TinyLLM``). Prefill uses SDPA
(compute-bound); decode steps use the fused GQA kernel with a growing
:class:`KVCache`. ``transformers`` / ``huggingface_hub`` are imported
lazily so this module imports without them.
"""
import math
import time

import torch
import torch.nn.functional as F

from models.qwen.attention import decode_attention_step, prefill_attention
from models.qwen.kernels import rmsnorm
from models.qwen.kv_cache import KVCache
from models.qwen.mlp import mlp_forward
from models.qwen.rope import build_cos_sin
from models.qwen.weights import MODEL_ID, load_config, load_weights

__all__ = ["MAXN", "QwenEngine", "TinyLLM"]

MAXN = 512


class QwenEngine:
    """Working Qwen2.5 greedy decoder with KV cache."""

    def __init__(self, weights_path=None, device="cuda:0",
                 repo_id=MODEL_ID, max_len=MAXN):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.max_len = int(max_len)
        cfg = load_config(repo_id)
        self.nlayers = cfg["num_hidden_layers"]
        self.H = cfg["num_attention_heads"]
        self.Hk = cfg["num_key_value_heads"]
        self.hidden = cfg["hidden_size"]
        self.dh = self.hidden // self.H
        self.scale = 1 / math.sqrt(self.dh)
        self.eps = cfg["rms_norm_eps"]
        self.w = load_weights(weights_path, device=device, repo_id=repo_id)
        n = min(2048, cfg["max_position_embeddings"])
        # Measured: HF inv_freq == theta=1e6 exactly; trust measurement.
        theta = cfg.get("rope_theta") or 1000000.0
        self.cos, self.sin = build_cos_sin(n, self.dh, theta, device)
        print("llm layers=%d H=%d/%d dh=%d eps=%g" % (
            self.nlayers, self.H, self.Hk, self.dh, self.eps), flush=True)

    # -- layers -------------------------------------------------------
    def _new_cache(self):
        return KVCache(self.nlayers, self.Hk, self.max_len, self.dh,
                       self.device)

    def _layer_prefill(self, x, i, cache):
        p = "model.layers.%d." % i
        w = self.w
        h = rmsnorm(x, w[p + "input_layernorm.weight"], self.eps)
        x = x + prefill_attention(h, w, p, self.cos, self.sin, self.H,
                                  self.Hk, self.dh, cache=cache, layer=i)
        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], self.eps)
        return x + mlp_forward(h, w, p)

    def _layer_decode(self, x, i, n, cache):
        p = "model.layers.%d." % i
        w = self.w
        h = rmsnorm(x, w[p + "input_layernorm.weight"], self.eps)
        x = x + decode_attention_step(h, w, p, self.cos, self.sin, n,
                                      self.H, self.Hk, self.dh, self.scale,
                                      cache=cache, layer=i)
        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], self.eps)
        return x + mlp_forward(h, w, p)

    def _dec_layer_wrap(self, i, n, x, Kcache, Vcache):
        # Legacy wrapper kept for source-engine call sites: accepts either a
        # KVCache or raw (Kcache, Vcache) tensor lists.
        if isinstance(Kcache, KVCache):
            return self._layer_decode(x, i, n, Kcache)
        p = "model.layers.%d." % i
        w = self.w
        h = rmsnorm(x, w[p + "input_layernorm.weight"], self.eps)
        x = x + decode_attention_step(h, w, p, self.cos, self.sin, n,
                                      self.H, self.Hk, self.dh, self.scale,
                                      Kcache=Kcache, Vcache=Vcache, layer=i)
        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], self.eps)
        return x + mlp_forward(h, w, p)

    def _logits(self, x):
        from models.qwen.weights import lm_head_weight
        h = rmsnorm(x.reshape(1, -1), self.w["model.norm.weight"], self.eps)
        return F.linear(h.reshape(-1), lm_head_weight(self.w))

    # -- generation ---------------------------------------------------
    # Qwen2.5 end tokens: 151645 <|im_end|> (eos_token_id), 151643
    # <|endoftext|>. Without stopping on these, greedy decode blows past
    # the answer and role-plays the rest of the chat template.
    STOP_IDS = frozenset((151645, 151643))

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
                            self.w["model.embed_tokens.weight"]).reshape(d)
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
                            self.w["model.embed_tokens.weight"]).reshape(d)
            x = e
            for i in range(self.nlayers):
                x = self._layer_decode(x, i, n, cache)
            nxt = self._logits(x).argmax().item()
            if nxt in self.STOP_IDS:
                return
            yield nxt, None


# Backwards-compatible alias for the source-engine name.
TinyLLM = QwenEngine
