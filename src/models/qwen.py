"""Complete Qwen model (all 28 layers) importing fused Triton kernel.

Single file per spec: 1 fused layer x28 loops, batching enabled, KV cache room, VRAM check.
Imports from src.models.triton_kernels.qwen_fused (single file Triton kernels).
Static test against PyTorch reference in src.models.pytorch.qwen.
"""
import glob
import math
import os
import time

import torch
import torch.nn.functional as F

from src.models.triton_kernels.qwen_fused import (
    qwen_fused_decode_layer,
    build_cos_sin,
    estimate_kv_cache_mb,
    rmsnorm,
)
from src.models.pytorch.qwen import qwen_decode_layer_torch
from src.models.runtime.memory import check_budget
from src.models.runtime.device import allocated_mb, max_allocated_mb

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

def snapshot_path(repo_id=MODEL_ID, local_dir=None):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id, allow_patterns=["*.safetensors"], local_dir=local_dir)

def load_weights(weights_path=None, device="cuda:0", repo_id=MODEL_ID):
    from safetensors.torch import load_file
    if weights_path is None:
        weights_path = snapshot_path(repo_id)
    if os.path.isdir(weights_path):
        files = sorted(glob.glob(os.path.join(weights_path, "*.safetensors")))
        w={}
        for f in files:
            w.update(load_file(f, device=device))
        return dict(w)
    return dict(load_file(weights_path, device=device))

def load_hf_config(repo_id=MODEL_ID):
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(repo_id)
        hidden = cfg.hidden_size; heads = cfg.num_attention_heads
        return {
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "hidden_size": hidden,
            "head_dim": int(getattr(cfg, "head_dim", None) or hidden//heads),
            "rms_norm_eps": float(cfg.rms_norm_eps),
            "max_position_embeddings": cfg.max_position_embeddings,
            "rope_theta": float(getattr(cfg,"rope_theta",None) or 1000000.0),
        }
    except Exception:
        return dict(FALLBACK_DIMS)

def lm_head_weight(w):
    return w["lm_head.weight"] if "lm_head.weight" in w else w["model.embed_tokens.weight"]

class KVCacheBatched:
    """Batched KV cache: [B, Hk, max_len, Dh] per layer, with lengths [B]."""
    def __init__(self, nlayers, B, Hk, max_len, Dh, device="cuda:0", dtype=torch.float16):
        self.nlayers = nlayers
        self.B = B
        self.Hk = Hk
        self.max_len = max_len
        self.Dh = Dh
        self.device = device
        self.dtype = dtype
        # VRAM check
        est = estimate_kv_cache_mb(B, nlayers, Hk, max_len, Dh, bytes_per=2)
        try:
            check_budget(est, budget_mb=4000, headroom_mb=400, what="KVCacheBatched")
        except Exception as e:
            # don't crash import, just warn - still allocate but caller can catch
            print(f"[KVCache] VRAM warning: {e}", flush=True)
        self.K = [torch.empty(B, Hk, max_len, Dh, device=device, dtype=dtype) for _ in range(nlayers)]
        self.V = [torch.empty(B, Hk, max_len, Dh, device=device, dtype=dtype) for _ in range(nlayers)]
        self.lengths = torch.zeros(B, dtype=torch.long, device="cpu")

    def set_len(self, Bpos):
        # Bpos: [B] pos indices
        if isinstance(Bpos, int):
            self.lengths[:] = Bpos+1
        else:
            for b, p in enumerate(Bpos):
                v = int(p.item()) if isinstance(p, torch.Tensor) else int(p)
                self.lengths[b] = v+1

class QwenFused(torch.nn.Module):
    """Complete Qwen model: 28x fused decode layer, batched, KV cache.

    Usage: model = QwenFused(device="cuda:0", batch_size=2); model.generate(ids_batch)
    """
    STOP_IDS = frozenset((151645, 151643))
    def __init__(self, weights_path=None, device="cuda:0", repo_id=None, model=None, max_len=None, max_seq=None, batch_size=1, dtype=torch.float16, **_ignored):
        super().__init__()
        repo_id = repo_id or model or MODEL_ID
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.dtype = dtype
        self.max_len = int(max_len or max_seq or MAXN)
        self.batch_size = int(batch_size)
        cfg = load_hf_config(repo_id)
        self.nlayers = cfg["num_hidden_layers"]
        self.H = cfg["num_attention_heads"]
        self.Hk = cfg["num_key_value_heads"]
        self.hidden = cfg["hidden_size"]
        self.Dh = int(cfg.get("head_dim") or self.hidden // self.H)
        self.scale = 1 / math.sqrt(self.Dh)
        self.eps = cfg["rms_norm_eps"]
        self.w = load_weights(weights_path, device=device, repo_id=repo_id)
        self.qk_norm = any(k.endswith("self_attn.q_norm.weight") for k in self.w)
        n = min(2048, cfg["max_position_embeddings"])
        theta = cfg.get("rope_theta") or 1000000.0
        self.cos, self.sin = build_cos_sin(n, self.Dh, theta, device=device, dtype=torch.float32)
        print(f"[QwenFused] layers={self.nlayers} H={self.H}/{self.Hk} Dh={self.Dh} hidden={self.hidden} batch={self.batch_size} max_len={self.max_len}{' qk-norm' if self.qk_norm else ''}", flush=True)

    def _qk(self, prefix):
        qw = self.w.get(prefix+"self_attn.q_norm.weight")
        kw = self.w.get(prefix+"self_attn.k_norm.weight")
        return (qw, kw) if qw is not None and kw is not None else None

    def _layer_prefill(self, x, i, Kcache, Vcache):
        # x [T, hidden] T variable, Kcache/Vcache per layer [Hk, max_len, Dh] (non-batched for prefill)
        # we reuse qwen_fused_layer_prefill
        from src.models.triton_kernels.qwen_fused import qwen_fused_layer_prefill
        p = f"model.layers.{i}."
        # need per-layer cache as [Hk, max_len, Dh] but our batched cache is [B,Hk,max_len,Dh]
        # for prefill B=1 we can adapt
        # Cache handling for prefill: we store via qwen_fused_layer_prefill with temp cache
        # Simpler: inline prefill here without fused wrapper to avoid shape mismatch
        h = rmsnorm(x, self.w[p+"input_layernorm.weight"], self.eps)
        q = F.linear(h, self.w[p+"self_attn.q_proj.weight"], self.w.get(p+"self_attn.q_proj.bias"))
        k = F.linear(h, self.w[p+"self_attn.k_proj.weight"], self.w.get(p+"self_attn.k_proj.bias"))
        v = F.linear(h, self.w[p+"self_attn.v_proj.weight"], self.w.get(p+"self_attn.v_proj.bias"))
        T = x.shape[0]
        if self.qk_norm:
            qw, kw = self._qk(p)
            if qw is not None:
                q = rmsnorm(q.reshape(T, self.H, self.Dh), qw).reshape(T, self.H*self.Dh)
                k = rmsnorm(k.reshape(T, self.Hk, self.Dh), kw).reshape(T, self.Hk*self.Dh)
        from src.models.triton_kernels.qwen_fused import rope_batched
        qr = rope_batched(q.reshape(T, self.H, self.Dh), self.cos, self.sin, 0).reshape(T, self.H*self.Dh)
        kr = rope_batched(k.reshape(T, self.Hk, self.Dh), self.cos, self.sin, 0).reshape(T, self.Hk*self.Dh)
        if Kcache is not None:
            Kcache[i][:, :T] = kr.reshape(T, self.Hk, self.Dh).transpose(0,1)
            Vcache[i][:, :T] = v.reshape(T, self.Hk, self.Dh).transpose(0,1)
        q4 = qr.view(T, self.H, self.Dh).transpose(0,1).unsqueeze(0)
        k4 = kr.view(T, self.Hk, self.Dh).transpose(0,1).unsqueeze(0).repeat_interleave(self.H//self.Hk, dim=1)
        v4 = v.view(T, self.Hk, self.Dh).transpose(0,1).unsqueeze(0).repeat_interleave(self.H//self.Hk, dim=1)
        o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)[0].transpose(0,1).reshape(T, self.H*self.Dh)
        o = F.linear(o, self.w[p+"self_attn.o_proj.weight"], None)
        x = x + o
        h2 = rmsnorm(x, self.w[p+"post_attention_layernorm.weight"], self.eps)
        gate = F.linear(h2, self.w[p+"mlp.gate_proj.weight"], None)
        up = F.linear(h2, self.w[p+"mlp.up_proj.weight"], None)
        from src.models.triton_kernels.qwen_fused import swiglu
        return x + F.linear(swiglu(gate, up), self.w[p+"mlp.down_proj.weight"], None)

    def _layer_decode_batched(self, x, i, Kcache, Vcache, pos):
        p = f"model.layers.{i}."
        return qwen_fused_decode_layer(x, Kcache[i], Vcache[i], pos, self.cos, self.sin, self.w, p, self.H, self.Hk, self.Dh, self.scale, self.eps, self._qk(p))

    def _logits(self, x):
        # x [B, hidden]
        h = rmsnorm(x, self.w["model.norm.weight"], self.eps)
        return F.linear(h, lm_head_weight(self.w))

    @torch.no_grad()
    def generate(self, ids_batch, max_new_tokens=32):
        """Batched greedy decode. ids_batch: List[List[int]] or [B,T] tensor, B=batch_size.
        Returns dict with ids [B, gen_len]"""
        if isinstance(ids_batch, torch.Tensor):
            # [B, T]
            B, T0 = ids_batch.shape
            ids_list = [ids_batch[b].tolist() for b in range(B)]
        else:
            # List[List[int]]
            if isinstance(ids_batch[0], int):
                ids_batch = [ids_batch]
            B = len(ids_batch)
            ids_list = ids_batch
            T0 = max(len(x) for x in ids_list)
            # pad to T0 for prefill (we handle per-batch prefill separately)
        if T0 + int(max_new_tokens) > int(self.max_len):
            raise ValueError(
                f"prompt {T0} + max_new_tokens {max_new_tokens} exceeds "
                f"max_len {self.max_len}; shorten history/observation")
        dev = self.device
        # Batched KV cache
        est = estimate_kv_cache_mb(B, self.nlayers, self.Hk, self.max_len, self.Dh)
        check_budget(est, budget_mb=4000, what="QwenFused generate")
        Kcache = [torch.empty(B, self.Hk, self.max_len, self.Dh, device=dev, dtype=self.w["model.embed_tokens.weight"].dtype) for _ in range(self.nlayers)]
        Vcache = [torch.empty(B, self.Hk, self.max_len, self.Dh, device=dev, dtype=self.w["model.embed_tokens.weight"].dtype) for _ in range(self.nlayers)]
        # Prefill per batch (different lengths)
        # For simplicity pad ids to max len with 0, but mask? We'll do per-batch prefill loop for variable lengths
        # Actually we can do batched prefill by padding and using same x but lengths differ -> do loop per batch for correctness then merge?
        # Simpler: if all same length, batch prefill; else loop.
        lens = [len(ids) for ids in ids_list]
        max_len_input = max(lens)
        # embed all
        # Use per-batch prefill via python loop then stack hidden?
        # Instead we do sequential prefill per batch but share weights: loop b
        # For batched cache we need to set per-batch positions
        # We'll do prefill for each batch element separately and fill its slice of cache
        out_ids = [ [] for _ in range(B) ]
        # First, handle case where B=1 or all same length: we can batch
        # We'll implement generic: loop per batch for prefill to keep KV correct
        # Keep hidden per batch
        hidden_per_batch = []
        for b in range(B):
            ids = ids_list[b]
            x = F.embedding(torch.tensor(ids, device=dev), self.w["model.embed_tokens.weight"])  # [T, hidden]
            # temporary per-layer cache slices for this batch: we write directly into batched cache's b-th slice
            # need temp shape [Hk, max_len, Dh] view
            # Create views that share storage? Instead we pass batched cache and let _layer_prefill handle per-batch?
            # For now use simple per-batch cache loop that writes into Kcache[b]
            # We'll create per-batch K/V lists of shape [Hk, max_len, Dh] that are views onto batched cache
            Kviews = [Kcache[i][b] for i in range(self.nlayers)]
            Vviews = [Vcache[i][b] for i in range(self.nlayers)]
            for i in range(self.nlayers):
                # _layer_prefill expects Kcache[i] as [Hk, max_len, Dh]
                # we pass list of views
                # but our _layer_prefill currently expects batched? Let's inline prefill per batch directly
                p = f"model.layers.{i}."
                h = rmsnorm(x, self.w[p+"input_layernorm.weight"], self.eps)
                q = F.linear(h, self.w[p+"self_attn.q_proj.weight"], self.w.get(p+"self_attn.q_proj.bias"))
                k = F.linear(h, self.w[p+"self_attn.k_proj.weight"], self.w.get(p+"self_attn.k_proj.bias"))
                v = F.linear(h, self.w[p+"self_attn.v_proj.weight"], self.w.get(p+"self_attn.v_proj.bias"))
                T = x.shape[0]
                if self.qk_norm:
                    qw, kw = self._qk(p)
                    if qw is not None:
                        q = rmsnorm(q.reshape(T, self.H, self.Dh), qw).reshape(T, self.H*self.Dh)
                        k = rmsnorm(k.reshape(T, self.Hk, self.Dh), kw).reshape(T, self.Hk*self.Dh)
                from src.models.triton_kernels.qwen_fused import rope_batched
                qr = rope_batched(q.reshape(T, self.H, self.Dh), self.cos, self.sin, 0).reshape(T, self.H*self.Dh)
                kr = rope_batched(k.reshape(T, self.Hk, self.Dh), self.cos, self.sin, 0).reshape(T, self.Hk*self.Dh)
                Kviews[i][:, :T] = kr.reshape(T, self.Hk, self.Dh).transpose(0,1)
                Vviews[i][:, :T] = v.reshape(T, self.Hk, self.Dh).transpose(0,1)
                q4 = qr.view(T, self.H, self.Dh).transpose(0,1).unsqueeze(0)
                k4 = kr.view(T, self.Hk, self.Dh).transpose(0,1).unsqueeze(0).repeat_interleave(self.H//self.Hk, dim=1)
                v4 = v.view(T, self.Hk, self.Dh).transpose(0,1).unsqueeze(0).repeat_interleave(self.H//self.Hk, dim=1)
                o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)[0].transpose(0,1).reshape(T, self.H*self.Dh)
                o = F.linear(o, self.w[p+"self_attn.o_proj.weight"], None)
                x = x + o
                h2 = rmsnorm(x, self.w[p+"post_attention_layernorm.weight"], self.eps)
                gate = F.linear(h2, self.w[p+"mlp.gate_proj.weight"], None)
                up = F.linear(h2, self.w[p+"mlp.up_proj.weight"], None)
                from src.models.triton_kernels.qwen_fused import swiglu
                x = x + F.linear(swiglu(gate, up), self.w[p+"mlp.down_proj.weight"], None)
            hidden_per_batch.append(x[-1])  # [hidden]
        # stack hidden for decode
        x = torch.stack(hidden_per_batch, 0)  # [B, hidden]
        # decode loop batched
        # lengths per batch
        pos_per_batch = lens[:]  # next pos = len(ids)
        # first token after prefill
        logits = self._logits(x)  # [B, vocab]
        nxt = logits.argmax(dim=-1)  # [B]
        for b in range(B):
            nid = int(nxt[b].item())
            if nid not in self.STOP_IDS:
                out_ids[b].append(nid)
            else:
                out_ids[b] = []  # no continue
        # we need to handle batches where first token is EOS -> they finish
        active = [ (out_ids[b] != [] or len(out_ids[b])==0 and int(nxt[b].item()) not in self.STOP_IDS) for b in range(B) ]
        # Actually track finished
        finished = [ int(nxt[b].item()) in self.STOP_IDS for b in range(B) ]
        # decode steps
        t0 = time.perf_counter()
        for step in range(max_new_tokens-1):
            if all(finished):
                break
            # prepare next x: embeddings of nxt for active batches, zero for finished (will be ignored)
            # still need to run fused layer for all B but finished batches will be masked? For simplicity we still run but ignore output
            e = F.embedding(nxt, self.w["model.embed_tokens.weight"])  # [B, hidden]
            # pos as list of ints per batch
            pos = pos_per_batch[:]  # before increment
            # fused layers loop 28
            for i in range(self.nlayers):
                e = self._layer_decode_batched(e, i, Kcache, Vcache, pos)
            x = e
            logits = self._logits(x)
            nxt_new = logits.argmax(dim=-1)
            for b in range(B):
                if finished[b]:
                    continue
                nid = int(nxt_new[b].item())
                if nid in self.STOP_IDS:
                    finished[b]=True
                else:
                    out_ids[b].append(nid)
                # update pos for next iter if not finished
                pos_per_batch[b] += 1
            nxt = nxt_new
        dt = time.perf_counter() - t0
        total_tokens = sum(len(x) for x in out_ids)
        return {"ids": out_ids if B>1 else out_ids[0], "total_tokens": total_tokens, "time_s": dt, "vram_mb": max_allocated_mb()}

    @staticmethod
    def test_against_torch(batch_size=2, max_len=64, atol=1e-2, rtol=1e-2):
        """Static parity test: fused layer vs torch reference + VRAM check."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[QwenFused.test] device={device} batch={batch_size}", flush=True)
        # check VRAM
        est = estimate_kv_cache_mb(batch_size, 28, 8, 512, 128)
        print(f"  estimated KV cache {est:.1f} MB, allocated {allocated_mb():.1f} MB", flush=True)
        try:
            check_budget(est, budget_mb=4000, what="test_qwen")
            print("  VRAM budget OK", flush=True)
        except Exception as e:
            print(f"  VRAM budget fail: {e}", flush=True)
            return False
        torch.manual_seed(0)
        hidden = 1024; H=16; Hk=8; Dh=128; scale=1/math.sqrt(Dh)
        B=batch_size
        x = torch.randn(B, hidden, device=device, dtype=torch.float16) * 0.5
        Kcache = torch.randn(B, Hk, 512, Dh, device=device, dtype=torch.float16) * 0.5
        Vcache = torch.randn(B, Hk, 512, Dh, device=device, dtype=torch.float16) * 0.5
        def rand_w(*shape):
            return (torch.randn(*shape, device=device, dtype=torch.float32) * 0.02).to(torch.float16)
        w = {
            "model.layers.0.input_layernorm.weight": torch.ones(hidden, device=device, dtype=torch.float16),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(hidden, device=device, dtype=torch.float16),
            "model.layers.0.self_attn.q_proj.weight": rand_w(H*Dh, hidden),
            "model.layers.0.self_attn.k_proj.weight": rand_w(Hk*Dh, hidden),
            "model.layers.0.self_attn.v_proj.weight": rand_w(Hk*Dh, hidden),
            "model.layers.0.self_attn.o_proj.weight": rand_w(hidden, H*Dh),
            "model.layers.0.mlp.gate_proj.weight": rand_w(hidden*2, hidden),
            "model.layers.0.mlp.up_proj.weight": rand_w(hidden*2, hidden),
            "model.layers.0.mlp.down_proj.weight": rand_w(hidden, hidden*2),
        }
        cos, sin = build_cos_sin(512, Dh, device=device, dtype=torch.float32)
        # fused
        Kf = Kcache.clone(); Vf = Vcache.clone()
        Kt = Kcache.clone(); Vt = Vcache.clone()
        pos=10
        try:
            out_fused = qwen_fused_decode_layer(x.clone(), Kf, Vf, pos, cos, sin, w, "model.layers.0.", H, Hk, Dh, scale)
            out_torch = qwen_decode_layer_torch(x.clone(), Kt, Vt, pos, cos, sin, w, "model.layers.0.", H, Hk, Dh, scale)
            err = (out_fused.float() - out_torch.float()).abs().max().item()
            mean_err = (out_fused.float() - out_torch.float()).abs().mean().item()
            print(f"  layer parity max_err={err:.2e} mean={mean_err:.2e}", flush=True)
            ok = err < 5e-2  # fp16 tolerance
            print(f"  {'PASS' if ok else 'FAIL'}", flush=True)
            return ok
        except Exception as e:
            print(f"  test error: {e}", flush=True)
            import traceback; traceback.print_exc()
            return False

# For backward compat
TinyLLM = QwenFused
QwenEngine = QwenFused

if __name__ == "__main__":
    QwenFused.test_against_torch()
