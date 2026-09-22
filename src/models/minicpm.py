"""MiniCPM5-1B Q4_K_M runner: 24-layer Llama decode over packed weights.

Loads the GGUF file once (packed u8 stays on CUDA, ~651MB), reuses the
repo's dtype-agnostic pieces (rmsnorm, rope, swiglu, gqa_decode_attn) and
the fused Q4_K/Q6_K GEMV kernels from
:mod:`src.models.triton_kernels.minicpm_q4k` for every matmul:

- decode (per token): embed gather + dequant of input rows only, then
  24 x fused-GEMV layers with fp16 KV cache, argmax sampling.
- prefill: per-layer transient dequant + batched torch matmuls/SDPA
  (one layer hot at a time, ~230MB transient), KV filled in place.

Matches the :class:`QwenFused` generate/stream contract so
:class:`LlmModel` can select it via ``LlmConfig.backend="minicpm_q4k"``.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.models.gguf import read_gguf
from src.models.runtime.device import max_allocated_mb
from src.models.runtime.memory import check_budget
from src.models.triton_kernels.minicpm_q4k import (
    Q6K_BYTES,
    Q4K_BYTES,
    dequantize_q4_k_torch,
    dequantize_q6_k_torch,
    estimate_q4k_mb,
    q4k_gemv,
    q6k_gemv,
)

__all__ = ["MiniCPMFused", "resolve_gguf", "MiniCPMConfig"]


def _gqa_decode_torch(q, K, V, scale):
    """Exact torch GQA decode (CPU path / parity reference).

    q [Hq,D], K/V [Hk,N,D] -> O [Hq,D]. Hq % Hk == 0.
    """
    Hq, D = q.shape
    Hk, N, _ = K.shape
    assert Hq % Hk == 0
    g = Hq // Hk
    out = torch.empty(Hq, D, dtype=torch.float32)
    qf = q.to(torch.float32)
    for h in range(Hq):
        kh = h // g
        s = (qf[h] * K[kh].to(torch.float32)).sum(-1) * scale
        p = torch.softmax(s, dim=-1)
        out[h] = (p.unsqueeze(-1) * V[kh].to(torch.float32)).sum(0)
    return out.to(q.dtype)


def resolve_gguf(path=None):
    """Find the Q4_K_M file: explicit path, or HF cache download."""
    if path and os.path.isfile(path):
        return path
    if path and os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.endswith(".gguf"):
                return os.path.join(path, f)
        raise FileNotFoundError(f"no .gguf in {path}")
    from huggingface_hub import snapshot_download

    d = snapshot_download("openbmb/MiniCPM5-1B-GGUF", allow_patterns=["*Q4_K_M*"])
    for f in sorted(os.listdir(d)):
        if f.endswith(".gguf"):
            return os.path.join(d, f)
    raise FileNotFoundError("no Q4_K_M gguf in HF cache")


class MiniCPMConfig:
    """Dims resolved from GGUF meta, with known-good fallbacks."""

    def __init__(self, meta):
        g = lambda k, d: meta.get(k, d)  # noqa: E731
        self.hidden = int(g("llama.embedding_length", 1536))
        self.ff = int(g("llama.feed_forward_length", 4608))
        self.nlayers = int(g("llama.block_count", 24))
        self.H = int(g("llama.attention.head_count", 16))
        self.Hk = int(g("llama.attention.head_count_kv", 2))
        self.Dh = int(g("llama.attention.key_length", 128))
        self.vocab = int(g("llama.vocab_size", 130560))
        self.theta = float(g("llama.rope.freq_base", 5000000.0))
        self.eps = float(g("llama.attention.layer_norm_rms_epsilon", 1e-6))
        self.eos = 1


class MiniCPMFused(torch.nn.Module):
    """Q4_K_M MiniCPM5-1B: packed weights + fused decode, fp16 KV."""

    def __init__(self, gguf_path=None, device="cuda:0", model="openbmb/MiniCPM5-1B",
                 max_len=1024, max_new_tokens=64, layers=None, **_):
        super().__init__()
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model_id = model
        self.max_len = int(max_len)
        self.max_new_tokens = int(max_new_tokens)
        parsed = read_gguf(resolve_gguf(gguf_path))
        self.cfg = MiniCPMConfig(parsed["meta"])
        self.eos_id = self.cfg.eos
        self.STOP_IDS = (self.eos_id,)
        # layers: None = all (generate path); subset = light test loads
        self._only_layers = None if layers is None else set(int(i) for i in layers)
        self._load(parsed)
        print(f"[MiniCPMFused] device={device} layers={self.cfg.nlayers} "
              f"packed={self.packed_mb:.0f}MB", flush=True)

    # -- load ----------------------------------------------------------
    def _mmap(self, parsed):
        import mmap

        f = open(parsed["path"], "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._mm_file = f  # keep alive with the mapping
        return mm

    def _row_bytes(self, mm, info):
        # GGUF dims are reversed: rows = last dim, K = first dim product.
        dims = info["shape"]
        rows = dims[-1]
        k = info["nelem"] // rows
        raw = mm[info["offset"]:info["offset"] + info["nbytes"]]
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, -1).copy()
        return torch.from_numpy(arr), rows, k

    def _load(self, parsed):
        from src.models.triton_kernels.qwen_fused import build_cos_sin

        mm = self._mmap(parsed)
        dev = self.device
        self.w = {}
        self._types = {}
        self._sc = {}
        n_q4 = n_q6 = n_f16 = 0
        keep = self._only_layers
        for t in parsed["tensors"]:
            name, typ = t["name"], t["type"]
            if name.startswith("blk."):
                try:
                    li = int(name.split(".")[1])
                except Exception:
                    li = -1
                if keep is not None and li not in keep:
                    continue
            if name in ("token_embd.weight", "output.weight", "output_norm.weight") \
                    and keep is not None:
                # parity tests need norms + head; embeddings only for generate
                if name == "token_embd.weight":
                    continue
            if typ in ("Q4_K", "Q6_K"):
                arr, _, _ = self._row_bytes(mm, t)
                cuda_arr = arr.to(dev)
                self.w[name] = cuda_arr
                self._types[name] = typ
                try:
                    from src.models.triton_kernels.minicpm_q4k import (
                        _q4k_scales, _q6k_scales)
                    if typ == "Q4_K":
                        self._sc[name] = _q4k_scales(cuda_arr)
                    else:
                        self._sc[name] = _q6k_scales(cuda_arr)
                except Exception:
                    pass
                n_q4 += t["nelem"] if typ == "Q4_K" else 0
                n_q6 += t["nelem"] if typ == "Q6_K" else 0
            elif typ == "F32" and name.endswith(".weight"):
                arr = np.frombuffer(
                    mm[t["offset"]:t["offset"] + t["nbytes"]],
                    dtype=np.float32).copy()
                ten = torch.from_numpy(arr).reshape(t["shape"][::-1])
                # norms are [hidden] vectors; store fp16 on device
                self.w[name] = ten.to(dev).to(torch.float16).reshape(-1)
                n_f16 += ten.numel()
            # else: ignore (rope factors etc. recomputed)
        self.packed_mb = (sum(v.numel() for v in self.w.values()
                              if v.dtype == torch.uint8) / (1024 ** 2))
        check_budget(self.packed_mb + 50.0, budget_mb=4000,
                     what="MiniCPMFused weights")
        cfg = self.cfg
        cos, sin = build_cos_sin(self.max_len, cfg.Dh, theta=cfg.theta,
                                 device=dev, dtype=torch.float32)
        self.cos, self.sin = cos, sin
        self.scale = 1.0 / math.sqrt(cfg.Dh)
        # host d constants unused (kernels read per-block scales); kept for parity
        _ = (n_q4, n_q6, n_f16)

    def _dq_mat(self, name):
        """Dequantize one packed matrix to fp16 CUDA, via CPU.

        Prefill runs on tight VRAM (shared 4GB card): dequantizing on CUDA
        needs hundreds of MB transient fp32. RAM is plentiful, so the
        packed rows go D2H, dequantize on CPU, fp16 comes back. One layer
        hot at a time; per-layer PCIe ~230MB is milliseconds.
        """
        fn = (dequantize_q4_k_torch if self._types.get(name) == "Q4_K"
              else dequantize_q6_k_torch)
        blk = self.w[name].to("cpu")
        out = fn(blk)
        del blk
        return out.to(torch.float16).to(self.device)

    def _gemv(self, name, x):
        """GEMV dispatched on the tensor's REAL quant type from the GGUF table.

        Q4_K_M is a per-layer mix (here: 12 layers all-Q4_K, 12 with Q6_K
        v/down). Never assume the type from the op name.
        """
        if self._types.get(name) == "Q6_K":
            return q6k_gemv(self.w[name], x, self._sc.get(name))
        return q4k_gemv(self.w[name], x, self._sc.get(name))

    # -- one decode step -------------------------------------------------
    def _layer_decode(self, x, i, Kcache, Vcache, pos):
        """x [B, D] single token -> [B, D]. pos: int or per-batch list."""
        from src.models.triton_kernels.qwen_fused import rope as _rope
        from src.models.triton_kernels.qwen_fused import rmsnorm
        from src.models.triton_kernels.qwen_fused import swiglu
        if x.is_cuda:
            from src.models.triton_kernels.attention import gqa_decode_attn as _gqa
        else:
            _gqa = _gqa_decode_torch

        cfg = self.cfg
        B = x.shape[0]
        p = f"blk.{i}."
        sc = self._sc.get
        h = rmsnorm(x, self.w[p + "attn_norm.weight"], cfg.eps)
        q = self._gemv(p + "attn_q.weight", h)          # [B, H*Dh]
        k = self._gemv(p + "attn_k.weight", h)          # [B, Hk*Dh]
        v = self._gemv(p + "attn_v.weight", h)          # [B, Hk*Dh]
        outs = []
        for b in range(B):
            pb = pos[b] if isinstance(pos, list) else int(pos)
            qb = _rope(q[b:b + 1].reshape(1, cfg.H, cfg.Dh),
                       self.cos, self.sin, pb)
            kb = _rope(k[b:b + 1].reshape(1, cfg.Hk, cfg.Dh),
                       self.cos, self.sin, pb)
            Kcache[i][b, :, pb] = kb[0]
            Vcache[i][b, :, pb] = v.reshape(B, cfg.Hk, cfg.Dh)[b]
            outs.append(_gqa(
                qb[0], Kcache[i][b][:, :pb + 1], Vcache[i][b][:, :pb + 1],
                self.scale))
        o = torch.stack(outs, 0).reshape(B, cfg.H * cfg.Dh)
        o = self._gemv(p + "attn_output.weight", o)
        x = x + o
        h2 = rmsnorm(x, self.w[p + "ffn_norm.weight"], cfg.eps)
        gate = self._gemv(p + "ffn_gate.weight", h2)
        up = self._gemv(p + "ffn_up.weight", h2)
        return x + self._gemv(p + "ffn_down.weight", swiglu(gate, up))

    def _embed(self, ids):
        """ids [B, T] int -> [B, T, D] fp16 (dequant gathered rows only)."""
        B, T = ids.shape
        flat = ids.reshape(-1)
        rows = self.w["token_embd.weight"][flat]  # [B*T, rowbytes] u8
        return dequantize_q4_k_torch(rows, out_dtype=torch.float16).reshape(B, T, -1)

    def _logits(self, x):
        """x [B, D] -> [B, vocab] fp32."""
        from src.models.triton_kernels.qwen_fused import rmsnorm

        h = rmsnorm(x, self.w["output_norm.weight"], self.cfg.eps)
        return self._gemv("output.weight", h).float()

    # -- prefill (transient per-layer dequant, batched torch) ------------
    def _layer_prefill(self, x, i, Kb, Vb):
        """x [B, T, D], Kb/Vb [B, Hk, max_len, Dh] single-batch views.

        Transient per-layer fp16 dequant (freed at layer end, ~230MB peak).
        """
        from src.models.triton_kernels.qwen_fused import rope_batched
        from src.models.triton_kernels.qwen_fused import rmsnorm
        from src.models.triton_kernels.qwen_fused import swiglu

        cfg = self.cfg
        B, T, D = x.shape
        p = f"blk.{i}."
        dev = self.device
        # transient fp16 copies, freed at layer end
        W = {}
        for name in ("attn_q", "attn_k", "attn_output", "ffn_gate", "ffn_up"):
            W[name] = self._dq_mat(p + name + ".weight")
        W["attn_v"] = self._dq_mat(p + "attn_v.weight")
        W["ffn_down"] = self._dq_mat(p + "ffn_down.weight")
        try:
            h = rmsnorm(x.reshape(-1, D), self.w[p + "attn_norm.weight"], cfg.eps).reshape(B, T, D)
            q = (h @ W["attn_q"].t()).reshape(B, T, cfg.H, cfg.Dh)
            k = (h @ W["attn_k"].t()).reshape(B, T, cfg.Hk, cfg.Dh)
            v = (h @ W["attn_v"].t()).reshape(B, T, cfg.Hk, cfg.Dh)
            qr = rope_batched(q.reshape(B * T, cfg.H, cfg.Dh),
                              self.cos.to(dev), self.sin.to(dev), 0).reshape(B, T, cfg.H, cfg.Dh)
            kr = rope_batched(k.reshape(B * T, cfg.Hk, cfg.Dh),
                              self.cos.to(dev), self.sin.to(dev), 0).reshape(B, T, cfg.Hk, cfg.Dh)
            Kb[:, :, :T] = kr.transpose(1, 2)
            Vb[:, :, :T] = v.transpose(1, 2)
            q4 = qr.transpose(1, 2)
            k4 = kr.transpose(1, 2).repeat_interleave(cfg.H // cfg.Hk, dim=1)
            v4 = v.transpose(1, 2).repeat_interleave(cfg.H // cfg.Hk, dim=1)
            # math SDPA backend: no cuBLAS/cuDNN workspace, survives tight
            # shared-GPU boxes where the first flash allocation would OOM.
            with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False,
                    enable_math=True):
                o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True
                                                   ).transpose(1, 2).reshape(B, T, cfg.H * cfg.Dh)
            o = o @ W["attn_output"].t()
            x = x + o
            h2 = rmsnorm(x.reshape(-1, D), self.w[p + "ffn_norm.weight"], cfg.eps).reshape(B, T, D)
            gate = h2 @ W["ffn_gate"].t()
            up = h2 @ W["ffn_up"].t()
            x = x + (swiglu(gate, up) @ W["ffn_down"].t())
            return x
        finally:
            del W

    # -- generate (QwenFused contract) ------------------------------------
    @torch.no_grad()
    def generate(self, ids_batch, max_new_tokens=32):
        """Greedy decode. ids: List[List[int]] or [B,T]. Returns dict(ids, ...)."""
        if isinstance(ids_batch, torch.Tensor):
            ids_list = [ids_batch[b].tolist() for b in range(ids_batch.shape[0])]
        else:
            ids_list = [ids_batch] if ids_batch and isinstance(ids_batch[0], int) else list(ids_batch)
        B = len(ids_list)
        T0 = max(len(x) for x in ids_list)
        if T0 + int(max_new_tokens) > int(self.max_len):
            raise ValueError(
                f"prompt {T0} + max_new_tokens {max_new_tokens} exceeds "
                f"max_len {self.max_len}")
        dev = self.device
        cfg = self.cfg
        from src.models.triton_kernels.qwen_fused import estimate_kv_cache_mb
        check_budget(estimate_kv_cache_mb(B, cfg.nlayers, cfg.Hk, self.max_len, cfg.Dh),
                     budget_mb=4000, what="MiniCPMFused generate")
        Kcache = [torch.empty(B, cfg.Hk, self.max_len, cfg.Dh, device=dev, dtype=torch.float16)
                  for _ in range(cfg.nlayers)]
        Vcache = [torch.empty(B, cfg.Hk, self.max_len, cfg.Dh, device=dev, dtype=torch.float16)
                  for _ in range(cfg.nlayers)]
        t0 = time.perf_counter()
        out_ids = [[] for _ in range(B)]
        # prefill per batch (prompts may differ in length)
        hidden = []
        for b in range(B):
            ids = torch.tensor([ids_list[b]], device=dev)
            x = self._embed(ids)
            for i in range(cfg.nlayers):
                x = self._layer_prefill(
                    x, i, Kcache[i][b:b + 1], Vcache[i][b:b + 1])
                if i % 4 == 3 and dev.startswith("cuda"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            hidden.append(x[0, -1])
        x = torch.stack(hidden, 0)
        del hidden
        if dev.startswith("cuda"):
            # Prefill peak fragments the cache; return wrung memory before
            # decode so fused launches (and their tiny outputs) always fit.
            try:
                import gc as _gc

                _gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        ttft = time.perf_counter() - t0
        logits = self._logits(x)
        nxt = logits.argmax(dim=-1)
        finished = [False] * B
        for b in range(B):
            nid = int(nxt[b].item())
            if nid in self.STOP_IDS:
                finished[b] = True
            else:
                out_ids[b].append(nid)
        pos = [len(ids) for ids in ids_list]
        t1 = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if all(finished):
                break
            e = self._embed(nxt.reshape(B, 1))
            for i in range(cfg.nlayers):
                e = self._layer_decode(e, i, Kcache, Vcache, pos)
            x = e.reshape(B, -1)
            logits = self._logits(x)
            nxt = logits.argmax(dim=-1)
            for b in range(B):
                if finished[b]:
                    continue
                nid = int(nxt[b].item())
                if nid in self.STOP_IDS:
                    finished[b] = True
                else:
                    out_ids[b].append(nid)
                pos[b] += 1
        dt = time.perf_counter() - t1
        n = max(sum(len(x) for x in out_ids), 1)
        return {"ids": out_ids[0] if B == 1 else out_ids,
                "ttft": ttft, "decode_tps": n / max(dt, 1e-9),
                "vram_mb": max_allocated_mb()}

    def generate_stream(self, ids, max_new_tokens=32):
        """Yield (tok_id, ttft) like QwenFused for LlmModel.stream()."""
        out = self.generate(ids, max_new_tokens)
        ids_out = out["ids"] if isinstance(out["ids"], list) else []
        for tok in ids_out:
            yield int(tok), float(out.get("ttft", 0.0))

    @staticmethod
    def test_layer_parity(layer=0, atol=5e-2, rtol=5e-2):
        """V3 gate: fused decode step vs torch-eager step on REAL weights."""
        from src.models.triton_kernels.qwen_fused import rope as _rope
        from src.models.triton_kernels.attention import gqa_decode_attn
        from src.models.triton_kernels.qwen_fused import rmsnorm
        from src.models.triton_kernels.qwen_fused import swiglu

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[MiniCPMFused.layer-test] device={device} layer={layer}", flush=True)
        if device == "cpu":
            print("  SKIP (needs CUDA for fused path)", flush=True)
            return True
        try:
            run = MiniCPMFused(max_len=128, layers=[layer])
        except Exception as exc:
            print(f"  SKIP ({exc})", flush=True)
            return True
        cfg = run.cfg
        torch.manual_seed(0)
        B, D, pos = 1, cfg.hidden, 10
        x = (torch.randn(B, D, device=device, dtype=torch.float32) * 0.5
             ).to(torch.float16)
        Kcache = [torch.zeros(B, cfg.Hk, run.max_len, cfg.Dh,
                              device=device, dtype=torch.float16)
                  for _ in range(cfg.nlayers)]
        Vcache = [torch.zeros(B, cfg.Hk, run.max_len, cfg.Dh,
                              device=device, dtype=torch.float16)
                  for _ in range(cfg.nlayers)]
        try:
            out_f = run._layer_decode(x.clone(), layer, Kcache, Vcache, pos)
        except Exception as exc:
            print(f"  fused FAIL ({exc})", flush=True)
            import traceback
            traceback.print_exc()
            return False
        # torch-eager reference on dequantized copies
        p = f"blk.{layer}."
        W = {}
        for n_, fn in (("attn_q", dequantize_q4_k_torch),
                       ("attn_k", dequantize_q4_k_torch),
                       ("attn_v", dequantize_q6_k_torch),
                       ("attn_output", dequantize_q4_k_torch),
                       ("ffn_gate", dequantize_q4_k_torch),
                       ("ffn_up", dequantize_q4_k_torch),
                       ("ffn_down", dequantize_q6_k_torch)):
            W[n_] = fn(run.w[p + n_ + ".weight"]).to(torch.float16)
        try:
            h = rmsnorm(x.clone(), run.w[p + "attn_norm.weight"], cfg.eps)
            q = (h @ W["attn_q"].t()).reshape(B, cfg.H, cfg.Dh)
            k = (h @ W["attn_k"].t()).reshape(B, cfg.Hk, cfg.Dh)
            v = (h @ W["attn_v"].t()).reshape(B, cfg.Hk, cfg.Dh)
            qr = _rope(q, run.cos.to(device), run.sin.to(device), pos)
            kr = _rope(k, run.cos.to(device), run.sin.to(device), pos)
            # mirror the fused side exactly: N=pos+1 slots, zeros + current
            Kcache2 = torch.zeros(B, cfg.Hk, pos + 1, cfg.Dh,
                                  device=device, dtype=torch.float16)
            Vcache2 = torch.zeros(B, cfg.Hk, pos + 1, cfg.Dh,
                                  device=device, dtype=torch.float16)
            Kcache2[:, :, pos] = kr
            Vcache2[:, :, pos] = v
            o = torch.stack([gqa_decode_attn(qr[b], Kcache2[b], Vcache2[b],
                                             run.scale) for b in range(B)], 0)
            o = (o.reshape(B, cfg.H * cfg.Dh) @ W["attn_output"].t())
            x2 = x + o
            h2 = rmsnorm(x2, run.w[p + "ffn_norm.weight"], cfg.eps)
            x2 = x2 + ((swiglu(h2 @ W["ffn_gate"].t(), h2 @ W["ffn_up"].t()))
                       @ W["ffn_down"].t())
        except Exception as exc:
            print(f"  torch FAIL ({exc})", flush=True)
            import traceback
            traceback.print_exc()
            return False
        err = (out_f.float() - x2.float()).abs().max().item()
        rel = err / max(x2.float().abs().max().item(), 1e-9)
        ok = err < atol or rel < rtol
        print(f"  layer parity max_err={err:.2e} rel={rel:.2e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        return ok


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true", help="run V3 layer parity gate")
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()
    if args.test:
        raise SystemExit(0 if MiniCPMFused.test_layer_parity(layer=args.layer) else 1)
    print("MiniCPMFused loaded. Use --test for the V3 parity gate.")
