"""Parity: LLM Triton kernels vs torch reference (no weights needed).

Ported from VOICE/test_llm_kernels.py onto the new `triton_kernels`
layout. Skips (does not fail) when torch/triton/cuda or the ported
kernels are unavailable.
"""
import math
import unittest

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


def _load_llm_kernels():
    """Return (rmsnorm, rope, swiglu, gqa_decode_attn) or raise SkipTest."""
    candidates = ("triton_kernels.qwen", "triton_kernels.llm",
                  "triton_kernels.llm_kernels", "triton_kernels")
    last = None
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["*"])
            fns = [getattr(mod, n, None) for n in
                   ("rmsnorm", "rope", "swiglu", "gqa_decode_attn")]
            if all(callable(f) for f in fns):
                return tuple(fns)
            last = f"{name} missing symbols"
        except Exception as exc:
            last = f"{name}: {exc}"
    raise unittest.SkipTest(f"LLM triton kernels not ported yet ({last})")


def _need_cuda(tc):
    if not _HAS_TORCH:
        tc.skipTest("torch missing")
    if not torch.cuda.is_available():
        tc.skipTest("needs cuda")


class TestLlmKernels(unittest.TestCase):
    def test_rmsnorm(self):
        _need_cuda(self)
        rmsnorm, _, _, _ = _load_llm_kernels()
        torch.manual_seed(0)
        dev = "cuda:0"
        x = torch.randn(4, 896, device=dev)
        w = torch.randn(896, device=dev)
        got, ref = rmsnorm(x, w), F.rms_norm(x, (896,), w, 1e-6)
        err = (got.float() - ref.float()).abs().max().item()
        self.assertLess(err, 1e-5, f"rmsnorm maxerr={err:.2e}")

    def test_rope(self):
        _need_cuda(self)
        _, rope, _, _ = _load_llm_kernels()
        torch.manual_seed(0)
        dev = "cuda:0"
        dh, pos, mp = 64, 7, 128
        xr = torch.randn(3, dh, device=dev)
        inv = 1.0 / (1000000 ** (torch.arange(0, dh, 2, device=dev).float() / dh))
        ang = pos * inv
        cos = torch.cos(ang).unsqueeze(0).expand(mp, -1).contiguous()
        sin = torch.sin(ang).unsqueeze(0).expand(mp, -1).contiguous()
        x0, x1 = xr[..., :dh // 2], xr[..., dh // 2:]
        ref = torch.cat([x0 * cos[pos] - x1 * sin[pos],
                         x0 * sin[pos] + x1 * cos[pos]], -1)
        err = (rope(xr, cos, sin, pos).float() - ref.float()).abs().max().item()
        self.assertLess(err, 1e-5, f"rope maxerr={err:.2e}")

    def test_swiglu(self):
        _need_cuda(self)
        _, _, swiglu, _ = _load_llm_kernels()
        torch.manual_seed(0)
        dev = "cuda:0"
        g = torch.randn(4, 4864, device=dev)
        u = torch.randn(4, 4864, device=dev)
        err = (swiglu(g, u).float() - (F.silu(g) * u).float()).abs().max().item()
        self.assertLess(err, 1e-5, f"swiglu maxerr={err:.2e}")

    def test_gqa_decode_attn(self):
        _need_cuda(self)
        _, _, _, gqa_decode_attn = _load_llm_kernels()
        torch.manual_seed(0)
        dev = "cuda:0"
        for Hq, Hk, n in ((14, 2, 9), (14, 2, 300), (8, 8, 64)):
            dh = 64
            q = torch.randn(Hq, dh, device=dev)
            K = torch.randn(Hk, n, dh, device=dev)
            V = torch.randn(Hk, n, dh, device=dev)
            outs = []
            for h in range(Hq):
                s = (q[h] @ K[h // (Hq // Hk)].T) * (1 / math.sqrt(dh))
                outs.append(torch.softmax(s, -1) @ V[h // (Hq // Hk)])
            ref = torch.stack(outs)
            got = gqa_decode_attn(q, K, V, 1 / math.sqrt(dh))
            err = (got.float() - ref.float()).abs().max().item()
            self.assertLess(err, 1e-4, f"gqa Hq={Hq} Hk={Hk} n={n} err={err:.2e}")


if __name__ == "__main__":
    unittest.main()
