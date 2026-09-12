"""Parity: STT (Whisper) Triton kernels vs torch reference.

Ported from STT/triton/test_kernels.py onto the new `triton_kernels`
layout. Skips when torch/triton/cuda or the ported kernels are missing.
"""
import unittest

try:
    import torch

    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False


def _load_stt_kernels():
    candidates = ("triton_kernels.whisper", "triton_kernels.stt",
                  "triton_kernels.whisper_kernels", "triton_kernels")
    last = None
    for name in candidates:
        try:
            mod = __import__(name, fromlist=["*"])
            fns = [getattr(mod, n, None) for n in
                   ("layernorm", "row_softmax", "decode_attn",
                    "batched_decode_attn")]
            if all(callable(f) for f in fns):
                return tuple(fns)
            last = f"{name} missing symbols"
        except Exception as exc:
            last = f"{name}: {exc}"
    raise unittest.SkipTest(f"STT triton kernels not ported yet ({last})")


def _need_cuda(tc):
    if not _HAS_TORCH:
        tc.skipTest("torch missing")
    if not torch.cuda.is_available():
        tc.skipTest("needs cuda")


class TestWhisperKernels(unittest.TestCase):
    def test_layernorm(self):
        _need_cuda(self)
        layernorm, _, _, _ = _load_stt_kernels()
        torch.manual_seed(0)
        dev = "cuda:0"
        x = torch.randn(32, 512, device=dev)
        w = torch.randn(512, device=dev)
        b = torch.randn(512, device=dev)
        got = layernorm(x, w, b)
        ref = torch.nn.functional.layer_norm(x, (512,), w, b, 1e-5)
        err = (got.float() - ref.float()).abs().max().item()
        self.assertLess(err, 1e-5, f"layernorm maxerr={err:.2e}")

    def test_row_softmax(self):
        _need_cuda(self)
        _, row_softmax, _, _ = _load_stt_kernels()
        torch.manual_seed(0)
        s = torch.randn(8, 1500, device="cuda:0")
        err = (row_softmax(s).float() - torch.softmax(s, -1).float()
               ).abs().max().item()
        self.assertLess(err, 1e-5, f"row_softmax maxerr={err:.2e}")

    def test_decode_attn(self):
        _need_cuda(self)
        _, _, decode_attn, _ = _load_stt_kernels()
        torch.manual_seed(0)
        dev = "cuda:0"
        for n in (7, 128, 1500):
            H, dh = 8, 64
            q = torch.randn(H, dh, device=dev)
            K = torch.randn(H, n, dh, device=dev)
            V = torch.randn(H, n, dh, device=dev)
            ref = torch.softmax((q.unsqueeze(1) @ K.transpose(-1, -2)).squeeze(1)
                                * 0.125, -1).unsqueeze(1) @ V
            got = decode_attn(q, K, V, 0.125)
            err = (got.float() - ref.squeeze(1).float()).abs().max().item()
            self.assertLess(err, 1e-4, f"decode_attn n={n} err={err:.2e}")

    def test_batched_decode_attn(self):
        _need_cuda(self)
        _, _, _, batched_decode_attn = _load_stt_kernels()
        torch.manual_seed(0)
        dev = "cuda:0"
        for B, n in ((4, 7), (4, 64)):
            H, dh = 8, 64
            q = torch.randn(B, H, dh, device=dev)
            K = torch.randn(B, H, n, dh, device=dev)
            V = torch.randn(B, H, n, dh, device=dev)
            ref = torch.softmax((q.unsqueeze(2) @ K.transpose(-1, -2)).squeeze(2)
                                * 0.125, -1).unsqueeze(2) @ V
            got = batched_decode_attn(q, K, V, 0.125)
            err = (got.float() - ref.squeeze(2).float()).abs().max().item()
            self.assertLess(err, 1e-4, f"batched_decode B={B} n={n} err={err:.2e}")


if __name__ == "__main__":
    unittest.main()
