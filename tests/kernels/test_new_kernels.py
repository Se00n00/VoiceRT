"""Parity for newly added hand-written Triton kernels (STT/LLM/TTS/VAD legs).

Skips (does not fail) when torch/triton/cuda is unavailable.
"""
import unittest

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


def _need_cuda(tc):
    if not _HAS_TORCH:
        tc.skipTest("torch missing")
    if not torch.cuda.is_available():
        tc.skipTest("needs cuda")


class TestNewKernels(unittest.TestCase):
    def test_fused_qkv_whisper(self):
        """Whisper d=512/H=8: q/v biased, k unbiased (verified convention)."""
        _need_cuda(self)
        from triton_kernels.attention import fused_qkv
        torch.manual_seed(0)
        dev = "cuda:0"
        D = 512
        for dtype in (torch.float32, torch.bfloat16):
            x = (torch.randn(D, device=dev, dtype=torch.float32) * 0.5).to(dtype)
            wq = (torch.randn(D, D, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            wk = (torch.randn(D, D, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            wv = (torch.randn(D, D, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            bq = (torch.randn(D, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            bv = (torch.randn(D, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            q, k, v = fused_qkv(x, wq, wk, wv, bq, bv, None)
            rq = F.linear(x.float(), wq.float(), bq.float())
            rk = F.linear(x.float(), wk.float(), None)
            rv = F.linear(x.float(), wv.float(), bv.float())
            for got, ref, name in ((q, rq, "q"), (k, rk, "k"), (v, rv, "v")):
                err = (got.float() - ref.float()).abs().max().item()
                tol = 2e-5 if dtype == torch.float32 else 1e-2
                self.assertLess(err, tol, f"fused_qkv {name} {dtype} err={err:.2e}")

    def test_fused_qkv_gqa_qwen(self):
        """Qwen d=896/Hq=14/Hk=2: q/k/v ALL biased (verified convention)."""
        _need_cuda(self)
        from triton_kernels.attention import fused_qkv_gqa
        torch.manual_seed(1)
        dev = "cuda:0"
        D, Dq, Dkv = 896, 896, 128
        for dtype in (torch.float32, torch.bfloat16):
            x = (torch.randn(D, device=dev, dtype=torch.float32) * 0.5).to(dtype)
            wq = (torch.randn(Dq, D, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            wk = (torch.randn(Dkv, D, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            wv = (torch.randn(Dkv, D, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            bq = (torch.randn(Dq, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            bk = (torch.randn(Dkv, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            bv = (torch.randn(Dkv, device=dev, dtype=torch.float32) * 0.05).to(dtype)
            q, k, v = fused_qkv_gqa(x, wq, wk, wv, bq, bk, bv)
            rq = F.linear(x.float(), wq.float(), bq.float())
            rk = F.linear(x.float(), wk.float(), bk.float())
            rv = F.linear(x.float(), wv.float(), bv.float())
            for got, ref, name in ((q, rq, "q"), (k, rk, "k"), (v, rv, "v")):
                err = (got.float() - ref.float()).abs().max().item()
                tol = 3e-5 if dtype == torch.float32 else 1e-2
                self.assertLess(err, tol, f"fused_qkv_gqa {name} {dtype} err={err:.2e}")

    def test_conv1d_silu(self):
        """Kokoro-ish conv C=512/L=200: parity vs conv1d+silu; FAIL if bf16>1e-2."""
        _need_cuda(self)
        from triton_kernels.conv1d import conv1d_silu
        torch.manual_seed(2)
        dev = "cuda:0"
        # Small shape for speed + one realistic Kokoro-ish shape.
        for (N, Cin, Cout, L, K) in ((1, 8, 8, 32, 3), (1, 32, 32, 200, 3)):
            for dtype in (torch.float32, torch.bfloat16):
                x = (torch.randn(N, Cin, L, device=dev, dtype=torch.float32) * 0.5).to(dtype)
                w = (torch.randn(Cout, Cin, K, device=dev, dtype=torch.float32) * 0.1).to(dtype)
                b = (torch.randn(Cout, device=dev, dtype=torch.float32) * 0.1).to(dtype)
                got = conv1d_silu(x, w, b, stride=1, padding=1)
                ref = F.silu(F.conv1d(x.float(), w.float(), b.float(), padding=1))
                err = (got.float() - ref.float()).abs().max().item()
                if dtype == torch.bfloat16 and err > 1e-2:
                    raise AssertionError(f"conv1d_silu bf16 diff {err:.2e} > 1e-2")
                tol = 1e-5 if dtype == torch.float32 else 1e-2
                self.assertLess(err, tol, f"conv1d_silu {dtype} err={err:.2e}")

    def test_in1d_silu(self):
        """Fused InstanceNorm1d+SiLU: parity vs instance_norm+silu; FAIL if bf16>1e-2."""
        _need_cuda(self)
        from triton_kernels.conv1d import in1d_silu
        torch.manual_seed(3)
        dev = "cuda:0"
        for (N, C, L) in ((2, 8, 32), (1, 64, 200)):
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(N, C, L, device=dev, dtype=torch.float32).to(dtype)
                got = in1d_silu(x)
                m = x.float().mean(-1, keepdim=True)
                va = x.float().var(-1, unbiased=False, keepdim=True)
                ref = F.silu((x.float() - m) / torch.sqrt(va + 1e-5))
                err = (got.float() - ref.float()).abs().max().item()
                if dtype == torch.bfloat16 and err > 1e-2:
                    raise AssertionError(f"in1d_silu bf16 diff {err:.2e} > 1e-2")
                tol = 1e-5 if dtype == torch.float32 else 1e-2
                self.assertLess(err, tol, f"in1d_silu {dtype} err={err:.2e}")
        # 3D instance_norm equivalence spot-check (no affine).
        x = torch.randn(2, 4, 32, device=dev, dtype=torch.float32)
        got = in1d_silu(x)
        ref = F.silu(F.instance_norm(x))
        err = (got.float() - ref.float()).abs().max().item()
        self.assertLess(err, 2e-5, f"in1d vs instance_norm err={err:.2e}")

    def test_lstm_cell(self):
        """Silero-ish LSTM hidden 128: parity vs nn.LSTMCell."""
        _need_cuda(self)
        from triton_kernels.activation import lstm_cell
        torch.manual_seed(4)
        dev = "cuda:0"
        for (B, H, I) in ((1, 128, 64), (2, 128, 128)):
            for dtype in (torch.float32, torch.bfloat16):
                x = (torch.randn(B, I, device=dev, dtype=torch.float32) * 0.5).to(dtype)
                h = (torch.randn(B, H, device=dev, dtype=torch.float32) * 0.5).to(dtype)
                c = (torch.randn(B, H, device=dev, dtype=torch.float32) * 0.5).to(dtype)
                w_ih = (torch.randn(4 * H, I, device=dev, dtype=torch.float32) * 0.1).to(dtype)
                w_hh = (torch.randn(4 * H, H, device=dev, dtype=torch.float32) * 0.1).to(dtype)
                b = (torch.randn(4 * H, device=dev, dtype=torch.float32) * 0.1).to(dtype)
                hn, cn = lstm_cell(x, h, c, w_ih, w_hh, b)
                cell = torch.nn.LSTMCell(I, H, device=dev, dtype=torch.float32)
                with torch.no_grad():
                    cell.weight_ih.copy_(w_ih.float())
                    cell.weight_hh.copy_(w_hh.float())
                    cell.bias_ih.copy_(b.float())
                    cell.bias_hh.zero_()
                    rhn, rcn = cell(x.float(), (h.float(), c.float()))
                for got, ref, name in ((hn, rhn, "h"), (cn, rcn, "c")):
                    err = (got.float() - ref.float()).abs().max().item()
                    tol = 5e-6 if dtype == torch.float32 else 1e-2
                    self.assertLess(err, tol, f"lstm_cell {name} {dtype} err={err:.2e}")

    def test_rope_batched(self):
        _need_cuda(self)
        from triton_kernels.rope import rope, rope_batched
        torch.manual_seed(3)
        T, H, dh = 21, 14, 64
        for dtype in (torch.float32, torch.bfloat16):
            x = torch.randn(T, H, dh, device="cuda", dtype=dtype)
            cos = torch.randn(2048, dh // 2, device="cuda", dtype=dtype)
            sin = torch.randn(2048, dh // 2, device="cuda", dtype=dtype)
            ref = torch.cat([rope(x[t:t + 1], cos, sin, t)
                             for t in range(T)], dim=0)
            got = rope_batched(x, cos, sin, 0)
            err = (got.float() - ref.float()).abs().max().item()
            tol = 1e-5 if dtype == torch.float32 else 1e-2
            self.assertLess(err, tol, f"rope_batched {dtype} err={err:.2e}")


if __name__ == "__main__":
    unittest.main()
