"""LayerNorm Triton kernel."""
import torch
import triton
import triton.language as tl
from src.models.triton_kernels.utils import next_pow2


@triton.jit
def _ln_kernel(X, Y, W, B, stride, N, eps,
               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride + cols, mask=cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    var = tl.sum((x - mean) * (x - mean), 0) / N
    xh = (x - mean) * tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=cols < N, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (xh * w + b).to(tl.float32),
             mask=cols < N)


def layernorm(x, w, b, eps=1e-5):
    assert x.is_cuda
    x = x.reshape(-1, x.shape[-1]).contiguous()
    N = x.shape[-1]
    assert N <= 1024, "tile layernorm for larger N"
    y = torch.empty_like(x)
    BLOCK = next_pow2(N)
    grid = (x.numel() // N,)
    _ln_kernel[grid](x, y, w, b, x.stride(-2) if x.dim() > 1 else N,
                     N, eps, BLOCK)
    return y
