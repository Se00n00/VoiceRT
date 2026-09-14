"""RMSNorm Triton kernel (Qwen-style, no mean centering)."""
import torch
import triton
import triton.language as tl
from src.models.triton_kernels.utils import next_pow2


@triton.jit
def _rmsnorm_kernel(X, Y, W, stride, N, eps,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride + cols, mask=cols < N,
                other=0.0).to(tl.float32)
    var = tl.sum(x * x, 0) / N
    xh = x * tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (xh * w).to(tl.float32),
             mask=cols < N)


def rmsnorm(x, w, eps=1e-6):
    assert x.is_cuda
    x = x.reshape(-1, x.shape[-1]).contiguous()
    N = x.shape[-1]
    assert N <= 1024
    y = torch.empty_like(x)
    _rmsnorm_kernel[(x.shape[0],)](x, y, w, x.stride(0), N, eps,
                                   next_pow2(N))
    return y
