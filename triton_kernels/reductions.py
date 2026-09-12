"""Row-wise reduction Triton kernels (sum / max).

Sources contain reductions only fused inside layernorm/softmax/attention;
these standalone row reductions are new small kernels written for this
package (real working kernels, not stubs).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _row_sum_kernel(X, Y, stride, N,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = 0.0
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * stride + cols, mask=cols < N,
                    other=0.0).to(tl.float32)
        acc += tl.sum(x, 0)
    tl.store(Y + row, acc)


def row_sum(x):
    """Sum over the last dim: x [..., N] -> [...] (fp32 accumulate)."""
    assert x.is_cuda
    xf = x.reshape(-1, x.shape[-1]).contiguous()
    N = xf.shape[-1]
    y = torch.empty((xf.shape[0],), device=x.device, dtype=torch.float32)
    _row_sum_kernel[(xf.shape[0],)](xf, y, xf.stride(0), N, 1024)
    return y.reshape(x.shape[:-1])


@triton.jit
def _row_max_kernel(X, Y, stride, N,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    mx = float("-inf")
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * stride + cols, mask=cols < N,
                    other=float("-inf")).to(tl.float32)
        mx = tl.maximum(mx, tl.max(x, 0))
    tl.store(Y + row, mx)


def row_max(x):
    """Max over the last dim: x [..., N] -> [...] (fp32)."""
    assert x.is_cuda
    xf = x.reshape(-1, x.shape[-1]).contiguous()
    N = xf.shape[-1]
    y = torch.empty((xf.shape[0],), device=x.device, dtype=torch.float32)
    _row_max_kernel[(xf.shape[0],)](xf, y, xf.stride(0), N, 1024)
    return y.reshape(x.shape[:-1])
