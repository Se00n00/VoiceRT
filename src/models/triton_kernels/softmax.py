"""Row-wise softmax Triton kernel (any N via tiling)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(X, Y, stride, N,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # pass 1: max (BLOCK fixed at 1024, loop over tiles for any N)
    mx = float("-inf")
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * stride + cols, mask=cols < N,
                    other=float("-inf")).to(tl.float32)
        mx = tl.maximum(mx, tl.max(x, 0))
    # pass 2: sum + normalize
    s = 0.0
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * stride + cols, mask=cols < N,
                    other=float("-inf")).to(tl.float32)
        s += tl.sum(tl.exp(x - mx), 0)
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(X + row * stride + cols, mask=cols < N,
                    other=float("-inf")).to(tl.float32)
        tl.store(Y + row * stride + cols, tl.exp(x - mx) / s,
                 mask=cols < N)


def row_softmax(x):
    assert x.is_cuda and x.is_contiguous()
    y = torch.empty_like(x)
    N = x.shape[-1]
    BLOCK = 1024
    grid = (x.numel() // N,)
    _softmax_kernel[grid](x, y, x.stride(-2) if x.dim() > 1 else N,
                          N, BLOCK)
    return y
