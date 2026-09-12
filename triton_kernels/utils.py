"""Shared grid / dtype helpers for triton_kernels (no CUDA tensors required)."""
import triton
import triton.language as tl
import torch


def next_pow2(n):
    """Smallest power of two >= n (n >= 1)."""
    n = int(n)
    if n < 1:
        raise ValueError("next_pow2 requires n >= 1")
    return triton.next_power_of_2(n)


def grid1d(n, block=1024):
    """1D launch grid covering n elements with tiles of `block`.

    Returns a 1-tuple usable directly as a Triton grid:
        _kernel[grid1d(n)](...)
    """
    n = int(n)
    block = int(block)
    if n < 0:
        raise ValueError("grid1d requires n >= 0")
    if block <= 0:
        raise ValueError("grid1d requires block > 0")
    return ((n + block - 1) // block,)


def cdiv(a, b):
    """Ceiling integer division."""
    a = int(a)
    b = int(b)
    if b <= 0:
        raise ValueError("cdiv requires b > 0")
    return -(-a // b)


def num_warps_for_width(n):
    """Reasonable num_warps for a row-width-n elementwise kernel."""
    n = int(n)
    if n <= 256:
        return 2
    if n <= 1024:
        return 4
    return 8


def tl_dtype(dtype):
    """Map a torch.dtype to the matching tl.* dtype."""
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.int32:
        return tl.int32
    if dtype == torch.int64:
        return tl.int64
    raise ValueError(f"unsupported dtype for triton: {dtype}")


def check_cuda(*tensors):
    """Raise a clear error if any tensor is not a CUDA tensor."""
    for t in tensors:
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError("expected CUDA tensor, got %r" % (t,))
