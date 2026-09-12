"""Embedding lookup (gather) Triton kernel.

Sources contain no embedding kernel; this is a new small row-gather written
for this package (real working kernel, not a stub).
"""
import torch
import triton
import triton.language as tl
from triton_kernels.utils import grid1d


@triton.jit
def _embedding_fwd_kernel(W, IDX, Y,
                          stride_w, stride_y,
                          D,
                          BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    idx = tl.load(IDX + row)
    offs = tl.arange(0, BLOCK_D)
    w_row = tl.load(W + idx * stride_w + offs, mask=offs < D)
    tl.store(Y + row * stride_y + offs, w_row, mask=offs < D)


def embedding_lookup(weight, indices):
    """Gather rows: weight [V,D], indices [...] (int64) -> [..., D].

    Mirrors torch.nn.functional.embedding for CUDA inputs.
    """
    assert weight.is_cuda and indices.is_cuda
    assert weight.dim() == 2
    V, D = weight.shape
    flat = indices.reshape(-1).contiguous()
    y = torch.empty((flat.shape[0], D), device=weight.device, dtype=weight.dtype)
    BLOCK_D = triton.next_power_of_2(D)
    assert BLOCK_D <= 4096, "tile embedding_lookup for very wide D"
    _embedding_fwd_kernel[(flat.shape[0],)](weight, flat, y,
                                            weight.stride(0), y.stride(0),
                                            D, BLOCK_D)
    return y.reshape(*indices.shape, D)
