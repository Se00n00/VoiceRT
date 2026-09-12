"""Blocked fp16/bf16 GEMM Triton kernel (C = A @ B)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(A, B, C,
                         M, N, K,
                         stride_am, stride_ak,
                         stride_bk, stride_bn,
                         stride_cm, stride_cn,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid_m // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid_m % group_size_m)
    pid_n = pid_n % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (k * BLOCK_K + offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(k * BLOCK_K + offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(C.dtype.element_ty)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a, b):
    """Blocked GEMM: a [M,K] @ b [K,N] -> [M,N] (fp16/bf16/fp32, CUDA).

    Small reference-style kernel with L2-friendly swizzling; compute-bound
    shapes should still prefer cuBLAS.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert b.dtype == a.dtype
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 256, 64, 8
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _matmul_kernel[grid](a, b, c, M, N, K,
                               a.stride(0), a.stride(1),
                               b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1),
                               BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    return c
