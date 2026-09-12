"""Fused Conv1d forward + SiLU Triton kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_silu_fwd_kernel(
    X,          # input: pointer to (N, C_in, L_in) flat, row-major
    W,          # weight: pointer to (C_out, C_in, K) flat, row-major
    B,          # bias: pointer to (C_out,) flat
    Y,          # output: pointer to (N, C_out, L_out) flat, row-major
    # Strides for X (input)
    stride_x_n, stride_x_c, stride_x_l,  # N, C_in, L_in dims
    # Strides for W (weight)
    stride_w_c, stride_w_in, stride_w_k,  # C_out, C_in, K
    # Strides for B (bias, just unit stride)
    stride_b,
    # Strides for Y (output)
    stride_y_n, stride_y_c, stride_y_l,  # N, C_out, L_out dims
    # Conv hyperparameters (all constexpr)
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    K: tl.constexpr,
    PADDING: tl.constexpr,
    DILATION: tl.constexpr,
    STRIDE: tl.constexpr,
    L_IN: tl.constexpr,
    L_OUT: tl.constexpr,
):
    """
    Fused Conv1d forward + SiLU activation.
    Each program instance computes ONE output element y[n, c_out, pos].
    Launched over grid: (N * C_out * L_OUT).
    Input:  (N, C_in, L_in)   with L_in = L_OUT * STRIDE - 2*PADDING + DILATION*(K-1) + 1
    Weight: (C_out, C_in, K)
    Bias:   (C_out,)
    Output: (N, C_out, L_out)
    """
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    pos = tl.program_id(2)

    # Compute output offset in flat Y
    y_offset = n * stride_y_n + c_out * stride_y_c + pos * stride_y_l

    # Compute input window start position
    inp_start = pos * STRIDE - PADDING

    # Accumulate conv result (scalar fp32; Triton store needs scalar
    # here because each program computes ONE output element).
    acc = 0.0

    # Loop over input channels and kernel positions.
    # NOTE: Triton rejects `continue` (unsupported AST) and chained
    # comparisons like `0 <= x < N`, so out-of-bounds taps use masked
    # loads producing 0.0 instead of skipping the iteration.
    for cin in range(C_in):
        for k in range(K):
            # Input position within the window
            inp_pos = inp_start + k * DILATION
            valid = (inp_pos >= 0) & (inp_pos < L_IN)
            # Weight index is always valid (c_out is fixed per program instance)
            w_idx = c_out * stride_w_c + cin * stride_w_in + k * stride_w_k

            x_idx = n * stride_x_n + cin * stride_x_c + inp_pos * stride_x_l

            x_val = tl.load(X + x_idx, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(W + w_idx).to(tl.float32)
            acc = acc + x_val * w_val

    # Add bias (broadcast across n and pos)
    acc = acc + tl.load(B + c_out * stride_b).to(tl.float32)

    # SiLU activation: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-acc))
    acc = acc * sig

    # Store output
    tl.store(Y + y_offset, acc)


def conv1d_silu(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Fused Conv1d + SiLU matching torch.nn.Conv1d output shapes.

    x:      [N, C_in, L_in] CUDA tensor.
    weight: [C_out, C_in, K] CUDA tensor.
    bias:   [C_out] CUDA tensor or None (zeros used when None).
    Returns [N, C_out, L_out] with
    L_out = (L_in + 2*padding - dilation*(K-1) - 1) // stride + 1.
    """
    assert x.is_cuda and weight.is_cuda
    assert x.dim() == 3 and weight.dim() == 3
    N, C_in, L_in = x.shape
    C_out, C_in_w, K = weight.shape
    assert C_in == C_in_w
    if bias is None:
        bias = torch.zeros((C_out,), device=x.device, dtype=torch.float32)
    assert bias.is_cuda and bias.shape == (C_out,)
    L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    assert L_out > 0
    xc = x.contiguous()
    wc = weight.contiguous()
    bc = bias.contiguous()
    y = torch.empty((N, C_out, L_out), device=x.device, dtype=torch.float32)
    grid = (N, C_out, L_out)
    conv1d_silu_fwd_kernel[grid](
        xc, wc, bc, y,
        xc.stride(0), xc.stride(1), xc.stride(2),
        wc.stride(0), wc.stride(1), wc.stride(2),
        bc.stride(0),
        y.stride(0), y.stride(1), y.stride(2),
        N, C_in, C_out, K, padding, dilation, stride, L_in, L_out,
    )
    return y.to(x.dtype)


@triton.jit
def _in1d_silu_kernel(X, Y, stride_row, L, eps, BLOCK: tl.constexpr):
    """Fused InstanceNorm1d (over last dim) + SiLU.

    One program per instance row (N*C rows for [N, C, L] input).
    No `continue`, no chained comparisons; all OOB lanes masked.
    """
    row = tl.program_id(0)
    base = row * stride_row
    n_f = L * 1.0
    # -- mean over L --
    acc = 0.0
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask, x, 0.0), 0)
    mean = acc / n_f
    # -- variance over L --
    var_acc = 0.0
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - mean, 0.0)
        var_acc += tl.sum(d * d, 0)
    var = var_acc / n_f
    inv = 1.0 / tl.sqrt(var + eps)
    # -- norm + SiLU store --
    for start in range(0, L, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < L
        x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
        xn = (x - mean) * inv
        sig = 1.0 / (1.0 + tl.exp(-xn))
        y = xn * sig
        tl.store(Y + base + offs, y.to(Y.dtype.element_ty), mask=mask)


def in1d_silu(x, eps=1e-5):
    """Fused InstanceNorm1d over last dim + SiLU.

    x: [..., L] CUDA tensor (e.g. [N, C, L] Kokoro activations).
    Normalizes each [..., :] row to zero mean / unit variance (eps)
    with fp32 accumulation, then applies SiLU. Matches
    ``silu((x - mean) / sqrt(var + eps))`` and, for 3D inputs,
    ``silu(instance_norm(x))`` with no affine weights.
    """
    assert x.is_cuda
    assert x.dim() >= 2
    shp = x.shape
    L = shp[-1]
    assert L > 0
    R = 1
    for s in shp[:-1]:
        R *= int(s)
    xf = x.reshape(R, L).contiguous()
    yf = torch.empty((R, L), device=x.device, dtype=x.dtype)
    BLOCK = 1024
    grid = (R,)
    _in1d_silu_kernel[grid](xf, yf, xf.stride(0), L, float(eps), BLOCK)
    return yf.reshape(shp)
