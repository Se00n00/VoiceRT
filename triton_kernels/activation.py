"""Fused activation kernels (SwiGLU / SiLU)."""
import torch
import triton
import triton.language as tl
from triton_kernels.utils import grid1d


@triton.jit
def _swiglu_kernel(A, B, Y, N,
                   BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs, mask=offs < N).to(tl.float32)
    b = tl.load(B + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.store(Y + offs, a / (1.0 + tl.exp(-a)) * b, mask=offs < N)


def swiglu(gate, up):
    """silu(gate) * up, elementwise fused. Shapes must match."""
    assert gate.is_cuda and gate.shape == up.shape
    y = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 1024
    _swiglu_kernel[grid1d(n, BLOCK)](
        gate.reshape(-1), up.reshape(-1), y.reshape(-1), n, BLOCK)
    return y


@triton.jit
def _silu_kernel(X, Y, N,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs, mask=offs < N).to(tl.float32)
    tl.store(Y + offs, x / (1.0 + tl.exp(-x)), mask=offs < N)


def silu(x):
    """Fused SiLU: x * sigmoid(x), elementwise."""
    assert x.is_cuda
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    _silu_kernel[grid1d(n, BLOCK)](x.reshape(-1), y.reshape(-1), n, BLOCK)
    return y


@triton.jit
def _lstm_cell_kernel(
    X, Hprev, Cprev, Wih, Whh, B, Hnew, Cnew,
    stride_xb, stride_xi,
    stride_hb, stride_h,
    stride_wih0, stride_wih1,
    stride_whh0, stride_whh1,
    I, H,
    HAS_BIAS: tl.constexpr,
    BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """One fused LSTM step (dominant op of the Silero VAD recurrent net).

    Grid is (B*H,): each program computes one hidden unit for one batch
    element, including its four gates (GEMV over I+H), sigmoid/tanh
    activations and the c/h update. No `continue`, no chained comparisons.
    """
    pid = tl.program_id(0)
    b = pid // H
    hid = pid % H
    acc_i = 0.0
    acc_f = 0.0
    acc_g = 0.0
    acc_o = 0.0
    # -- input GEMV part: X[b, :] @ Wih[[hid, hid+H, hid+2H, hid+3H], :].T --
    for i0 in range(0, I, BLOCK_I):
        offs_i = i0 + tl.arange(0, BLOCK_I)
        mask_i = offs_i < I
        xv = tl.load(X + b * stride_xb + offs_i * stride_xi,
                     mask=mask_i, other=0.0).to(tl.float32)
        wi = tl.load(Wih + hid * stride_wih0 + offs_i * stride_wih1,
                     mask=mask_i, other=0.0).to(tl.float32)
        wf = tl.load(Wih + (hid + H) * stride_wih0 + offs_i * stride_wih1,
                     mask=mask_i, other=0.0).to(tl.float32)
        wg = tl.load(Wih + (hid + H * 2) * stride_wih0 + offs_i * stride_wih1,
                     mask=mask_i, other=0.0).to(tl.float32)
        wo = tl.load(Wih + (hid + H * 3) * stride_wih0 + offs_i * stride_wih1,
                     mask=mask_i, other=0.0).to(tl.float32)
        acc_i += tl.sum(xv * wi, 0)
        acc_f += tl.sum(xv * wf, 0)
        acc_g += tl.sum(xv * wg, 0)
        acc_o += tl.sum(xv * wo, 0)
    # -- recurrent GEMV part: Hprev[b, :] @ Whh[...].T --
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        hv = tl.load(Hprev + b * stride_hb + offs_h * stride_h,
                     mask=mask_h, other=0.0).to(tl.float32)
        wi = tl.load(Whh + hid * stride_whh0 + offs_h * stride_whh1,
                     mask=mask_h, other=0.0).to(tl.float32)
        wf = tl.load(Whh + (hid + H) * stride_whh0 + offs_h * stride_whh1,
                     mask=mask_h, other=0.0).to(tl.float32)
        wg = tl.load(Whh + (hid + H * 2) * stride_whh0 + offs_h * stride_whh1,
                     mask=mask_h, other=0.0).to(tl.float32)
        wo = tl.load(Whh + (hid + H * 3) * stride_whh0 + offs_h * stride_whh1,
                     mask=mask_h, other=0.0).to(tl.float32)
        acc_i += tl.sum(hv * wi, 0)
        acc_f += tl.sum(hv * wf, 0)
        acc_g += tl.sum(hv * wg, 0)
        acc_o += tl.sum(hv * wo, 0)
    if HAS_BIAS:
        acc_i += tl.load(B + hid).to(tl.float32)
        acc_f += tl.load(B + hid + H).to(tl.float32)
        acc_g += tl.load(B + hid + H * 2).to(tl.float32)
        acc_o += tl.load(B + hid + H * 3).to(tl.float32)
    # -- gates: sigmoid/tanh via tl.sigmoid (tanh(x) = 2*sigmoid(2x)-1) --
    i_g = tl.sigmoid(acc_i)
    f_g = tl.sigmoid(acc_f)
    g_g = 2.0 * tl.sigmoid(2.0 * acc_g) - 1.0
    o_g = tl.sigmoid(acc_o)
    c_prev = tl.load(Cprev + b * stride_hb + hid * stride_h).to(tl.float32)
    c_new = f_g * c_prev + i_g * g_g
    tanh_c = 2.0 * tl.sigmoid(2.0 * c_new) - 1.0
    h_new = o_g * tanh_c
    tl.store(Cnew + b * stride_hb + hid * stride_h,
             c_new.to(Cnew.dtype.element_ty))
    tl.store(Hnew + b * stride_hb + hid * stride_h,
             h_new.to(Hnew.dtype.element_ty))


def lstm_cell(x, h, c, w_ih, w_hh, b=None):
    """One LSTM step: (x, h, c, w_ih, w_hh, b) -> (h_new, c_new).

    Shapes: x [I] or [B, I], h/c [H] or [B, H], w_ih [4H, I],
    w_hh [4H, H], b [4H] or [8H] (ih+hh halves summed) or None.
    Gate order is torch's i,f,g,o. Parity-tested vs nn.LSTMCell.
    """
    assert x.is_cuda and h.is_cuda and c.is_cuda
    assert w_ih.is_cuda and w_hh.is_cuda
    assert w_ih.dim() == 2 and w_hh.dim() == 2
    H4, I = w_ih.shape
    H4b, Hb = w_hh.shape
    assert H4 == H4b
    assert H4 % 4 == 0
    H = H4 // 4
    assert Hb == H
    assert w_ih.shape[1] == I
    squeeze = False
    if x.dim() == 1:
        assert x.shape[0] == I
        assert h.shape == (H,) and c.shape == (H,)
        x2 = x.reshape(1, I).contiguous()
        h2 = h.reshape(1, H).contiguous()
        c2 = c.reshape(1, H).contiguous()
        squeeze = True
    else:
        assert x.dim() == 2 and h.dim() == 2 and c.dim() == 2
        assert x.shape[1] == I and h.shape[1] == H and c.shape[1] == H
        assert x.shape[0] == h.shape[0] == c.shape[0]
        x2 = x.contiguous()
        h2 = h.contiguous()
        c2 = c.contiguous()
    Bsz = x2.shape[0]
    if b is not None:
        b = torch.as_tensor(b, device=x.device)
        if b.numel() == 8 * H:
            b = (b[:4 * H] + b[4 * H:]).contiguous()
        assert b.shape == (4 * H,)
        bc = b.contiguous()
        has_bias = True
    else:
        bc = x2.reshape(-1)[:1].contiguous()
        has_bias = False
    wihc = w_ih.contiguous()
    whhc = w_hh.contiguous()
    h_new = torch.empty((Bsz, H), device=x.device, dtype=x.dtype)
    c_new = torch.empty((Bsz, H), device=x.device, dtype=x.dtype)
    BLOCK_I, BLOCK_H = 64, 64
    grid = (Bsz * H,)
    _lstm_cell_kernel[grid](
        x2, h2, c2, wihc, whhc, bc, h_new, c_new,
        x2.stride(0), x2.stride(1),
        h2.stride(0), h2.stride(1),
        wihc.stride(0), wihc.stride(1),
        whhc.stride(0), whhc.stride(1),
        I, H, has_bias, BLOCK_I, BLOCK_H,
    )
    if squeeze:
        return h_new.reshape(H), c_new.reshape(H)
    return h_new, c_new
