"""Rotary position embedding (HF/NeoX half-rotation) Triton kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(X, Y, COS, SIN, pos, stride, N, HALF: tl.constexpr):
    # HF/NeoX-style half rotation (matches apply_rotary_pos_emb):
    # out[:H] = x[:H]*cos - x[H:]*sin ; out[H:] = x[:H]*sin + x[H:]*cos
    row = tl.program_id(0)
    i = tl.arange(0, HALF)
    x1 = tl.load(X + row * stride + i).to(tl.float32)
    x2 = tl.load(X + row * stride + HALF + i).to(tl.float32)
    c = tl.load(COS + pos * HALF + i).to(tl.float32)
    s = tl.load(SIN + pos * HALF + i).to(tl.float32)
    tl.store(Y + row * stride + i, x1 * c - x2 * s)
    tl.store(Y + row * stride + HALF + i, x1 * s + x2 * c)


def rope(x, cos, sin, pos):
    """x [..., dh] (dh even) -> rotated, same shape. cos/sin [maxpos, dh/2].
    Half-rotation convention matching HF apply_rotary_pos_emb."""
    assert x.is_cuda
    shp = x.shape
    xf = x.reshape(-1, shp[-1]).contiguous()
    y = torch.empty_like(xf)
    dh = shp[-1]
    assert dh % 2 == 0
    _rope_kernel[(xf.shape[0],)](xf, y, cos, sin, pos, xf.stride(0), dh,
                                 dh // 2)
    return y.reshape(shp)


@triton.jit
def _rope_batch_kernel(X, Y, COS, SIN, pos0, stride_t, stride_h, H,
                       HALF: tl.constexpr):
    # One program per (t, h). Position = pos0 + t (true prefill positions).
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H
    pos = pos0 + t
    i = tl.arange(0, HALF)
    off = t * stride_t + h * stride_h
    x1 = tl.load(X + off + i).to(tl.float32)
    x2 = tl.load(X + off + HALF + i).to(tl.float32)
    c = tl.load(COS + pos * HALF + i).to(tl.float32)
    s = tl.load(SIN + pos * HALF + i).to(tl.float32)
    tl.store(Y + off + i, x1 * c - x2 * s)
    tl.store(Y + off + HALF + i, x1 * s + x2 * c)


def rope_batched(x, cos, sin, pos0=0):
    """x [T, H, dh] -> rotated, positions pos0..pos0+T-1. ONE kernel launch
    for the whole prefill (replaces 2T single-row launches per layer)."""
    assert x.is_cuda and x.dim() == 3
    T, H, dh = x.shape
    assert dh % 2 == 0
    xc = x.contiguous()
    y = torch.empty_like(xc)
    _rope_batch_kernel[(T * H,)](xc, y, cos, sin, pos0, xc.stride(0),
                                 xc.stride(1), H, dh // 2)
    return y
