"""Kernel surface for the Qwen leg.

Re-exports RMSNorm / RoPE / SwiGLU / GQA decode attention from
``triton_kernels`` when available, otherwise provides real torch fallbacks
with identical semantics (HF/NeoX half-rotation for RoPE).
"""
import torch
import torch.nn.functional as F

__all__ = [
    "rmsnorm",
    "rope",
    "rope_batched",
    "swiglu",
    "gqa_decode_attn",
    "HAVE_TRITON_KERNELS",
]

HAVE_TRITON_KERNELS = False
_tk_rms = _tk_rope = _tk_swiglu = _tk_gqa = None
_tk_rope_batched = None
try:
    from triton_kernels.rmsnorm import rmsnorm as _tk_rms  # noqa: F401
    from triton_kernels.rope import rope as _tk_rope  # noqa: F401
    from triton_kernels.activation import swiglu as _tk_swiglu  # noqa: F401
    from triton_kernels.attention import gqa_decode_attn as _tk_gqa  # noqa: F401
    HAVE_TRITON_KERNELS = True
    try:
        from triton_kernels.rope import rope_batched as _tk_rope_batched  # noqa: F401
    except Exception:
        _tk_rope_batched = None
except Exception:
    HAVE_TRITON_KERNELS = False


def _cuda(t):
    return isinstance(t, torch.Tensor) and t.is_cuda


def rmsnorm(x, w, eps=1e-6):
    if HAVE_TRITON_KERNELS and _cuda(x):
        return _tk_rms(x, w, eps)
    xf = x.float()
    var = (xf * xf).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * w.float()).to(x.dtype).reshape(x.shape)


def rope(x, cos, sin, pos):
    """Half-rotation RoPE matching HF ``apply_rotary_pos_emb``."""
    if HAVE_TRITON_KERNELS and _cuda(x):
        return _tk_rope(x, cos, sin, pos)
    shp = x.shape
    dh = shp[-1]
    half = dh // 2
    xf = x.reshape(-1, dh).float()
    c = cos[pos].to(torch.float32)
    s = sin[pos].to(torch.float32)
    x1, x2 = xf[:, :half], xf[:, half:]
    y = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return y.to(x.dtype).reshape(shp)


def rope_batched(x, cos, sin, pos0=0):
    """[T, H, dh] -> rotated with positions pos0..pos0+T-1.

    One Triton launch for the whole prefill; torch loop fallback.
    """
    if HAVE_TRITON_KERNELS and _tk_rope_batched is not None and _cuda(x):
        return _tk_rope_batched(x, cos, sin, pos0)
    T = x.shape[0]
    return torch.cat([rope(x[t:t + 1], cos, sin, pos0 + t)
                      for t in range(T)], dim=0)


def swiglu(gate, up):
    if HAVE_TRITON_KERNELS and _cuda(gate):
        return _tk_swiglu(gate, up)
    return F.silu(gate.float()).to(gate.dtype) * up


def gqa_decode_attn(q, K, V, scale):
    """q [Hq,D], K/V [Hk,N,D] -> O [Hq,D], Hq % Hk == 0."""
    if HAVE_TRITON_KERNELS and _cuda(q):
        return _tk_gqa(q, K, V, scale)
    Hq, D = q.shape
    Hk = K.shape[0]
    group = Hq // Hk
    Ke = K.repeat_interleave(group, dim=0).float()
    Ve = V.repeat_interleave(group, dim=0).float()
    scores = torch.einsum("hd,hnd->hn", q.float(), Ke) * scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("hn,hnd->hd", probs, Ve)
