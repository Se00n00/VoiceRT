"""Complete Triton kernel surface for the Qwen (LLM) leg.

Single import point for everything the LLM needs:
rmsnorm, rope, rope_batched, swiglu, gqa_decode_attn, fused_qkv_gqa.

Each wrapper tries the hand-written Triton kernel on CUDA and falls back
to an exact torch reference on CPU (or when Triton is unavailable), so the
engine runs everywhere. ``HAVE_TRITON_KERNELS`` is True when the Triton
implementations import cleanly (they still require CUDA at call time).
"""
import torch
import torch.nn.functional as F

__all__ = [
    "rmsnorm",
    "rope",
    "rope_batched",
    "swiglu",
    "gqa_decode_attn",
    "fused_qkv_gqa",
    "build_inv_freq",
    "build_cos_sin",
    "apply_rope",
    "HAVE_TRITON_KERNELS",
]

HAVE_TRITON_KERNELS = False
_tk_rms = _tk_rope = _tk_rope_batched = _tk_swiglu = _tk_gqa = None
_tk_fused_gqa = None
try:
    from src.models.triton_kernels.rmsnorm import rmsnorm as _tk_rms  # noqa: F401
    from src.models.triton_kernels.rope import rope as _tk_rope  # noqa: F401
    from src.models.triton_kernels.rope import rope_batched as _tk_rope_batched  # noqa: F401
    from src.models.triton_kernels.activation import swiglu as _tk_swiglu  # noqa: F401
    from src.models.triton_kernels.attention import gqa_decode_attn as _tk_gqa  # noqa: F401
    from src.models.triton_kernels.attention import fused_qkv_gqa as _tk_fused_gqa  # noqa: F401
    HAVE_TRITON_KERNELS = True
except Exception:
    HAVE_TRITON_KERNELS = False


def _cuda(t):
    return isinstance(t, torch.Tensor) and t.is_cuda


def rmsnorm(x, w, eps=1e-6):
    """RMSNorm over last dim. Triton fast path, else exact torch."""
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
    """silu(gate) * up, fused. Triton fast path, else exact torch."""
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


def fused_qkv_gqa(x, wq, wk, wv, bq=None, bk=None, bv=None):
    """Fused QKV GEMV for GQA decode. Triton fast path, else 3x F.linear."""
    if HAVE_TRITON_KERNELS and _cuda(x):
        try:
            return _tk_fused_gqa(x, wq, wk, wv, bq, bk, bv)
        except Exception:
            pass
    return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))


def build_inv_freq(head_dim, theta=1000000.0):
    """1 / (theta ** (arange(0, dh, 2) / dh)), shape [dh/2]."""
    return 1.0 / (float(theta)
                  ** (torch.arange(0, head_dim, 2).float() / head_dim))


def build_cos_sin(max_pos, head_dim, theta=1000000.0, device="cuda:0",
                  dtype=None):
    """Cos/sin tables [max_pos, dh/2], matching HF rotary embedding."""
    inv = build_inv_freq(head_dim, theta)
    ang = torch.arange(max_pos).float().unsqueeze(1) * inv.unsqueeze(0)
    cos = torch.cos(ang).to(device)
    sin = torch.sin(ang).to(device)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def apply_rope(x, cos, sin, pos):
    """Half-rotation RoPE; Triton fast path when available."""
    if HAVE_TRITON_KERNELS and _cuda(x):
        return _tk_rope(x, cos, sin, pos)
    return rope(x, cos, sin, pos)
