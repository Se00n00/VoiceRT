"""RoPE helpers for Qwen: inverse-frequency table + cos/sin cache.

Uses ``triton_kernels.rope`` when importable, else builds the tables with
torch (measured HF convention: theta=1e6 for Qwen2.5-0.5B).
"""
import torch

__all__ = ["build_inv_freq", "build_cos_sin", "apply_rope"]

try:
    from triton_kernels.rope import rope as _tk_rope  # noqa: F401
    _HAVE_TK_ROPE = True
except Exception:
    try:
        from triton_kernels.llm import rope as _tk_rope  # noqa: F401
        _HAVE_TK_ROPE = True
    except Exception:
        _tk_rope = None
        _HAVE_TK_ROPE = False


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
    if _HAVE_TK_ROPE and isinstance(x, torch.Tensor) and x.is_cuda:
        return _tk_rope(x, cos, sin, pos)
    from models.qwen.kernels import rope as _rope
    return _rope(x, cos, sin, pos)
