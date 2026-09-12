"""Kernel surface for the Whisper leg.

Tries to re-export the fused kernels (layernorm, decode attention) from
``triton_kernels``; when that package (or CUDA/triton) is unavailable it
falls back to real torch implementations with identical shapes/semantics so
the engine still runs (e.g. on CPU).
"""
import torch
import torch.nn.functional as F

__all__ = [
    "layernorm",
    "row_softmax",
    "decode_attn",
    "batched_decode_attn",
    "HAVE_TRITON_KERNELS",
]

HAVE_TRITON_KERNELS = False
_tk_ln = _tk_dec = _tk_bdec = _tk_softmax = None
try:
    from triton_kernels.whisper import (  # noqa: F401
        batched_decode_attn as _tk_bdec,
        decode_attn as _tk_dec,
        layernorm as _tk_ln,
        row_softmax as _tk_softmax,
    )
    HAVE_TRITON_KERNELS = True
except Exception:
    try:
        from triton_kernels.layernorm import layernorm as _tk_ln  # noqa: F401
        from triton_kernels.softmax import row_softmax as _tk_softmax  # noqa: F401
        from triton_kernels.attention import (  # noqa: F401
            batched_decode_attn as _tk_bdec,
            decode_attn as _tk_dec,
        )
        HAVE_TRITON_KERNELS = True
    except Exception:
        HAVE_TRITON_KERNELS = False


def layernorm(x, w, b, eps=1e-5):
    """LayerNorm over the last dim. Triton fast path, else torch."""
    if HAVE_TRITON_KERNELS and x.is_cuda:
        return _tk_ln(x, w, b, eps)
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def row_softmax(x):
    """Row-wise softmax. Triton fast path, else torch."""
    if HAVE_TRITON_KERNELS and x.is_cuda:
        return _tk_softmax(x)
    return F.softmax(x, dim=-1)


def decode_attn(q, K, V, scale):
    """q [H,D], K/V [H,N,D] -> O [H,D]. Single-query decode attention."""
    if HAVE_TRITON_KERNELS and q.is_cuda:
        return _tk_dec(q, K, V, scale)
    scores = torch.einsum("hd,hnd->hn", q.float(), K.float()) * scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("hn,hnd->hd", probs, V)


def batched_decode_attn(q, K, V, scale):
    """q [B,H,D], K/V [B,H,N,D] -> O [B,H,D]."""
    if HAVE_TRITON_KERNELS and q.is_cuda:
        return _tk_bdec(q, K, V, scale)
    scores = torch.einsum("bhd,bhnd->bhn", q.float(), K.float()) * scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("bhn,bhnd->bhd", probs, V)
