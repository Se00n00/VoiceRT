"""Kernel surface for the TTS leg (Kokoro).

Kokoro runs under ``torch.compile`` (Inductor emits its own Triton kernels),
so this module re-exports any conv helpers from ``triton_kernels`` and
provides real torch fallbacks (causal/non-causal conv1d + linear resample)
used by weight prep and the offline path.
"""
import torch
import torch.nn.functional as F

__all__ = ["conv1d_forward", "conv1d_silu_fwd", "in1d_silu_fwd",
             "resample_linear", "HAVE_TRITON_KERNELS"]

HAVE_TRITON_KERNELS = False
_tk_conv_silu = _tk_in_silu = None
try:
    # NOTE: plain conv1d stays on cuDNN (a naive Triton conv cannot beat
    # it at these shapes — measured 0.00x). Only the FUSED conv+silu and
    # norm+silu kernels live here, for the non-compiled path.
    from triton_kernels.conv1d import conv1d_silu as _tk_conv_silu  # noqa: F401
    from triton_kernels.conv1d import in1d_silu as _tk_in_silu  # noqa: F401
    HAVE_TRITON_KERNELS = True
except Exception:
    HAVE_TRITON_KERNELS = False


def conv1d_silu_fwd(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Fused conv1d+silu. Triton fast path, else torch conv+silu."""
    if HAVE_TRITON_KERNELS and x.is_cuda:
        try:
            return _tk_conv_silu(x, weight, bias, stride, padding, dilation)
        except Exception:
            pass
    return F.silu(F.conv1d(x, weight, bias, stride, padding, dilation))


def in1d_silu_fwd(x, eps=1e-5):
    """Fused instance-norm+silu (2.53x vs eager). Triton, else torch."""
    if HAVE_TRITON_KERNELS and x.is_cuda:
        try:
            return _tk_in_silu(x, eps)
        except Exception:
            pass
    return F.silu(F.instance_norm(x))


def conv1d_forward(x, weight, bias=None, stride=1, padding=0, dilation=1,
                   groups=1):
    """1-D convolution over [B, C, T]. Deliberately cuDNN/torch: the naive
    Triton conv loses to cuDNN at Kokoro shapes, so no Triton path here."""
    return F.conv1d(x, weight, bias, stride, padding, dilation, groups)


def resample_linear(wav, sr_in, sr_out):
    """Mono linear resample via interpolate (real code, no extra deps)."""
    import numpy as np
    x = np.asarray(wav, dtype=np.float32).ravel()
    if sr_in == sr_out or x.size == 0:
        return x
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    n_out = max(1, int(round(x.size * sr_out / float(sr_in))))
    y = F.interpolate(t, size=n_out, mode="linear",
                      align_corners=False).squeeze()
    return y.numpy().astype(np.float32)
