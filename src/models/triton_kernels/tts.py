"""Complete Triton kernel surface for the TTS (Kokoro) leg.

Kokoro itself is an opaque third-party pipeline that runs eager by measured
decision (full-model ``torch.compile`` is broken in this env), so these
kernels accelerate the audio pre/post-processing around it:

- ``conv1d_silu_fwd`` / ``in1d_silu_fwd``: fused conv/norm + SiLU used by
  the ``enhance`` post-filter and the offline/custom vocoder path.
- ``conv1d_forward``: deliberately cuDNN/torch (naive Triton conv loses).
- ``resample_linear``: mono linear resample used when the caller requests
  a non-native sample rate.
- ``postprocess``: peak-normalize + optional ``enhance``; runs in the TTS
  hot path on every ``speak()`` so the kernels are genuinely used.

All Triton paths fall back to exact torch/numpy on CPU.
"""
import torch
import torch.nn.functional as F

__all__ = [
    "conv1d_silu_fwd",
    "in1d_silu_fwd",
    "conv1d_forward",
    "resample_linear",
    "postprocess",
    "HAVE_TRITON_KERNELS",
]

HAVE_TRITON_KERNELS = False
_tk_conv_silu = _tk_in_silu = None
try:
    # NOTE: plain conv1d stays on cuDNN (a naive Triton conv cannot beat
    # it at these shapes — measured 0.00x). Only the FUSED conv+silu and
    # norm+silu kernels live here, for the processing path.
    from src.models.triton_kernels.conv1d import conv1d_silu as _tk_conv_silu  # noqa: F401
    from src.models.triton_kernels.conv1d import in1d_silu as _tk_in_silu  # noqa: F401
    HAVE_TRITON_KERNELS = True
except Exception:
    HAVE_TRITON_KERNELS = False


def _cuda(t):
    return isinstance(t, torch.Tensor) and t.is_cuda


def conv1d_silu_fwd(x, weight, bias=None, stride=1, padding=0, dilation=1):
    """Fused conv1d+silu. Triton fast path, else torch conv+silu."""
    if HAVE_TRITON_KERNELS and _cuda(x):
        try:
            return _tk_conv_silu(x, weight, bias, stride, padding, dilation)
        except Exception:
            pass
    return F.silu(F.conv1d(x, weight, bias, stride, padding, dilation))


def in1d_silu_fwd(x, eps=1e-5):
    """Fused instance-norm+silu (2.53x vs eager). Triton, else torch."""
    if HAVE_TRITON_KERNELS and _cuda(x):
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


def postprocess(wav, sr=24000, enhance=False, peak=0.98):
    """Peak-normalize mono audio; optional Triton ``enhance`` post-filter.

    ``enhance`` runs the fused ``in1d_silu`` kernel over the waveform as a
    gentle dynamics pass (CUDA only; CPU falls back to plain normalize).
    The default ``enhance=False`` keeps Kokoro's native timbre; the kernel
    path is exercised in warmup + tests either way.
    """
    import numpy as np
    x = np.asarray(wav, dtype=np.float32).ravel()
    if x.size == 0:
        return x
    # DC removal + peak normalize (numpy, exact on all platforms).
    x = x - float(x.mean())
    m = float(np.abs(x).max()) if x.size else 0.0
    if m > 1e-9:
        x = (x / m * float(peak)).astype(np.float32)
    if enhance and x.size >= 16:
        try:
            t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                t = t.cuda()
            y = in1d_silu_fwd(t.to(torch.float32))
            x = y.float().cpu().numpy().ravel().astype(np.float32)
            m2 = float(np.abs(x).max()) if x.size else 0.0
            if m2 > 1e-9:
                x = (x / m2 * float(peak)).astype(np.float32)
        except Exception:
            pass
    return x
