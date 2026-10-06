"""Pure-torch TTS post-processing (no Triton).

Postprocess = DC-remove -> peak-normalize -> optional InstanceNorm+SiLU
enhance -> renormalize. Resample = F.interpolate linear (explainable).
"""
import torch
import torch.nn.functional as F
import numpy as np

HAVE_TRITON_KERNELS = False

def in1d_silu_torch(x, eps=1e-5):
    return F.silu(F.instance_norm(x.float())).to(x.dtype)

def conv1d_silu_torch(x, weight, bias=None, stride=1, padding=0, dilation=1):
    return F.silu(F.conv1d(x, weight, bias, stride, padding, dilation))

def conv1d_silu(x, weight, bias=None, stride=1, padding=0, dilation=1):
    return F.silu(F.conv1d(x.float(), weight.float(),
                           bias.float() if bias is not None else None,
                           stride, padding, dilation)).to(x.dtype)


def conv1d_silu_fwd(x, weight, bias=None, stride=1, padding=0, dilation=1):
    return conv1d_silu(x, weight, bias, stride, padding, dilation)


def in1d_silu(x, eps=1e-5):
    return F.silu(F.instance_norm(x.float())).to(x.dtype)


def in1d_silu_fwd(x, eps=1e-5):
    return in1d_silu(x, eps)


def resample_linear(wav, sr_in, sr_out):
    x = np.asarray(wav, dtype=np.float32).ravel()
    if sr_in == sr_out or x.size == 0:
        return x
    t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    n_out = max(1, int(round(x.size * sr_out / float(sr_in))))
    y = F.interpolate(t, size=n_out, mode="linear", align_corners=False).squeeze()
    return y.numpy().astype(np.float32)


def resample_batched(wavs, sr_in, sr_out):
    return [resample_linear(w, sr_in, sr_out) for w in wavs]


def postprocess(wav, sr=24000, enhance=False, peak=0.98):
    return postprocess_torch(wav, sr, enhance, peak)


def postprocess_batched(wavs, sr=24000, enhance=False, peak=0.98, batch_size=8):
    if isinstance(wavs, torch.Tensor):
        if wavs.dim() == 2:
            wavs = [wavs[b].cpu().numpy() for b in range(wavs.shape[0])]
        else:
            wavs = [wavs[b, 0].cpu().numpy() if wavs.dim() == 3
                    else wavs[b].cpu().numpy() for b in range(wavs.shape[0])]
    return [postprocess_torch(w, sr, enhance, peak) for w in wavs]


def estimate_tts_mb(B, L, C=1, bytes_per=4):
    return B * C * L * bytes_per / (1024 ** 2) + 50  # +50MB kokoro overhead


def postprocess_torch(wav, sr=24000, enhance=False, peak=0.98):
    x=np.asarray(wav, dtype=np.float32).ravel()
    if x.size==0:
        return x
    x=x-float(x.mean())
    m=float(np.abs(x).max()) if x.size else 0.0
    if m>1e-9:
        x=(x/m*float(peak)).astype(np.float32)
    if enhance and x.size>=16:
        try:
            t=torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
            y=in1d_silu_torch(t.float())
            x=y.float().numpy().ravel().astype(np.float32)
            m2=float(np.abs(x).max()) if x.size else 0.0
            if m2>1e-9:
                x=(x/m2*float(peak)).astype(np.float32)
        except Exception:
            pass
    return x
