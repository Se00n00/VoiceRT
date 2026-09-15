"""PyTorch reference for TTS post-processing (no Triton)."""
import torch
import torch.nn.functional as F
import numpy as np

def in1d_silu_torch(x, eps=1e-5):
    return F.silu(F.instance_norm(x.float())).to(x.dtype)

def conv1d_silu_torch(x, weight, bias=None, stride=1, padding=0, dilation=1):
    return F.silu(F.conv1d(x, weight, bias, stride, padding, dilation))

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
