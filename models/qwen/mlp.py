"""Qwen MLP block: SiLU-gated (SwiGLU) feed-forward."""
import torch.nn.functional as F

from models.qwen.kernels import swiglu

__all__ = ["mlp_forward"]


def mlp_forward(h, w, prefix):
    """Gated MLP output-projection result (caller adds the residual)."""
    g = F.linear(h, w[prefix + "mlp.gate_proj.weight"], None)
    u = F.linear(h, w[prefix + "mlp.up_proj.weight"], None)
    return F.linear(swiglu(g, u), w[prefix + "mlp.down_proj.weight"], None)
