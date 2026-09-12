"""Whisper encoder: conv frontend + stacked self-attn layers + cross KV.

Ported from ``STT/triton/engine.py`` (``WhisperTriton._enc_layer`` /
``WhisperTriton.encode``). Operates on a flat weight dict as loaded by
:mod:`models.whisper.weights`.
"""
import torch.nn.functional as F

from models.whisper.attention import D, H, XN, enc_self_attention
from models.whisper.kernels import layernorm

__all__ = ["N_ENC_LAYERS", "encoder_layer", "encode_mel", "cross_kv"]

N_ENC_LAYERS = 6


def _ln(x, w, b):
    # reshape may return a non-contiguous VIEW; kernels need dense rows
    return layernorm(x.reshape(-1, D).contiguous(), w, b).reshape(x.shape)


def encoder_layer(x, w, i):
    """One encoder layer with residual adds. x [B, T, D] -> [B, T, D]."""
    p = "model.encoder.layers.%d." % i
    h = _ln(x, w[p + "self_attn_layer_norm.weight"],
            w[p + "self_attn_layer_norm.bias"])
    x = x + enc_self_attention(h, w, p)
    h = _ln(x, w[p + "final_layer_norm.weight"],
            w[p + "final_layer_norm.bias"])
    m = F.linear(h, w[p + "fc1.weight"], w[p + "fc1.bias"])
    m = F.gelu(m)
    return x + F.linear(m, w[p + "fc2.weight"],
                       w[p + "fc2.bias"]).view(x.shape[0], x.shape[1], D)


def encode_mel(mel, w, n_layers=N_ENC_LAYERS):
    """mel [B,80,3000] -> memory [B,1500,512]."""
    h = F.conv1d(mel, w["model.encoder.conv1.weight"],
                 w["model.encoder.conv1.bias"], padding=1)
    h = F.gelu(h)
    h = F.conv1d(h, w["model.encoder.conv2.weight"],
                 w["model.encoder.conv2.bias"], stride=2, padding=1)
    h = F.gelu(h).transpose(1, 2)
    h = h + w["model.encoder.embed_positions.weight"][:h.shape[1]]
    for i in range(n_layers):
        h = encoder_layer(h, w, i)
    h = layernorm(h.reshape(-1, D), w["model.encoder.layer_norm.weight"],
                  w["model.encoder.layer_norm.bias"]).reshape(h.shape)
    return h


def cross_kv(memory, w, n_layers=N_ENC_LAYERS):
    """memory [B,1500,512] -> per-layer (K, V) cross-attention caches."""
    B = memory.shape[0]
    cross = []
    for i in range(n_layers):
        p = "model.decoder.layers.%d.encoder_attn." % i
        k = F.linear(memory, w[p + "k_proj.weight"], None)
        v = F.linear(memory, w[p + "v_proj.weight"], w[p + "v_proj.bias"])
        cross.append((
            k.view(B, XN, H, 64).transpose(1, 2),
            v.view(B, XN, H, 64).transpose(1, 2),
        ))
    # Squeeze the batch dim for the single-utterance path.
    if B == 1:
        cross = [(k[0].reshape(H, XN, 64), v[0].reshape(H, XN, 64))
                 for k, v in cross]
    return cross
