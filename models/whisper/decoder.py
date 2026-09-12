"""Whisper decoder: single-step layer + greedy decode loop.

Ported from ``STT/triton/engine.py`` (``WhisperTriton._dec_layer`` /
``WhisperTriton.transcribe``).
"""
import torch
import torch.nn.functional as F

from models.whisper.attention import (D, decode_cross_attention_step,
                                      decode_self_attention_step)
from models.whisper.kernels import layernorm
from models.whisper.tokenizer import EOT, SOT

__all__ = ["N_DEC_LAYERS", "MAXN", "decoder_step", "greedy_decode"]

N_DEC_LAYERS = 6
MAXN = 448


def _ln(x, w, b):
    return layernorm(x.reshape(-1, D).contiguous(), w, b).reshape(x.shape)


def decoder_step(x, w, i, sk, sv, n, Kx, Vx):
    """One decoder layer at position ``n``. x [D] -> [D]."""
    p = "model.decoder.layers.%d." % i
    h = _ln(x, w[p + "self_attn_layer_norm.weight"],
            w[p + "self_attn_layer_norm.bias"])
    x = x + decode_self_attention_step(h, w, p, sk, sv, n)
    h = _ln(x, w[p + "encoder_attn_layer_norm.weight"],
            w[p + "encoder_attn_layer_norm.bias"])
    x = x + decode_cross_attention_step(h, w, p, Kx, Vx)
    h = _ln(x, w[p + "final_layer_norm.weight"],
            w[p + "final_layer_norm.bias"])
    m = F.linear(h, w[p + "fc1.weight"], w[p + "fc1.bias"])
    return x + F.linear(F.gelu(m), w[p + "fc2.weight"], w[p + "fc2.bias"])


@torch.no_grad()
def greedy_decode(w, cross, device, max_tokens=MAXN, sot=SOT, eot=EOT,
                  n_layers=N_DEC_LAYERS):
    """Greedy loop over cached cross KV -> token id list."""
    sk = [torch.empty(8, MAXN, 64, device=device) for _ in range(n_layers)]
    sv = [torch.empty(8, MAXN, 64, device=device) for _ in range(n_layers)]
    tok = torch.tensor([sot], device=device)
    ids = []
    for n in range(min(max_tokens, MAXN)):
        e = F.embedding(tok, w["model.decoder.embed_tokens.weight"])
        e = e + w["model.decoder.embed_positions.weight"][n]
        x = e.reshape(D)
        for i in range(n_layers):
            x = decoder_step(x, w, i, sk[i], sv[i], n, *cross[i])
        x = layernorm(x.reshape(1, D), w["model.decoder.layer_norm.weight"],
                      w["model.decoder.layer_norm.bias"]).reshape(D)
        # LM head is tied to decoder embeddings (no separate tensor/bias)
        nxt = F.linear(
            x, w["model.decoder.embed_tokens.weight"]).argmax().item()
        ids.append(nxt)
        if nxt == eot:
            break
        tok = torch.tensor([nxt], device=device)
    return ids
