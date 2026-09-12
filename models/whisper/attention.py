"""Whisper attention blocks: SDPA encoder self-attn + fused decode attn steps.

Shapes follow the original ``STT/triton/engine.py`` port:
D=512 model dim, H=8 heads, DH=64 head dim, XN=1500 encoder frames.
"""
import torch.nn.functional as F

from models.whisper.kernels import decode_attn

__all__ = [
    "D",
    "H",
    "DH",
    "FF",
    "XN",
    "enc_self_attention",
    "decode_self_attention_step",
    "decode_cross_attention_step",
]

D, H, DH, FF, XN = 512, 8, 64, 2048, 1500


def enc_self_attention(x, w, prefix, n_heads=H, head_dim=DH):
    """Encoder self-attention block (SDPA, non-causal). Returns projection
    output (caller adds the residual), matching the source engine."""
    B, T, _ = x.shape
    h = x
    q = F.linear(h, w[prefix + "self_attn.q_proj.weight"],
                 w[prefix + "self_attn.q_proj.bias"])
    k = F.linear(h, w[prefix + "self_attn.k_proj.weight"], None)
    v = F.linear(h, w[prefix + "self_attn.v_proj.weight"],
                 w[prefix + "self_attn.v_proj.bias"])
    q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    o = o.transpose(1, 2).reshape(B, T, n_heads * head_dim)
    return F.linear(o, w[prefix + "self_attn.out_proj.weight"],
                    w[prefix + "self_attn.out_proj.bias"])


def decode_self_attention_step(x, w, prefix, sk, sv, n, scale=0.125,
                               n_heads=H, head_dim=DH):
    """One decode step of decoder self-attention with KV-cache write.

    ``x`` is [D]; ``sk``/``sv`` are [H, MAXN, DH] caches. Returns the
    output-projection result (caller adds the residual).

    Tries the fused Triton QKV path (q/v biased, k unbiased -- VERIFIED
    from the weight/code convention: k_proj is always called with
    ``None`` bias); falls back to the torch path on any failure.
    """
    try:
        if x.is_cuda:
            from triton_kernels.attention import fused_qkv as _fused_qkv
            qf, k1f, v1f = _fused_qkv(
                x.reshape(-1).contiguous(),
                w[prefix + "self_attn.q_proj.weight"],
                w[prefix + "self_attn.k_proj.weight"],
                w[prefix + "self_attn.v_proj.weight"],
                w[prefix + "self_attn.q_proj.bias"],
                w[prefix + "self_attn.v_proj.bias"],
                None,
            )
            q = qf.reshape(n_heads, head_dim)
            k1 = k1f.reshape(n_heads, head_dim)
            v1 = v1f.reshape(n_heads, head_dim)
        else:
            raise RuntimeError("CPU: use torch path")
    except Exception:
        q = F.linear(x, w[prefix + "self_attn.q_proj.weight"],
                     w[prefix + "self_attn.q_proj.bias"]).reshape(n_heads, head_dim)
        k1 = F.linear(x, w[prefix + "self_attn.k_proj.weight"], None).reshape(
            n_heads, head_dim)
        v1 = F.linear(x, w[prefix + "self_attn.v_proj.weight"],
                      w[prefix + "self_attn.v_proj.bias"]).reshape(n_heads, head_dim)
    sk[:, n] = k1
    sv[:, n] = v1
    o = decode_attn(q, sk[:, :n + 1], sv[:, :n + 1], scale)
    return F.linear(o.reshape(n_heads * head_dim),
                    w[prefix + "self_attn.out_proj.weight"],
                    w[prefix + "self_attn.out_proj.bias"])


def decode_cross_attention_step(x, w, prefix, Kx, Vx, scale=0.125,
                                n_heads=H, head_dim=DH):
    """One decode step of decoder cross-attention over encoder KV."""
    q = F.linear(x, w[prefix + "encoder_attn.q_proj.weight"],
                 w[prefix + "encoder_attn.q_proj.bias"]).reshape(n_heads, head_dim)
    o = decode_attn(q, Kx, Vx, scale)
    return F.linear(o.reshape(n_heads * head_dim),
                    w[prefix + "encoder_attn.out_proj.weight"],
                    w[prefix + "encoder_attn.out_proj.bias"])
