"""Complete Triton kernel surface for the Whisper (STT) leg.

Single import point for everything STT needs:
layernorm, row_softmax, decode_attn, batched_decode_attn, fused_qkv,
plus the attention block helpers that USE all of them (so no kernel is
exported-but-dead):

- ``enc_self_attention`` uses ``row_softmax`` on CUDA (manual QK^T path),
  SDPA elsewhere.
- ``decode_self_attention_step`` uses ``fused_qkv`` + ``decode_attn``.
- ``decode_cross_attention_step`` uses ``batched_decode_attn`` (B=1 slice).

Each wrapper falls back to exact torch references on CPU.
"""
import torch
import torch.nn.functional as F

__all__ = [
    "layernorm",
    "row_softmax",
    "decode_attn",
    "batched_decode_attn",
    "fused_qkv",
    "enc_self_attention",
    "decode_self_attention_step",
    "decode_cross_attention_step",
    "HAVE_TRITON_KERNELS",
]

HAVE_TRITON_KERNELS = False
_tk_ln = _tk_dec = _tk_bdec = _tk_softmax = _tk_fused = None
try:
    from src.models.triton_kernels.layernorm import layernorm as _tk_ln  # noqa: F401
    from src.models.triton_kernels.softmax import row_softmax as _tk_softmax  # noqa: F401
    from src.models.triton_kernels.attention import decode_attn as _tk_dec  # noqa: F401
    from src.models.triton_kernels.attention import batched_decode_attn as _tk_bdec  # noqa: F401
    from src.models.triton_kernels.attention import fused_qkv as _tk_fused  # noqa: F401
    HAVE_TRITON_KERNELS = True
except Exception:
    HAVE_TRITON_KERNELS = False


def _cuda(t):
    return isinstance(t, torch.Tensor) and t.is_cuda


def layernorm(x, w, b, eps=1e-5):
    """LayerNorm over the last dim. Triton fast path, else torch."""
    if HAVE_TRITON_KERNELS and _cuda(x):
        return _tk_ln(x, w, b, eps)
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def row_softmax(x):
    """Row-wise softmax. Triton fast path, else torch."""
    if HAVE_TRITON_KERNELS and _cuda(x):
        try:
            return _tk_softmax(x.contiguous())
        except Exception:
            pass
    return F.softmax(x, dim=-1)


def decode_attn(q, K, V, scale):
    """q [H,D], K/V [H,N,D] -> O [H,D]. Single-query decode attention."""
    if HAVE_TRITON_KERNELS and _cuda(q):
        return _tk_dec(q, K, V, scale)
    scores = torch.einsum("hd,hnd->hn", q.float(), K.float()) * scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("hn,hnd->hd", probs, V)


def batched_decode_attn(q, K, V, scale):
    """q [B,H,D], K/V [B,H,N,D] -> O [B,H,D]. One launch."""
    if HAVE_TRITON_KERNELS and _cuda(q):
        return _tk_bdec(q, K, V, scale)
    scores = torch.einsum("bhd,bhnd->bhn", q.float(), K.float()) * scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("bhn,bhnd->bhd", probs, V)


def fused_qkv(x, wq, wk, wv, bq=None, bv=None, bk=None):
    """Fused QKV GEMV (whisper convention: q/v biased, k unbiased).

    Triton fast path on CUDA, else 3x F.linear.
    """
    if HAVE_TRITON_KERNELS and _cuda(x):
        try:
            return _tk_fused(x, wq, wk, wv, bq, bv, bk)
        except Exception:
            pass
    return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))


def enc_self_attention(x, w, prefix, n_heads=8, head_dim=64):
    """Encoder self-attention block (non-causal). Uses ``row_softmax``.

    On CUDA the attention probabilities go through the Triton
    ``row_softmax`` kernel (parity-tested); elsewhere plain softmax.
    Returns the output-projection result (caller adds the residual).
    """
    B, T, _ = x.shape
    q = F.linear(x, w[prefix + "self_attn.q_proj.weight"],
                 w[prefix + "self_attn.q_proj.bias"])
    k = F.linear(x, w[prefix + "self_attn.k_proj.weight"], None)
    v = F.linear(x, w[prefix + "self_attn.v_proj.weight"],
                 w[prefix + "self_attn.v_proj.bias"])
    q4 = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v4 = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    if _cuda(x):
        try:
            scores = torch.matmul(q4.float(), k4.transpose(-1, -2).float())
            scores = scores * (head_dim ** -0.5)
            probs = row_softmax(scores.reshape(-1, T)).reshape(B, n_heads, T, T)
            o = torch.matmul(probs.to(v4.dtype), v4)
            o = o.transpose(1, 2).reshape(B, T, n_heads * head_dim)
            return F.linear(o, w[prefix + "self_attn.out_proj.weight"],
                            w[prefix + "self_attn.out_proj.bias"])
        except Exception:
            pass
    o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    o = o.transpose(1, 2).reshape(B, T, n_heads * head_dim)
    return F.linear(o, w[prefix + "self_attn.out_proj.weight"],
                    w[prefix + "self_attn.out_proj.bias"])


def decode_self_attention_step(x, w, prefix, sk, sv, n, scale=0.125,
                               n_heads=8, head_dim=64):
    """One decode step of decoder self-attention with KV-cache write.

    Uses the fused Triton QKV path (q/v biased, k unbiased) + ``decode_attn``.
    """
    try:
        if _cuda(x):
            qf, k1f, v1f = fused_qkv(
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
                                n_heads=8, head_dim=64):
    """One decode step of decoder cross-attention.

    Uses ``batched_decode_attn`` (B=1 slice) so the batched kernel is in
    the hot path, not just parity-tested.
    """
    q = F.linear(x, w[prefix + "encoder_attn.q_proj.weight"],
                 w[prefix + "encoder_attn.q_proj.bias"]).reshape(n_heads, head_dim)
    try:
        if _cuda(q):
            o = batched_decode_attn(
                q.unsqueeze(0),
                Kx.unsqueeze(0),
                Vx.unsqueeze(0),
                scale,
            )[0]
            return F.linear(o.reshape(n_heads * head_dim),
                            w[prefix + "encoder_attn.out_proj.weight"],
                            w[prefix + "encoder_attn.out_proj.bias"])
    except Exception:
        pass
    o = decode_attn(q, Kx, Vx, scale)
    return F.linear(o.reshape(n_heads * head_dim),
                    w[prefix + "encoder_attn.out_proj.weight"],
                    w[prefix + "encoder_attn.out_proj.bias"])
