"""Qwen attention blocks: causal SDPA prefill + fused GQA decode step."""
import torch
import torch.nn.functional as F

from models.qwen.kernels import gqa_decode_attn, rmsnorm, rope, rope_batched

__all__ = ["prefill_attention", "decode_attention_step"]


def prefill_attention(x, w, prefix, cos, sin, n_heads, n_kv_heads, head_dim,
                      cache=None, layer=0):
    """Full-sequence prefill attention with RoPE positions 0..T-1.

    Returns the output-projection result (caller adds the residual). When
    ``cache`` (a :class:`KVCache`) is given, K/V are stored for decode.
    """
    T = x.shape[0]
    q = F.linear(x, w[prefix + "self_attn.q_proj.weight"],
                 w[prefix + "self_attn.q_proj.bias"])
    k = F.linear(x, w[prefix + "self_attn.k_proj.weight"],
                 w[prefix + "self_attn.k_proj.bias"])
    v = F.linear(x, w[prefix + "self_attn.v_proj.weight"],
                 w[prefix + "self_attn.v_proj.bias"])
    # Batched RoPE: ONE launch per q/k for the whole prefill (positions
    # 0..T-1). The old per-position loop did 2T launches + 2T cats/layer.
    qr = rope_batched(q.reshape(T, n_heads, head_dim), cos, sin, 0).reshape(
        T, n_heads * head_dim)
    kr = rope_batched(k.reshape(T, n_kv_heads, head_dim), cos, sin, 0).reshape(
        T, n_kv_heads * head_dim)
    if cache is not None:
        cache.store_prefill(layer, kr.reshape(T, n_kv_heads, head_dim),
                             v.reshape(T, n_kv_heads, head_dim))
    q4 = qr.view(T, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k4 = kr.view(T, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k4 = k4.repeat_interleave(n_heads // n_kv_heads, dim=1)
    v4 = v.view(T, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    v4 = v4.repeat_interleave(n_heads // n_kv_heads, dim=1)
    o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)[0]
    o = o.transpose(0, 1).reshape(T, n_heads * head_dim)
    return F.linear(o, w[prefix + "self_attn.o_proj.weight"], None)


def decode_attention_step(x, w, prefix, cos, sin, pos, n_heads, n_kv_heads,
                          head_dim, scale, cache=None, layer=0,
                          Kcache=None, Vcache=None):
    """Single-token decode attention step with KV-cache write.

    Accepts either a :class:`KVCache` (``cache`` + ``layer``) or raw
    per-layer ``Kcache``/``Vcache`` tensors (legacy call path).

    Tries the fused Triton GQA-QKV path (q/k/v ALL biased -- VERIFIED
    from the code: every proj passes ``w[...bias]``); falls back to the
    torch path on any failure.
    """
    d = n_heads * head_dim
    # NOTE (measured in-engine): the fused Triton QKV path is FASTER here
    # (31ms/10tok) than eager 3xF.linear (83ms/10tok) — the isolated 0.68x
    # microbench did not reproduce in situ, so the fused path stays primary
    # with the torch path as fallback. Trust in-engine numbers.
    try:
        if isinstance(x, torch.Tensor) and x.is_cuda:
            from triton_kernels.attention import fused_qkv_gqa as _fused_qkv_gqa
            qf, kf, vf = _fused_qkv_gqa(
                x.reshape(-1).contiguous(),
                w[prefix + "self_attn.q_proj.weight"],
                w[prefix + "self_attn.k_proj.weight"],
                w[prefix + "self_attn.v_proj.weight"],
                w[prefix + "self_attn.q_proj.bias"],
                w[prefix + "self_attn.k_proj.bias"],
                w[prefix + "self_attn.v_proj.bias"],
            )
            q = rope(qf.reshape(n_heads, head_dim), cos, sin, pos)
            k1 = rope(kf.reshape(n_kv_heads, head_dim), cos, sin, pos)
            v1 = vf.reshape(n_kv_heads, head_dim)
        else:
            raise RuntimeError("CPU: use torch path")
    except Exception:
        q = rope(F.linear(x, w[prefix + "self_attn.q_proj.weight"],
                          w[prefix + "self_attn.q_proj.bias"]).reshape(
                              n_heads, head_dim), cos, sin, pos)
        k1 = rope(F.linear(x, w[prefix + "self_attn.k_proj.weight"],
                           w[prefix + "self_attn.k_proj.bias"]).reshape(
                               n_kv_heads, head_dim), cos, sin, pos)
        v1 = F.linear(x, w[prefix + "self_attn.v_proj.weight"],
                      w[prefix + "self_attn.v_proj.bias"]).reshape(n_kv_heads,
                                                                   head_dim)
    if cache is not None:
        cache.store_step(layer, pos, k1, v1)
        K, V = cache.get(layer, pos + 1)
    else:
        Kcache[layer][:, pos] = k1
        Vcache[layer][:, pos] = v1
        K, V = Kcache[layer][:, :pos + 1], Vcache[layer][:, :pos + 1]
    o = gqa_decode_attn(q, K, V, scale)
    o = o.to(w[prefix + "self_attn.o_proj.weight"].dtype)
    return F.linear(o.reshape(d), w[prefix + "self_attn.o_proj.weight"], None)
