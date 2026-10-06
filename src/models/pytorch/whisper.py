"""Pure-torch Whisper ops for the STT leg (no Triton).

Encoder layer: LayerNorm -> self-attn (SDPA, non-causal) -> gelu FF.
Decoder layer: LayerNorm -> causal self-attn w/ KV cache -> cross-attn -> FF.
Fused QKV is 3x F.linear (cuBLAS). Mirror of the old triton surface API so
callers read unchanged.
"""
import torch
import torch.nn.functional as F
D, H, DH, FF, XN = 512, 8, 64, 2048, 1500
MAXN=448; SCALE=0.125

HAVE_TRITON_KERNELS = False

def layernorm_torch(x,w,b,eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), w,b,eps)

def decode_attn_torch(q,K,V,scale):
    scores = torch.einsum("hd,hnd->hn", q.float(), K.float())*scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("hn,hnd->hd", probs, V)

def batched_decode_torch(q,K,V,scale):
    scores = torch.einsum("bhd,bhnd->bhn", q.float(), K.float())*scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("bhn,bhnd->bhd", probs, V)

def layernorm(x, w, b, eps=1e-5):
    return layernorm_torch(x, w, b, eps)


def row_softmax(x):
    return F.softmax(x, dim=-1)


def decode_attn(q, K, V, scale):
    """q [H,D], K/V [H,N,D] -> O [H,D]. Single-query decode attention."""
    return decode_attn_torch(q, K, V, scale)


def batched_decode_attn(q, K, V, scale):
    """q [B,H,D], K/V [B,H,N,D] -> O [B,H,D]."""
    return batched_decode_torch(q, K, V, scale)


def fused_qkv(x, wq, wk, wv, bq=None, bv=None, bk=None):
    """Fused QKV (whisper convention: q/v biased, k unbiased) as 3x linear."""
    return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))


def fused_qkv_batched(x, wq, wk, wv, bq=None, bv=None, bk=None):
    return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))


def enc_self_attention(x, w, prefix, n_heads=8, head_dim=64):
    """Encoder self-attention block (non-causal, SDPA). Returns out-proj result."""
    B, T, _ = x.shape
    q = F.linear(x, w[prefix + "self_attn.q_proj.weight"],
                 w[prefix + "self_attn.q_proj.bias"])
    k = F.linear(x, w[prefix + "self_attn.k_proj.weight"], None)
    v = F.linear(x, w[prefix + "self_attn.v_proj.weight"],
                 w[prefix + "self_attn.v_proj.bias"])
    q4 = q.view(B, T, n_heads, head_dim).transpose(1, 2)
    k4 = k.view(B, T, n_heads, head_dim).transpose(1, 2)
    v4 = v.view(B, T, n_heads, head_dim).transpose(1, 2)
    o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    o = o.transpose(1, 2).reshape(B, T, n_heads * head_dim)
    return F.linear(o, w[prefix + "self_attn.out_proj.weight"],
                    w[prefix + "self_attn.out_proj.bias"])


def decode_self_attention_step(x, w, prefix, sk, sv, n, scale=0.125,
                               n_heads=8, head_dim=64):
    """One decode step of decoder self-attention with KV-cache write."""
    q = F.linear(x, w[prefix + "self_attn.q_proj.weight"],
                 w[prefix + "self_attn.q_proj.bias"]).reshape(n_heads, head_dim)
    k1 = F.linear(x, w[prefix + "self_attn.k_proj.weight"], None).reshape(
        n_heads, head_dim)
    v1 = F.linear(x, w[prefix + "self_attn.v_proj.weight"],
                  w[prefix + "self_attn.v_proj.bias"]).reshape(n_heads, head_dim)
    sk[:, n] = k1
    sv[:, n] = v1
    o = decode_attn_torch(q, sk[:, :n + 1], sv[:, :n + 1], scale)
    return F.linear(o.reshape(n_heads * head_dim),
                    w[prefix + "self_attn.out_proj.weight"],
                    w[prefix + "self_attn.out_proj.bias"])


def decode_cross_attention_step(x, w, prefix, Kx, Vx, scale=0.125,
                                n_heads=8, head_dim=64):
    """One decode step of decoder cross-attention."""
    q = F.linear(x, w[prefix + "encoder_attn.q_proj.weight"],
                 w[prefix + "encoder_attn.q_proj.bias"]).reshape(n_heads, head_dim)
    o = decode_attn_torch(q, Kx, Vx, scale)
    return F.linear(o.reshape(n_heads * head_dim),
                    w[prefix + "encoder_attn.out_proj.weight"],
                    w[prefix + "encoder_attn.out_proj.bias"])


def whisper_encoder_layer(x, w, prefix, B=None):
    """Encoder layer: layernorm + SDPA self-attn + gelu FF, [B,T,D]."""
    h = layernorm_torch(x, w[prefix + "self_attn_layer_norm.weight"],
                        w[prefix + "self_attn_layer_norm.bias"])
    B_, T, _ = h.shape
    q = F.linear(h, w[prefix + "self_attn.q_proj.weight"],
                 w[prefix + "self_attn.q_proj.bias"])
    k = F.linear(h, w[prefix + "self_attn.k_proj.weight"], None)
    v = F.linear(h, w[prefix + "self_attn.v_proj.weight"],
                 w[prefix + "self_attn.v_proj.bias"])
    q4 = q.view(B_, T, H, DH).transpose(1, 2)
    k4 = k.view(B_, T, H, DH).transpose(1, 2)
    v4 = v.view(B_, T, H, DH).transpose(1, 2)
    o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    o = o.transpose(1, 2).reshape(B_, T, D)
    o = F.linear(o, w[prefix + "self_attn.out_proj.weight"],
                 w[prefix + "self_attn.out_proj.bias"])
    x = x + o
    h2 = layernorm_torch(x, w[prefix + "final_layer_norm.weight"],
                         w[prefix + "final_layer_norm.bias"])
    m = F.linear(h2, w[prefix + "fc1.weight"], w[prefix + "fc1.bias"])
    m = F.gelu(m)
    return x + F.linear(m, w[prefix + "fc2.weight"], w[prefix + "fc2.bias"])


def estimate_whisper_kv_mb(B, n_layers=6, H=8, maxn=448, Dh=64, bytes_per=2):
    return 2 * n_layers * B * H * maxn * Dh * bytes_per / (1024 ** 2)


# legacy fused names (same math, no Triton)
whisper_fused_encoder_layer = whisper_encoder_layer


def whisper_decoder_layer_torch(x,w,prefix,sk,sv,n,Kx,Vx):
    # x [B,D]
    B = x.shape[0] if x.dim()==2 else 1
    if x.dim()==1:
        x=x.unsqueeze(0)
        was=True
    else:
        was=False
    h = layernorm_torch(x, w[prefix+"self_attn_layer_norm.weight"], w[prefix+"self_attn_layer_norm.bias"])
    q = F.linear(h, w[prefix+"self_attn.q_proj.weight"], w[prefix+"self_attn.q_proj.bias"]).view(B,H,DH)
    k1 = F.linear(h, w[prefix+"self_attn.k_proj.weight"], None).view(B,H,DH)
    v1 = F.linear(h, w[prefix+"self_attn.v_proj.weight"], w[prefix+"self_attn.v_proj.bias"]).view(B,H,DH)
    if sk.dim()==3:
        sk[:,n]=k1[0]; sv[:,n]=v1[0]
        K=sk[:,:n+1]; V=sv[:,:n+1]
        o = decode_attn_torch(q[0], K, V, SCALE).reshape(1,-1)
    else:
        for b in range(B):
            sk[b,:,n]=k1[b]; sv[b,:,n]=v1[b]
        K=sk[:,:,:n+1,:]; V=sv[:,:,:n+1,:]
        o = batched_decode_torch(q, K, V, SCALE).reshape(B,-1)
    o = F.linear(o, w[prefix+"self_attn.out_proj.weight"], w[prefix+"self_attn.out_proj.bias"])
    x = x + o
    h2 = layernorm_torch(x, w[prefix+"encoder_attn_layer_norm.weight"], w[prefix+"encoder_attn_layer_norm.bias"])
    q2 = F.linear(h2, w[prefix+"encoder_attn.q_proj.weight"], w[prefix+"encoder_attn.q_proj.bias"]).view(B,H,DH)
    if Kx.dim()==3:
        Kxb=Kx.unsqueeze(0).expand(B,-1,-1,-1); Vxb=Vx.unsqueeze(0).expand(B,-1,-1,-1)
    else:
        Kxb=Kx; Vxb=Vx
    o2 = batched_decode_torch(q2, Kxb, Vxb, SCALE).reshape(B,-1)
    o2 = F.linear(o2, w[prefix+"encoder_attn.out_proj.weight"], w[prefix+"encoder_attn.out_proj.bias"])
    x = x + o2
    h3 = layernorm_torch(x, w[prefix+"final_layer_norm.weight"], w[prefix+"final_layer_norm.bias"])
    m = F.linear(h3, w[prefix+"fc1.weight"], w[prefix+"fc1.bias"])
    m = F.gelu(m)
    x = x + F.linear(m, w[prefix+"fc2.weight"], w[prefix+"fc2.bias"])
    return x[0] if was else x


# legacy fused name for the decoder (same math, no Triton)
whisper_fused_decoder_layer = whisper_decoder_layer_torch
