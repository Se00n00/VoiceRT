"""PyTorch reference for Whisper decoder layer (no Triton)."""
import torch
import torch.nn.functional as F
D, H, DH, FF, XN = 512, 8, 64, 2048, 1500
MAXN=448; SCALE=0.125

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
