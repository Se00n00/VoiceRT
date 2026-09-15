"""PyTorch reference for Qwen decode layer & full model (no Triton, for comparison)."""
import math
import torch
import torch.nn.functional as F

def rmsnorm_torch(x, w, eps=1e-6):
    xf = x.float()
    var = (xf * xf).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * w.float()).to(x.dtype).reshape(x.shape)

def rope_torch(x, cos, sin, pos):
    shp = x.shape
    dh = shp[-1]
    half = dh // 2
    if isinstance(pos, torch.Tensor) and pos.dim()==1 and x.dim()==3:
        outs=[]
        for b in range(x.shape[0]):
            outs.append(rope_torch(x[b], cos, sin, int(pos[b].item())))
        return torch.stack(outs,0)
    xf = x.reshape(-1, dh).float()
    c = cos[int(pos)].to(torch.float32)
    s = sin[int(pos)].to(torch.float32)
    x1, x2 = xf[:, :half], xf[:, half:]
    y = torch.cat([x1*c - x2*s, x1*s + x2*c], dim=-1)
    return y.to(x.dtype).reshape(shp)

def swiglu_torch(gate, up):
    return F.silu(gate.float()).to(gate.dtype) * up

def gqa_decode_torch(q, K, V, scale):
    # q [Hq,D], K/V [Hk,N,D]
    Hq = q.shape[0]; Hk = K.shape[0]; group = Hq//Hk
    Ke = K.repeat_interleave(group, dim=0).float()
    Ve = V.repeat_interleave(group, dim=0).float()
    scores = torch.einsum("hd,hnd->hn", q.float(), Ke) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hn,hnd->hd", probs, Ve)
    return out.to(q.dtype)

def batched_gqa_torch(q, K, V, scale):
    # q [B,Hq,D], K/V [B,Hk,N,D]
    B, Hq, D = q.shape; Hk = K.shape[1]; group = Hq//Hk
    Ke = K.repeat_interleave(group, dim=1).float()
    Ve = V.repeat_interleave(group, dim=1).float()
    scores = torch.einsum("bhd,bhnd->bhn", q.float(), Ke) * scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhn,bhnd->bhd", probs, Ve)
    return out.to(q.dtype)

def build_cos_sin_torch(max_pos, head_dim, theta=1000000.0, device="cpu", dtype=None):
    inv = 1.0 / (float(theta) ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.arange(max_pos).float().unsqueeze(1) * inv.unsqueeze(0)
    cos = torch.cos(ang).to(device)
    sin = torch.sin(ang).to(device)
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin

def qwen_decode_layer_torch(x, Kcache, Vcache, pos, cos, sin, w, prefix, H, Hk, Dh, scale, eps=1e-6, qk_norm=None):
    """Torch reference for one decode layer with batching+KV cache. x [B, hidden]"""
    B, hidden = x.shape
    h = rmsnorm_torch(x, w[prefix+"input_layernorm.weight"], eps)
    qf = F.linear(h, w[prefix+"self_attn.q_proj.weight"], w.get(prefix+"self_attn.q_proj.bias"))
    kf = F.linear(h, w[prefix+"self_attn.k_proj.weight"], w.get(prefix+"self_attn.k_proj.bias"))
    vf = F.linear(h, w[prefix+"self_attn.v_proj.weight"], w.get(prefix+"self_attn.v_proj.bias"))
    q = qf.view(B, H, Dh); k1 = kf.view(B, Hk, Dh); v1 = vf.view(B, Hk, Dh)
    if qk_norm is not None:
        qw, kw = qk_norm
        q = rmsnorm_torch(q.reshape(B*H, Dh), qw).reshape(B, H, Dh)
        k1 = rmsnorm_torch(k1.reshape(B*Hk, Dh), kw).reshape(B, Hk, Dh)
    if isinstance(pos, int):
        for b in range(B):
            q[b] = rope_torch(q[b], cos, sin, pos)
            k1[b] = rope_torch(k1[b], cos, sin, pos)
        Kcache[:, :, pos, :] = k1; Vcache[:, :, pos, :] = v1
        N = pos+1
        K = Kcache[:, :, :N, :]; V = Vcache[:, :, :N, :]
        attn = batched_gqa_torch(q, K, V, scale)
        o = F.linear(attn.reshape(B, -1), w[prefix+"self_attn.o_proj.weight"], None)
        x = x + o
        h2 = rmsnorm_torch(x, w[prefix+"post_attention_layernorm.weight"], eps)
        gate = F.linear(h2, w[prefix+"mlp.gate_proj.weight"], None)
        up = F.linear(h2, w[prefix+"mlp.up_proj.weight"], None)
        mlp = F.linear(swiglu_torch(gate, up), w[prefix+"mlp.down_proj.weight"], None)
        return x + mlp
    else:
        # variable pos
        if len(set([int(p.item()) if isinstance(p, torch.Tensor) else int(p) for p in pos]))==1:
            p0 = int(pos[0].item()) if isinstance(pos[0], torch.Tensor) else int(pos[0])
            for b in range(B):
                q[b] = rope_torch(q[b], cos, sin, p0)
                k1[b] = rope_torch(k1[b], cos, sin, p0)
            Kcache[:, :, p0, :] = k1; Vcache[:, :, p0, :] = v1
            K = Kcache[:, :, :p0+1, :]; V = Vcache[:, :, :p0+1, :]
            attn = batched_gqa_torch(q, K, V, scale)
            o = F.linear(attn.reshape(B, -1), w[prefix+"self_attn.o_proj.weight"], None)
            x = x + o
            h2 = rmsnorm_torch(x, w[prefix+"post_attention_layernorm.weight"], eps)
            gate = F.linear(h2, w[prefix+"mlp.gate_proj.weight"], None)
            up = F.linear(h2, w[prefix+"mlp.up_proj.weight"], None)
            mlp = F.linear(swiglu_torch(gate, up), w[prefix+"mlp.down_proj.weight"], None)
            return x + mlp
        else:
            outs=[]
            wo = w[prefix+"self_attn.o_proj.weight"]
            for b in range(B):
                p = int(pos[b].item()) if isinstance(pos[b], torch.Tensor) else int(pos[b])
                q[b] = rope_torch(q[b].unsqueeze(0), cos, sin, p)[0]
                k1[b] = rope_torch(k1[b].unsqueeze(0), cos, sin, p)[0]
                Kcache[b, :, p, :] = k1[b]; Vcache[b, :, p, :] = v1[b]
                Kb = Kcache[b:b+1, :, :p+1, :]; Vb = Vcache[b:b+1, :, :p+1, :]
                qb = q[b:b+1]
                ob = batched_gqa_torch(qb, Kb, Vb, scale)[0]
                outs.append(F.linear(ob.reshape(-1), wo, None))
            o = torch.stack(outs,0)
            x = x + o
            h2 = rmsnorm_torch(x, w[prefix+"post_attention_layernorm.weight"], eps)
            gate = F.linear(h2, w[prefix+"mlp.gate_proj.weight"], None)
            up = F.linear(h2, w[prefix+"mlp.up_proj.weight"], None)
            mlp = F.linear(swiglu_torch(gate, up), w[prefix+"mlp.down_proj.weight"], None)
            return x + mlp
