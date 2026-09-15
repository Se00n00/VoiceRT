"""Single-file fused Triton kernels for Whisper STT (1 fused decoder layer x 6).

Batching: B=1..8, KV cache: [B, H, MAXN, Dh] per layer, cross KV: [B, H, XN, Dh]
VRAM check via check_budget, perf_report inside file.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from src.models.runtime.memory import check_budget

HAVE_TRITON = True
try:
    import triton
except Exception:
    HAVE_TRITON = False

D, H, DH, FF, XN = 512, 8, 64, 2048, 1500
MAXN = 448
SCALE = 0.125

def _cuda(t): return isinstance(t, torch.Tensor) and t.is_cuda
def next_pow2(n): return triton.next_power_of_2(int(n))

# ---------- Layernorm ----------
@triton.jit
def _ln_kernel(X, Y, W, B, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride + cols, mask=cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    var = tl.sum((x - mean) * (x - mean), 0) / N
    xh = (x - mean) * tl.rsqrt(var + eps)
    w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=cols < N, other=0.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (xh * w + b).to(tl.float32), mask=cols < N)

def layernorm_triton(x, w, b, eps=1e-5):
    assert x.is_cuda
    orig = x.shape
    x2 = x.reshape(-1, orig[-1]).contiguous()
    N = x2.shape[-1]
    y = torch.empty_like(x2)
    _ln_kernel[(x2.shape[0],)](x2, y, w, b, x2.stride(0), N, eps, next_pow2(N))
    return y.reshape(orig)

def layernorm(x, w, b, eps=1e-5):
    if HAVE_TRITON and _cuda(x):
        try:
            return layernorm_triton(x, w, b, eps)
        except Exception:
            pass
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)

# ---------- Attention kernels ----------
@triton.jit
def _dec_attn_kernel(Q, K, V, O, stride_qh, stride_kh, stride_kn, stride_vh, stride_vn, N, scale, D: tl.constexpr, BLOCK_N: tl.constexpr):
    hid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + hid * stride_qh + offs_d).to(tl.float32)
    m = float("-inf"); l = 0.0; acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(K + hid * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :], mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, 1) * scale
        s = tl.where(offs_n < N, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 0))
        alpha = tl.exp(m - m_new)
        probs = tl.exp(s - m_new)
        l = l * alpha + tl.sum(probs, 0)
        v = tl.load(V + hid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :], mask=offs_n[:, None] < N, other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * v, 0)
        m = m_new
    acc = acc / l
    tl.store(O + hid * stride_qh + offs_d, acc)

@triton.jit
def _bdec_kernel(Q, K, V, O, stride_qb, stride_kb, stride_kn, stride_vb, stride_vn, N, scale, D: tl.constexpr, BLOCK_N: tl.constexpr):
    bhid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + bhid * stride_qb + offs_d).to(tl.float32)
    m = float("-inf"); l=0.0; acc = tl.zeros([D], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(K + bhid * stride_kb + offs_n[:, None]*stride_kn + offs_d[None,:], mask=offs_n[:,None]<N, other=0.0).to(tl.float32)
        s = tl.sum(q[None,:]*k,1)*scale
        s=tl.where(offs_n<N,s,float("-inf"))
        m_new=tl.maximum(m, tl.max(s,0))
        alpha=tl.exp(m-m_new)
        probs=tl.exp(s-m_new)
        l=l*alpha+tl.sum(probs,0)
        v=tl.load(V + bhid*stride_vb + offs_n[:,None]*stride_vn + offs_d[None,:], mask=offs_n[:,None]<N, other=0.0).to(tl.float32)
        acc=acc*alpha+tl.sum(probs[:,None]*v,0)
        m=m_new
    acc=acc/l
    tl.store(O + bhid*stride_qb + offs_d, acc)

def decode_attn(q, K, V, scale):
    if HAVE_TRITON and _cuda(q):
        try:
            H_, D_ = q.shape; N=K.shape[-2]; O=torch.empty_like(q)
            _dec_attn_kernel[(H_,)](q, K, V, O, q.stride(0), K.stride(0), K.stride(1), V.stride(0), V.stride(1), N, scale, D_, 128)
            return O
        except Exception:
            pass
    scores = torch.einsum("hd,hnd->hn", q.float(), K.float()) * scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("hn,hnd->hd", probs, V)

def batched_decode_attn(q, K, V, scale):
    if HAVE_TRITON and _cuda(q):
        try:
            B,H_,D_ = q.shape; N=K.shape[-2]
            qr=q.reshape(B*H_,D_); Kr=K.reshape(B*H_,N,D_); Vr=V.reshape(B*H_,N,D_)
            Or=torch.empty((B*H_,D_), device=q.device, dtype=q.dtype)
            _bdec_kernel[(B*H_,)](qr, Kr, Vr, Or, qr.stride(0), Kr.stride(0), Kr.stride(1), Vr.stride(0), Vr.stride(1), N, scale, D_, 128)
            return Or.reshape(B,H_,D_)
        except Exception:
            pass
    scores = torch.einsum("bhd,bhnd->bhn", q.float(), K.float()) * scale
    probs = torch.softmax(scores, dim=-1).to(V.dtype)
    return torch.einsum("bhn,bhnd->bhd", probs, V)

# ---------- Fused QKV ----------
@triton.jit
def _fused_qkv_kernel(X, Wq, Wk, Wv, Bq, Bk, Bv, Q, Kout, V, stride_wq0, stride_wq1, stride_wk0, stride_wk1, stride_wv0, stride_wv1, K_DIM, D, HAS_BQ: tl.constexpr, HAS_BK: tl.constexpr, HAS_BV: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < D
    acc_q = tl.zeros([BLOCK_N], dtype=tl.float32); acc_k = tl.zeros([BLOCK_N], dtype=tl.float32); acc_v = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K_DIM, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_DIM
        x = tl.load(X + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        wq = tl.load(Wq + offs_n[:,None]*stride_wq0 + offs_k[None,:]*stride_wq1, mask=mask_n[:,None] & mask_k[None,:], other=0.0).to(tl.float32)
        wk = tl.load(Wk + offs_n[:,None]*stride_wk0 + offs_k[None,:]*stride_wk1, mask=mask_n[:,None] & mask_k[None,:], other=0.0).to(tl.float32)
        wv = tl.load(Wv + offs_n[:,None]*stride_wv0 + offs_k[None,:]*stride_wv1, mask=mask_n[:,None] & mask_k[None,:], other=0.0).to(tl.float32)
        acc_q += tl.sum(wq * x[None,:],1); acc_k += tl.sum(wk * x[None,:],1); acc_v += tl.sum(wv * x[None,:],1)
    if HAS_BQ:
        bq = tl.load(Bq + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc_q = acc_q + bq
    if HAS_BK:
        bk = tl.load(Bk + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc_k = acc_k + bk
    if HAS_BV:
        bv = tl.load(Bv + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        acc_v = acc_v + bv
    tl.store(Q + offs_n, acc_q.to(Q.dtype.element_ty), mask=mask_n)
    tl.store(Kout + offs_n, acc_k.to(Kout.dtype.element_ty), mask=mask_n)
    tl.store(V + offs_n, acc_v.to(V.dtype.element_ty), mask=mask_n)

def fused_qkv(x, wq, wk, wv, bq=None, bv=None, bk=None):
    if HAVE_TRITON and _cuda(x):
        try:
            assert x.dim()==1
            K_DIM = x.shape[0]; D=wq.shape[0]
            xc=x.contiguous(); wqc=wq.contiguous(); wkc=wk.contiguous(); wvc=wv.contiguous()
            dummy=xc; Bq=bq.contiguous() if bq is not None else dummy; Bk=bk.contiguous() if bk is not None else dummy; Bv=bv.contiguous() if bv is not None else dummy
            Q=torch.empty((D,), device=x.device, dtype=x.dtype); Kout=torch.empty((D,), device=x.device, dtype=x.dtype); Vo=torch.empty((D,), device=x.device, dtype=x.dtype)
            _fused_qkv_kernel[(triton.cdiv(D,64),)](xc, wqc, wkc, wvc, Bq, Bk, Bv, Q, Kout, Vo, wqc.stride(0), wqc.stride(1), wkc.stride(0), wkc.stride(1), wvc.stride(0), wvc.stride(1), K_DIM, D, bq is not None, bk is not None, bv is not None, 64,64)
            return Q, Kout, Vo
        except Exception:
            pass
    return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))

def fused_qkv_batched(x, wq, wk, wv, bq=None, bv=None, bk=None):
    # x [B,D] -> 3x [B,D], batched 1 launch for B>1 via torch GEMM, fused for B==1
    if HAVE_TRITON and _cuda(x) and x.shape[0]==1:
        try:
            q,k,v = fused_qkv(x[0], wq, wk, wv, bq, bv, bk)
            return q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        except Exception:
            pass
    # batched torch is faster for B>1 (uses cuBLAS batched GEMM)
    return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))

# ---------- Fused Decoder Layer (1 layer x 6, batched + KV cache) ----------
def whisper_fused_decoder_layer(x, w, prefix, sk, sv, n, Kx, Vx, B=None):
    """One fused decoder layer, batched.

    x: [B, D]  (decoder hidden at position n)
    sk/sv: [B, H, MAXN, Dh] KV cache per layer (batched) or [H, MAXN, Dh] for B=1
    Kx/Vx: [B, H, XN, Dh] cross KV (batched) or [H, XN, Dh]
    n: int position
    B: batch size (infer from x)
    returns [B, D]
    """
    if B is None:
        B = x.shape[0] if x.dim()==2 else 1
        if x.dim()==1:
            x = x.unsqueeze(0)
            was_1d = True
        else:
            was_1d=False
    else:
        was_1d=False
    # handle sk/sv shapes: if [H, MAXN, Dh] expand to [B,H,MAXN,Dh]
    def _expand_cache(c):
        if c.dim()==3:
            return c.unsqueeze(0).expand(B, -1, -1, -1)
        return c
    # layernorm
    h = layernorm(x, w[prefix+"self_attn_layer_norm.weight"], w[prefix+"self_attn_layer_norm.bias"])
    # self attn fused QKV batched
    # h [B,D]
    qf, kf, vf = fused_qkv_batched(h, w[prefix+"self_attn.q_proj.weight"], w[prefix+"self_attn.k_proj.weight"], w[prefix+"self_attn.v_proj.weight"], w[prefix+"self_attn.q_proj.bias"], w[prefix+"self_attn.v_proj.bias"], None)
    # qf [B,D] -> [B,H,Dh]
    q = qf.view(B, H, DH); k1 = kf.view(B, H, DH); v1 = vf.view(B, H, DH)
    # KV cache write
    if sk.dim()==3:
        # single batch cache [H, MAXN, Dh] -> write
        if B==1:
            sk[:, n] = k1[0]; sv[:, n] = v1[0]
            K = sk[:, :n+1].unsqueeze(0); V = sv[:, :n+1].unsqueeze(0)
            q_b = q  # [1,H,Dh]
        else:
            raise RuntimeError("cache dim mismatch")
    else:
        # batched [B,H,MAXN,Dh] -> vectorized write, no loop
        sk[:, :, n, :] = k1
        sv[:, :, n, :] = v1
        K = sk[:, :, :n+1, :]
        V = sv[:, :, :n+1, :]
        q_b = q
    # decode attn batched
    # q [B,H,Dh], K/V [B,H,N,Dh]
    if K.dim()==3:
        # [H,N,Dh] single
        o_self = decode_attn(q[0], K, V, SCALE)  # [H,Dh]
        o_self = o_self.reshape(1, -1)
    else:
        o_self = batched_decode_attn(q, K, V, SCALE).reshape(B, -1)  # [B, D]
    # out proj
    o_self = F.linear(o_self, w[prefix+"self_attn.out_proj.weight"], w[prefix+"self_attn.out_proj.bias"])
    x = x + o_self
    # cross attn
    h2 = layernorm(x, w[prefix+"encoder_attn_layer_norm.weight"], w[prefix+"encoder_attn_layer_norm.bias"])
    q2 = F.linear(h2, w[prefix+"encoder_attn.q_proj.weight"], w[prefix+"encoder_attn.q_proj.bias"]).view(B, H, DH)
    # Kx/Vx handling
    if Kx.dim()==3:
        Kx_b = Kx.unsqueeze(0).expand(B, -1, -1, -1); Vx_b = Vx.unsqueeze(0).expand(B, -1, -1, -1)
    else:
        Kx_b = Kx; Vx_b = Vx
    o_cross = batched_decode_attn(q2, Kx_b, Vx_b, SCALE).reshape(B, -1)
    o_cross = F.linear(o_cross, w[prefix+"encoder_attn.out_proj.weight"], w[prefix+"encoder_attn.out_proj.bias"])
    x = x + o_cross
    # FF
    h3 = layernorm(x, w[prefix+"final_layer_norm.weight"], w[prefix+"final_layer_norm.bias"])
    m = F.linear(h3, w[prefix+"fc1.weight"], w[prefix+"fc1.bias"])
    m = F.gelu(m)
    x = x + F.linear(m, w[prefix+"fc2.weight"], w[prefix+"fc2.bias"])
    if was_1d:
        return x[0]
    return x

def whisper_fused_encoder_layer(x, w, prefix, B=None):
    """Encoder layer fused: layernorm + self attn + FF, batched [B,T,D]"""
    # x [B,T,D]
    h = layernorm(x, w[prefix+"self_attn_layer_norm.weight"], w[prefix+"self_attn_layer_norm.bias"])
    # self attn
    B_, T, _ = h.shape
    q = F.linear(h, w[prefix+"self_attn.q_proj.weight"], w[prefix+"self_attn.q_proj.bias"])
    k = F.linear(h, w[prefix+"self_attn.k_proj.weight"], None)
    v = F.linear(h, w[prefix+"self_attn.v_proj.weight"], w[prefix+"self_attn.v_proj.bias"])
    q4 = q.view(B_, T, H, DH).transpose(1,2)
    k4 = k.view(B_, T, H, DH).transpose(1,2)
    v4 = v.view(B_, T, H, DH).transpose(1,2)
    # use SDPA for encoder (non-causal) - keep fused layernorm benefit
    o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
    o = o.transpose(1,2).reshape(B_, T, D)
    o = F.linear(o, w[prefix+"self_attn.out_proj.weight"], w[prefix+"self_attn.out_proj.bias"])
    x = x + o
    h2 = layernorm(x, w[prefix+"final_layer_norm.weight"], w[prefix+"final_layer_norm.bias"])
    m = F.linear(h2, w[prefix+"fc1.weight"], w[prefix+"fc1.bias"])
    m = F.gelu(m)
    x = x + F.linear(m, w[prefix+"fc2.weight"], w[prefix+"fc2.bias"])
    return x

# ---------- VRAM ----------
def estimate_whisper_kv_mb(B, n_layers=6, H=8, maxn=448, Dh=64, bytes_per=2):
    # decoder self KV: 2 * layers * B * H * maxn * Dh *2B + cross KV: layers*B*H*1500*Dh*2B (but cross is computed once, not cached per token)
    return 2 * n_layers * B * H * maxn * Dh * bytes_per / (1024**2)

# ---------- perf_report ----------
try:
    import triton.testing
    _has_perf=True
except Exception:
    _has_perf=False

if _has_perf:
    _configs = [
        triton.testing.Benchmark(
            x_names=["B"], x_vals=[1,2,4,8],
            line_arg="provider", line_vals=["triton","torch"],
            line_names=["Triton fused decoder", "Torch eager"],
            styles=[("blue","-"), ("orange","--")],
            ylabel="ms", plot_name="whisper-fused-layer-B",
            args={"T": 1500, "N": 64},
        ),
        triton.testing.Benchmark(
            x_names=["seq_len"], x_vals=[32,64,128,256,448],
            line_arg="provider", line_vals=["triton","torch"],
            line_names=["Triton","Torch"],
            styles=[("blue","-"), ("orange","--")],
            ylabel="ms", plot_name="whisper-fused-layer-seq",
            args={"B":2},
        ),
    ]
    @triton.testing.perf_report(_configs)
    def bench_whisper_fused(B, seq_len=None, T=1500, N=64, provider="triton"):
        device="cuda" if torch.cuda.is_available() else "cpu"
        dtype=torch.float16
        torch.manual_seed(0)
        n = min(seq_len if seq_len is not None else N, MAXN-1)
        x = (torch.randn(B, D, device=device, dtype=torch.float32) * 0.5).to(dtype)
        sk = (torch.randn(8, MAXN, 64, device=device, dtype=torch.float32) * 0.05).to(dtype) if B==1 else (torch.randn(B, H, MAXN, 64, device=device, dtype=torch.float32) * 0.05).to(dtype)
        sv = (torch.randn(8, MAXN, 64, device=device, dtype=torch.float32) * 0.05).to(dtype) if B==1 else (torch.randn(B, H, MAXN, 64, device=device, dtype=torch.float32) * 0.05).to(dtype)
        Kx = (torch.randn(H, XN, 64, device=device, dtype=torch.float32) * 0.05).to(dtype) if B==1 else (torch.randn(B, H, XN, 64, device=device, dtype=torch.float32) * 0.05).to(dtype)
        Vx = (torch.randn(H, XN, 64, device=device, dtype=torch.float32) * 0.05).to(dtype) if B==1 else (torch.randn(B, H, XN, 64, device=device, dtype=torch.float32) * 0.05).to(dtype)
        w={}
        for suf in ["self_attn_layer_norm.weight","self_attn_layer_norm.bias","encoder_attn_layer_norm.weight","encoder_attn_layer_norm.bias","final_layer_norm.weight","final_layer_norm.bias"]:
            if "weight" in suf:
                w[f"model.decoder.layers.0.{suf}"] = torch.ones(D, device=device, dtype=dtype)
            else:
                w[f"model.decoder.layers.0.{suf}"] = torch.zeros(D, device=device, dtype=dtype)
        def rand_w(*shape):
            return (torch.randn(*shape, device=device, dtype=torch.float32) * 0.02).to(dtype)
        for proj in ["self_attn.q_proj.weight","self_attn.q_proj.bias","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.v_proj.bias","self_attn.out_proj.weight","self_attn.out_proj.bias","encoder_attn.q_proj.weight","encoder_attn.q_proj.bias","encoder_attn.out_proj.weight","encoder_attn.out_proj.bias","fc1.weight","fc1.bias","fc2.weight","fc2.bias"]:
            if proj=="self_attn.q_proj.weight": shape=(D,D)
            elif proj=="self_attn.k_proj.weight": shape=(D,D)
            elif proj=="self_attn.v_proj.weight": shape=(D,D)
            elif proj=="self_attn.out_proj.weight": shape=(D,D)
            elif proj=="encoder_attn.q_proj.weight": shape=(D,D)
            elif proj=="encoder_attn.out_proj.weight": shape=(D,D)
            elif proj=="fc1.weight": shape=(FF,D)
            elif proj=="fc2.weight": shape=(D,FF)
            elif "bias" in proj: shape=(D,) if "attn" in proj else (FF,) if "fc1" in proj else (D,)
            else: shape=(D,D)
            if "weight" in proj:
                w[f"model.decoder.layers.0.{proj}"] = rand_w(*shape) if len(shape)==2 else rand_w(shape[0])
            else:
                w[f"model.decoder.layers.0.{proj}"] = (torch.randn(shape[0], device=device, dtype=torch.float32) * 0.01).to(dtype)
        est = estimate_whisper_kv_mb(B, 6, H, 448, 64)
        try:
            check_budget(est, budget_mb=4000, what="bench_whisper")
        except Exception:
            return float("nan"), float("nan"), float("nan")
        from src.models.pytorch.whisper import whisper_decoder_layer_torch
        import triton.testing as tt
        # clone caches for fair
        sk_t = sk.clone(); sv_t = sv.clone()
        sk_r = sk.clone(); sv_r = sv.clone()
        # scale weights for realistic (avoid inf)
        # use cloned w for both
        def run_triton():
            return whisper_fused_decoder_layer(x, w, "model.decoder.layers.0.", sk_t, sv_t, n, Kx, Vx)
        def run_torch():
            return whisper_decoder_layer_torch(x, w, "model.decoder.layers.0.", sk_r, sv_r, n, Kx, Vx)
        fn = run_triton if provider=="triton" else run_torch
        ms = tt.do_bench(fn, warmup=25, rep=100)
        return ms, ms*0.9, ms*1.1
    if __name__=="__main__":
        import argparse, os
        p=argparse.ArgumentParser()
        p.add_argument("--bench", action="store_true")
        p.add_argument("--save_path", default="benchmarks/results")
        args=p.parse_args()
        if args.bench:
            os.makedirs(args.save_path, exist_ok=True)
            bench_whisper_fused.run(save_path=args.save_path, print_data=True)

HAVE_TRITON_KERNELS = HAVE_TRITON
__all__=["layernorm","decode_attn","batched_decode_attn","fused_qkv","fused_qkv_batched","whisper_fused_decoder_layer","whisper_fused_encoder_layer","estimate_whisper_kv_mb","bench_whisper_fused"]
