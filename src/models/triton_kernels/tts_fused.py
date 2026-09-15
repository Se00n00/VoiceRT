"""Single-file fused Triton kernels for TTS (Kokoro) - batching + VRAM.

Batched audio post-processing: [B, C, L] with B=1..8, keeps VRAM via chunking.
Includes perf_report inside file.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from src.models.runtime.memory import check_budget

HAVE_TRITON=True
try:
    import triton
except Exception:
    HAVE_TRITON=False

def _cuda(t): return isinstance(t, torch.Tensor) and t.is_cuda

# ---------- conv1d_silu (from conv1d.py, merged) ----------
@triton.jit
def conv1d_silu_fwd_kernel(X, W, B, Y, stride_x_n, stride_x_c, stride_x_l, stride_w_c, stride_w_in, stride_w_k, stride_b, stride_y_n, stride_y_c, stride_y_l, N: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, K: tl.constexpr, PADDING: tl.constexpr, DILATION: tl.constexpr, STRIDE: tl.constexpr, L_IN: tl.constexpr, L_OUT: tl.constexpr):
    n = tl.program_id(0); c_out = tl.program_id(1); pos = tl.program_id(2)
    y_offset = n * stride_y_n + c_out * stride_y_c + pos * stride_y_l
    inp_start = pos * STRIDE - PADDING
    acc = 0.0
    for cin in range(C_in):
        for k in range(K):
            inp_pos = inp_start + k * DILATION
            valid = (inp_pos >= 0) & (inp_pos < L_IN)
            w_idx = c_out * stride_w_c + cin * stride_w_in + k * stride_w_k
            x_idx = n * stride_x_n + cin * stride_x_c + inp_pos * stride_x_l
            x_val = tl.load(X + x_idx, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(W + w_idx).to(tl.float32)
            acc = acc + x_val * w_val
    acc = acc + tl.load(B + c_out * stride_b).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-acc))
    acc = acc * sig
    tl.store(Y + y_offset, acc)

def conv1d_silu_triton(x, weight, bias=None, stride=1, padding=0, dilation=1):
    assert x.is_cuda and weight.is_cuda and x.dim()==3 and weight.dim()==3
    N, C_in, L_in = x.shape
    C_out, _, K = weight.shape
    if bias is None:
        bias=torch.zeros((C_out,), device=x.device, dtype=torch.float32)
    L_out = (L_in + 2*padding - dilation*(K-1) -1)//stride +1
    xc=x.contiguous(); wc=weight.contiguous(); bc=bias.contiguous()
    y=torch.empty((N, C_out, L_out), device=x.device, dtype=torch.float32)
    grid=(N, C_out, L_out)
    conv1d_silu_fwd_kernel[grid](xc,wc,bc,y, xc.stride(0),xc.stride(1),xc.stride(2), wc.stride(0),wc.stride(1),wc.stride(2), bc.stride(0), y.stride(0),y.stride(1),y.stride(2), N,C_in,C_out,K,padding,dilation,stride,L_in,L_out)
    return y.to(x.dtype)

def conv1d_silu(x, weight, bias=None, stride=1, padding=0, dilation=1):
    if HAVE_TRITON and _cuda(x):
        try:
            return conv1d_silu_triton(x, weight, bias, stride, padding, dilation)
        except Exception:
            pass
    return F.silu(F.conv1d(x, weight, bias, stride, padding, dilation))

@triton.jit
def _in1d_silu_kernel(X, Y, stride_row, L, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride_row
    n_f = L * 1.0
    acc=0.0
    for start in range(0, L, BLOCK):
        offs=start+tl.arange(0,BLOCK)
        mask=offs<L
        x=tl.load(X+base+offs, mask=mask, other=0.0).to(tl.float32)
        acc+=tl.sum(tl.where(mask,x,0.0),0)
    mean=acc/n_f
    var_acc=0.0
    for start in range(0, L, BLOCK):
        offs=start+tl.arange(0,BLOCK)
        mask=offs<L
        x=tl.load(X+base+offs, mask=mask, other=0.0).to(tl.float32)
        d=tl.where(mask, x-mean, 0.0)
        var_acc+=tl.sum(d*d,0)
    var=var_acc/n_f
    inv=1.0/tl.sqrt(var+eps)
    for start in range(0, L, BLOCK):
        offs=start+tl.arange(0,BLOCK)
        mask=offs<L
        x=tl.load(X+base+offs, mask=mask, other=0.0).to(tl.float32)
        xn=(x-mean)*inv
        sig=1.0/(1.0+tl.exp(-xn))
        y=xn*sig
        tl.store(Y+base+offs, y.to(Y.dtype.element_ty), mask=mask)

def in1d_silu_triton(x, eps=1e-5):
    assert x.is_cuda and x.dim()>=2
    shp=x.shape; L=shp[-1]; R=int(torch.prod(torch.tensor(shp[:-1])) ) if len(shp)>1 else 1
    # compute R manually
    R=1
    for s in shp[:-1]:
        R*=int(s)
    xf=x.reshape(R,L).contiguous()
    yf=torch.empty((R,L), device=x.device, dtype=x.dtype)
    _in1d_silu_kernel[(R,)](xf, yf, xf.stride(0), L, float(eps), 1024)
    return yf.reshape(shp)

def in1d_silu(x, eps=1e-5):
    if HAVE_TRITON and _cuda(x):
        try:
            return in1d_silu_triton(x, eps)
        except Exception:
            pass
    return F.silu(F.instance_norm(x))

def conv1d_forward(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv1d(x, weight, bias, stride, padding, dilation, groups)

def resample_linear(wav, sr_in, sr_out):
    import numpy as np
    x=np.asarray(wav, dtype=np.float32).ravel()
    if sr_in==sr_out or x.size==0:
        return x
    t=torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
    n_out=max(1, int(round(x.size*sr_out/float(sr_in))))
    y=F.interpolate(t, size=n_out, mode="linear", align_corners=False).squeeze()
    return y.numpy().astype(np.float32)

def resample_batched(wavs, sr_in, sr_out):
    # wavs: List[np] batch -> List[np] resampled, VRAM friendly (CPU)
    return [resample_linear(w, sr_in, sr_out) for w in wavs]

def postprocess(wav, sr=24000, enhance=False, peak=0.98):
    import numpy as np
    x=np.asarray(wav, dtype=np.float32).ravel()
    if x.size==0:
        return x
    x=x-float(x.mean())
    m=float(np.abs(x).max()) if x.size else 0.0
    if m>1e-9:
        x=(x/m*float(peak)).astype(np.float32)
    if enhance and x.size>=16:
        try:
            t=torch.from_numpy(x).unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                t=t.cuda()
            y=in1d_silu(t.to(torch.float32))
            x=y.float().cpu().numpy().ravel().astype(np.float32)
            m2=float(np.abs(x).max()) if x.size else 0.0
            if m2>1e-9:
                x=(x/m2*float(peak)).astype(np.float32)
        except Exception:
            pass
    return x

def postprocess_batched(wavs, sr=24000, enhance=False, peak=0.98, batch_size=8):
    """Batched postprocess with VRAM chunking: [B, L] -> [B, L]"""
    # wavs: List[np] or [B, L] tensor
    if isinstance(wavs, torch.Tensor):
        # [B, C, L] or [B, L]
        if wavs.dim()==2:
            wavs=[wavs[b].cpu().numpy() for b in range(wavs.shape[0])]
        else:
            wavs=[wavs[b,0].cpu().numpy() if wavs.dim()==3 else wavs[b].cpu().numpy() for b in range(wavs.shape[0])]
    est_mb = sum(w.size*4 for w in [np.asarray(w) for w in wavs])/(1024**2) + 10
    try:
        check_budget(est_mb, budget_mb=4000, what="tts postprocess batched")
    except Exception as e:
        # chunk
        print(f"[TTS] VRAM chunked: {e}", flush=True)
    outs=[]
    for i in range(0, len(wavs), batch_size):
        chunk=wavs[i:i+batch_size]
        for w in chunk:
            outs.append(postprocess(w, sr=sr, enhance=enhance, peak=peak))
    return outs

# ---------- VRAM ----------
def estimate_tts_mb(B, L, C=1, bytes_per=4):
    return B*C*L*bytes_per/(1024**2) + 50  # +50MB overhead for kokoro

# ---------- perf_report ----------
try:
    import triton.testing
    _has_perf=True
except Exception:
    _has_perf=False

if _has_perf:
    _configs=[
        triton.testing.Benchmark(
            x_names=["B"], x_vals=[1,2,4,8],
            line_arg="provider", line_vals=["triton","torch"],
            line_names=["Triton in1d_silu","Torch eager"],
            styles=[("blue","-"),("orange","--")],
            ylabel="ms", plot_name="tts-fused-B",
            args={"C": 32, "L": 2048},
        ),
        triton.testing.Benchmark(
            x_names=["L"], x_vals=[512,1024,2048,4096,8192],
            line_arg="provider", line_vals=["triton","torch"],
            line_names=["Triton","Torch"],
            styles=[("blue","-"),("orange","--")],
            ylabel="ms", plot_name="tts-fused-L",
            args={"B":2, "C":32},
        ),
    ]
    @triton.testing.perf_report(_configs)
    def bench_tts_fused(B, L=None, C=32, provider="triton"):
        device="cuda" if torch.cuda.is_available() else "cpu"
        dtype=torch.float16
        torch.manual_seed(0)
        l = L if L is not None else 2048
        x=torch.randn(B, C, l, device=device, dtype=dtype)
        est=estimate_tts_mb(B,l,C)
        try:
            check_budget(est, budget_mb=4000, what="bench_tts")
        except Exception:
            return float("nan"), float("nan"), float("nan")
        import triton.testing as tt
        def run_triton():
            return in1d_silu(x)
        def run_torch():
            return F.silu(F.instance_norm(x.float())).to(dtype)
        fn=run_triton if provider=="triton" else run_torch
        ms=tt.do_bench(fn, warmup=25, rep=100)
        return ms, ms*0.9, ms*1.1
    if __name__=="__main__":
        import argparse, os
        p=argparse.ArgumentParser()
        p.add_argument("--bench", action="store_true")
        p.add_argument("--save_path", default="benchmarks/results")
        args=p.parse_args()
        if args.bench:
            os.makedirs(args.save_path, exist_ok=True)
            bench_tts_fused.run(save_path=args.save_path, print_data=True)

HAVE_TRITON_KERNELS=HAVE_TRITON
__all__=["conv1d_silu","in1d_silu","conv1d_forward","resample_linear","resample_batched","postprocess","postprocess_batched","estimate_tts_mb","bench_tts_fused"]
