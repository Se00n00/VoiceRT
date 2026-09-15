"""LLM kernel-level: Triton vs PyTorch parity + latency."""
import argparse
import json
from pathlib import Path

import torch

from benchmarks.common.benchmark import seed_everything, dtype_from_str, ensure_output_dir, save_json, save_csv
from benchmarks.common.environments import get_environment
from benchmarks.common.metrics import latency_stats, correctness_metrics, check_tolerance
from benchmarks.common.timing import measure_latencies
from benchmarks.common.memory import reset_peak, snapshot


# Kernel registry: only kernels that actually exist
KERNELS = {}


def _register(name, triton_fn, torch_fn, make_inputs):
    KERNELS[name] = (triton_fn, torch_fn, make_inputs)


def _init_registry(device, dtype):
    from src.models.triton_kernels.rmsnorm import rmsnorm as triton_rms
    from src.models.triton_kernels.rope import rope as triton_rope, rope_batched as triton_rope_batched
    from src.models.triton_kernels.activation import swiglu as triton_swiglu
    from src.models.triton_kernels.attention import gqa_decode_attn as triton_gqa, fused_qkv_gqa as triton_fused_gqa

    def _rms_torch(x, w):
        # exact torch from qwen.py fallback
        xf = x.float()
        var = (xf * xf).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + 1e-6) * w.float()).to(x.dtype).reshape(x.shape)

    def _rope_torch(x, cos, sin, pos):
        shp = x.shape
        dh = shp[-1]
        half = dh // 2
        xf = x.reshape(-1, dh).float()
        c = cos[pos].to(torch.float32)
        s = sin[pos].to(torch.float32)
        x1, x2 = xf[:, :half], xf[:, half:]
        y = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        return y.to(x.dtype).reshape(shp)

    def _swiglu_torch(g, u):
        import torch.nn.functional as F
        return F.silu(g.float()).to(g.dtype) * u

    def _gqa_torch(q, K, V, scale):
        Hq, D = q.shape
        Hk = K.shape[0]
        group = Hq // Hk
        Ke = K.repeat_interleave(group, dim=0).float()
        Ve = V.repeat_interleave(group, dim=0).float()
        scores = torch.einsum("hd,hnd->hn", q.float(), Ke) * scale
        probs = torch.softmax(scores, dim=-1).float()
        out = torch.einsum("hn,hnd->hd", probs, Ve)
        return out.to(V.dtype)

    def _fused_gqa_torch(x, wq, wk, wv, bq, bk, bv):
        import torch.nn.functional as F
        return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))

    # small synthetic shapes matching Qwen3-0.6B dims
    def make_rms():
        # x [4,1024] w [1024]
        x = torch.randn(4, 1024, device=device, dtype=dtype)
        w = torch.randn(1024, device=device, dtype=dtype)
        return (x, w), {}

    def make_rope():
        dh = 128
        x = torch.randn(4, dh, device=device, dtype=dtype)
        cos = torch.randn(512, dh // 2, device=device, dtype=dtype)
        sin = torch.randn(512, dh // 2, device=device, dtype=dtype)
        return (x, cos, sin, 7), {}

    def make_rope_batched():
        T, H, dh = 32, 8, 128
        x = torch.randn(T, H, dh, device=device, dtype=dtype)
        cos = torch.randn(512, dh // 2, device=device, dtype=dtype)
        sin = torch.randn(512, dh // 2, device=device, dtype=dtype)
        return (x, cos, sin, 0), {}

    def make_swiglu():
        g = torch.randn(4, 1024, device=device, dtype=dtype)
        u = torch.randn(4, 1024, device=device, dtype=dtype)
        return (g, u), {}

    def make_gqa():
        Hq, Hk, N, D = 16, 8, 128, 128
        scale = 1.0 / (D ** 0.5)
        q = torch.randn(Hq, D, device=device, dtype=dtype)
        K = torch.randn(Hk, N, D, device=device, dtype=dtype)
        V = torch.randn(Hk, N, D, device=device, dtype=dtype)
        return (q, K, V, scale), {}

    def make_fused():
        Kdim, Dq, Dkv = 1024, 1024, 512
        x = torch.randn(Kdim, device=device, dtype=dtype)
        wq = torch.randn(Dq, Kdim, device=device, dtype=dtype)
        wk = torch.randn(Dkv, Kdim, device=device, dtype=dtype)
        wv = torch.randn(Dkv, Kdim, device=device, dtype=dtype)
        bq = torch.randn(Dq, device=device, dtype=dtype)
        bk = torch.randn(Dkv, device=device, dtype=dtype)
        bv = torch.randn(Dkv, device=device, dtype=dtype)
        return (x, wq, wk, wv, bq, bk, bv), {}

    KERNELS.clear()
    _register("rmsnorm", triton_rms, _rms_torch, make_rms)
    _register("rope", triton_rope, _rope_torch, make_rope)
    _register("rope_batched", triton_rope_batched, lambda x,c,s,p0=0: torch.cat([_rope_torch(x[t:t+1],c,s,p0+t) for t in range(x.shape[0])],0), make_rope_batched)
    _register("swiglu", triton_swiglu, _swiglu_torch, make_swiglu)
    _register("gqa_decode_attn", triton_gqa, _gqa_torch, make_gqa)
    _register("fused_qkv_gqa", triton_fused_gqa, _fused_gqa_torch, make_fused)


def run_one(name, warmup, iterations, atol, rtol):
    triton_fn, torch_fn, make_inputs = KERNELS[name]
    (args, kwargs) = make_inputs()
    # ensure inputs are same for both - clone
    import copy
    # numerical validation
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    with torch.no_grad():
        try:
            out_torch = torch_fn(*args, **kwargs)
            out_triton = triton_fn(*args, **kwargs)
        except Exception as e:
            return {"kernel": name, "error": str(e), "correctness": {}}
    # handle tuple outputs
    if isinstance(out_torch, tuple):
        metrics_list = []
        for a,b in zip(out_torch, out_triton):
            metrics_list.append(correctness_metrics(a, b, atol=atol, rtol=rtol))
        # aggregate max
        agg = {"max_abs_error": max(m["max_abs_error"] for m in metrics_list),
               "mean_abs_error": sum(m["mean_abs_error"] for m in metrics_list)/len(metrics_list),
               "relative_error": max(m["relative_error"] for m in metrics_list),
               "cosine_similarity": min(m["cosine_similarity"] for m in metrics_list)}
        correctness = agg
        ok, msg = check_tolerance(correctness, atol=atol, rtol=rtol)
    else:
        correctness = correctness_metrics(out_torch, out_triton, atol=atol, rtol=rtol)
        ok, msg = check_tolerance(correctness, atol=atol, rtol=rtol)

    # timing triton
    def fn_triton():
        with torch.no_grad():
            triton_fn(*args, **kwargs)
    def fn_torch():
        with torch.no_grad():
            torch_fn(*args, **kwargs)

    reset_peak()
    lat_triton = measure_latencies(fn_triton, warmup=warmup, iterations=iterations)
    peak_triton = snapshot()["peak_mb"]
    lat_torch = measure_latencies(fn_torch, warmup=warmup, iterations=iterations)
    peak_torch = snapshot()["peak_mb"]

    stats_t = latency_stats(lat_triton)
    stats_p = latency_stats(lat_torch)
    speedup = (stats_p["median_ms"] / stats_t["median_ms"]) if stats_t["median_ms"] > 0 else 0

    return {
        "kernel": name,
        "correctness": correctness,
        "correct": ok,
        "correct_msg": msg,
        "triton": {**stats_t, "peak_mb": peak_triton},
        "pytorch": {**stats_p, "peak_mb": peak_torch},
        "speedup": speedup,
        "warmup": warmup,
        "iterations": iterations,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM kernel benchmarks Triton vs PyTorch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp32","fp16","bf16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--kernels", nargs="*", default=None, help="subset of kernels")
    args = parser.parse_args()

    from benchmarks.common.benchmark import seed_everything, dtype_from_str
    seed_everything(args.seed)
    dtype = dtype_from_str(args.dtype)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, falling back to CPU (timings will be CPU)")

    _init_registry(device, dtype)
    env = get_environment(model_name="Qwen/Qwen2.5-0.5B-Instruct", dtype=args.dtype)

    selected = args.kernels if args.kernels else list(KERNELS.keys())
    results = []
    for name in selected:
        if name not in KERNELS:
            print(f"unknown kernel {name}, skipping")
            continue
        print(f"Benchmarking kernel {name} ...")
        r = run_one(name, args.warmup, args.iterations, args.atol, args.rtol)
        r.update(env)
        r["device"] = device
        r["dtype"] = args.dtype
        results.append(r)
        print(f"  {name}: triton {r.get('triton',{}).get('median_ms',0):.3f}ms vs torch {r.get('pytorch',{}).get('median_ms',0):.3f}ms speedup {r.get('speedup',0):.2f}x correct={r.get('correct')}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # save json
    save_json(results, out_dir / "llm_kernels_correctness.json")
    # csv for latency comparison
    rows = []
    for r in results:
        for backend in ("pytorch","triton"):
            if backend in r:
                s = r[backend]
                rows.append({
                    "kernel": r["kernel"],
                    "backend": backend,
                    "median_ms": s.get("median_ms"),
                    "p50_ms": s.get("p50_ms"),
                    "p95_ms": s.get("p95_ms"),
                    "p99_ms": s.get("p99_ms"),
                    "mean_ms": s.get("mean_ms"),
                    "min_ms": s.get("min_ms"),
                    "max_ms": s.get("max_ms"),
                    "peak_mb": s.get("peak_mb"),
                    "speedup": r.get("speedup") if backend=="triton" else 1.0,
                    "gpu_name": r.get("gpu_name"),
                    "correct": r.get("correct"),
                    "max_abs_error": r.get("correctness",{}).get("max_abs_error"),
                })
    save_csv(rows, out_dir / "llm_kernels_latency.csv", fieldnames=["kernel","backend","median_ms","p50_ms","p95_ms","p99_ms","mean_ms","min_ms","max_ms","peak_mb","speedup","gpu_name","correct","max_abs_error"])
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
