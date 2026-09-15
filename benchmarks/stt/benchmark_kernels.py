"""STT kernel-level: Triton vs PyTorch parity + latency."""
import argparse
import torch

from benchmarks.common.benchmark import seed_everything, dtype_from_str, ensure_output_dir, save_json, save_csv
from benchmarks.common.environments import get_environment
from benchmarks.common.metrics import latency_stats, correctness_metrics, check_tolerance
from benchmarks.common.timing import measure_latencies
from benchmarks.common.memory import reset_peak, snapshot

KERNELS = {}

def _init_registry(device, dtype):
    from src.models.triton_kernels.layernorm import layernorm as triton_ln
    from src.models.triton_kernels.softmax import row_softmax as triton_softmax
    from src.models.triton_kernels.attention import decode_attn as triton_dec, batched_decode_attn as triton_bdec, fused_qkv as triton_fused

    def ln_torch(x, w, b):
        import torch.nn.functional as F
        return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)

    def softmax_torch(x):
        import torch.nn.functional as F
        return F.softmax(x, dim=-1)

    def dec_torch(q, K, V, scale):
        scores = torch.einsum("hd,hnd->hn", q.float(), K.float()) * scale
        probs = torch.softmax(scores, dim=-1).to(V.dtype)
        return torch.einsum("hn,hnd->hd", probs, V)

    def bdec_torch(q, K, V, scale):
        scores = torch.einsum("bhd,bhnd->bhn", q.float(), K.float()) * scale
        probs = torch.softmax(scores, dim=-1).to(V.dtype)
        return torch.einsum("bhn,bhnd->bhd", probs, V)

    def fused_torch(x, wq, wk, wv, bq, bv, bk):
        import torch.nn.functional as F
        return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))

    def make_ln():
        x = torch.randn(32, 512, device=device, dtype=dtype)
        w = torch.randn(512, device=device, dtype=dtype)
        b = torch.randn(512, device=device, dtype=dtype)
        return (x, w, b), {}

    def make_softmax():
        x = torch.randn(8, 1500, device=device, dtype=dtype)
        return (x,), {}

    def make_dec():
        H, dh, N = 8, 64, 128
        q = torch.randn(H, dh, device=device, dtype=dtype)
        K = torch.randn(H, N, dh, device=device, dtype=dtype)
        V = torch.randn(H, N, dh, device=device, dtype=dtype)
        return (q, K, V, 0.125), {}

    def make_bdec():
        B, H, dh, N = 2, 8, 64, 64
        q = torch.randn(B, H, dh, device=device, dtype=dtype)
        K = torch.randn(B, H, N, dh, device=device, dtype=dtype)
        V = torch.randn(B, H, N, dh, device=device, dtype=dtype)
        return (q, K, V, 0.125), {}

    def make_fused():
        D = 512
        x = torch.randn(D, device=device, dtype=dtype)
        wq = torch.randn(D, D, device=device, dtype=dtype)
        wk = torch.randn(D, D, device=device, dtype=dtype)
        wv = torch.randn(D, D, device=device, dtype=dtype)
        bq = torch.randn(D, device=device, dtype=dtype)
        bv = torch.randn(D, device=device, dtype=dtype)
        return (x, wq, wk, wv, bq, bv, None), {}

    KERNELS.clear()
    KERNELS["layernorm"] = (triton_ln, ln_torch, make_ln)
    KERNELS["row_softmax"] = (triton_softmax, softmax_torch, make_softmax)
    KERNELS["decode_attn"] = (triton_dec, dec_torch, make_dec)
    KERNELS["batched_decode_attn"] = (triton_bdec, bdec_torch, make_bdec)
    KERNELS["fused_qkv"] = (triton_fused, fused_torch, make_fused)


def run_one(name, warmup, iterations, atol, rtol):
    triton_fn, torch_fn, make_inputs = KERNELS[name]
    (args, kwargs) = make_inputs()
    with torch.no_grad():
        try:
            out_torch = torch_fn(*args, **kwargs)
            out_triton = triton_fn(*args, **kwargs)
        except Exception as e:
            return {"kernel": name, "error": str(e)}
    if isinstance(out_torch, tuple):
        metrics = []
        for a,b in zip(out_torch, out_triton):
            metrics.append(correctness_metrics(a,b,atol=atol,rtol=rtol))
        correctness = {"max_abs_error": max(m["max_abs_error"] for m in metrics),
                       "mean_abs_error": sum(m["mean_abs_error"] for m in metrics)/len(metrics),
                       "relative_error": max(m["relative_error"] for m in metrics),
                       "cosine_similarity": min(m["cosine_similarity"] for m in metrics)}
    else:
        correctness = correctness_metrics(out_torch, out_triton, atol=atol, rtol=rtol)
    ok, msg = check_tolerance(correctness, atol=atol, rtol=rtol)

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
    speedup = (stats_p["median_ms"] / stats_t["median_ms"]) if stats_t["median_ms"] else 0
    return {"kernel": name, "correctness": correctness, "correct": ok, "correct_msg": msg,
            "triton": {**stats_t, "peak_mb": peak_triton},
            "pytorch": {**stats_p, "peak_mb": peak_torch},
            "speedup": speedup, "warmup": warmup, "iterations": iterations}


def main():
    parser = argparse.ArgumentParser(description="STT kernel benchmarks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp32","fp16","bf16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--kernels", nargs="*", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    dtype = dtype_from_str(args.dtype)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, CPU fallback")

    _init_registry(device, dtype)
    env = get_environment(model_name="openai/whisper-base", dtype=args.dtype)
    selected = args.kernels if args.kernels else list(KERNELS.keys())
    results = []
    for name in selected:
        if name not in KERNELS:
            print(f"unknown kernel {name}")
            continue
        print(f"Benchmarking STT kernel {name} ...")
        r = run_one(name, args.warmup, args.iterations, args.atol, args.rtol)
        r.update(env)
        r["device"] = device
        r["dtype"] = args.dtype
        results.append(r)
        print(f"  {name}: triton {r.get('triton',{}).get('median_ms',0):.3f}ms vs torch {r.get('pytorch',{}).get('median_ms',0):.3f}ms speedup {r.get('speedup',0):.2f}x correct={r.get('correct')}")
    from pathlib import Path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, out_dir / "stt_kernels_correctness.json")
    rows = []
    for r in results:
        for backend in ("pytorch","triton"):
            if backend in r:
                s = r[backend]
                rows.append({"kernel": r["kernel"], "backend": backend,
                             "median_ms": s.get("median_ms"), "p95_ms": s.get("p95_ms"), "peak_mb": s.get("peak_mb"),
                             "speedup": r.get("speedup") if backend=="triton" else 1.0,
                             "correct": r.get("correct"), "max_abs_error": r.get("correctness",{}).get("max_abs_error")})
    save_csv(rows, out_dir / "stt_kernels_latency.csv")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
