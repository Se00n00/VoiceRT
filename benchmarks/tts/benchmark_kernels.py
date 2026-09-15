"""TTS kernel-level: Triton vs PyTorch."""
import argparse
import torch

from benchmarks.common.benchmark import seed_everything, dtype_from_str, ensure_output_dir, save_json, save_csv
from benchmarks.common.environments import get_environment
from benchmarks.common.metrics import latency_stats, correctness_metrics, check_tolerance
from benchmarks.common.timing import measure_latencies
from benchmarks.common.memory import reset_peak, snapshot

KERNELS = {}

def _init_registry(device, dtype):
    from src.models.triton_kernels.conv1d import conv1d_silu as triton_conv, in1d_silu as triton_in
    import torch.nn.functional as F

    def conv_torch(x,w,b, stride=1, padding=1, dilation=1):
        return F.silu(F.conv1d(x.float(), w.float(), b.float(), stride=stride, padding=padding, dilation=dilation))
    def in_torch(x):
        m = x.float().mean(-1, keepdim=True)
        va = x.float().var(-1, unbiased=False, keepdim=True)
        return F.silu((x.float() - m) / torch.sqrt(va + 1e-5))

    def make_conv():
        N, Cin, Cout, L, K = 1, 32, 32, 200, 3
        x = torch.randn(N, Cin, L, device=device, dtype=dtype)
        w = torch.randn(Cout, Cin, K, device=device, dtype=dtype)
        b = torch.randn(Cout, device=device, dtype=dtype)
        return (x,w,b), {"stride":1,"padding":1}

    def make_in():
        N,C,L = 2, 32, 200
        x = torch.randn(N,C,L, device=device, dtype=dtype)
        return (x,), {}

    KERNELS.clear()
    KERNELS["conv1d_silu"] = (triton_conv, conv_torch, make_conv)
    KERNELS["in1d_silu"] = (triton_in, in_torch, make_in)

    # Also add resample/postprocess are not triton heavy, skip
    # Add silu alone? Already via conv.

def run_one(name, warmup, iterations, atol, rtol):
    triton_fn, torch_fn, make_inputs = KERNELS[name]
    (args, kwargs) = make_inputs()
    with torch.no_grad():
        try:
            out_torch = torch_fn(*args, **kwargs)
            out_triton = triton_fn(*args, **kwargs)
        except Exception as e:
            return {"kernel": name, "error": str(e)}
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
    parser = argparse.ArgumentParser(description="TTS kernel benchmarks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32","fp16","bf16"])
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
        print("CUDA not available")
    _init_registry(device, dtype)
    env = get_environment(model_name="tts-kokoro", dtype=args.dtype)
    selected = args.kernels if args.kernels else list(KERNELS.keys())
    results = []
    for name in selected:
        if name not in KERNELS:
            continue
        print(f"Benchmarking TTS kernel {name} ...")
        r = run_one(name, args.warmup, args.iterations, args.atol, args.rtol)
        r.update(env)
        r["device"] = device
        r["dtype"] = args.dtype
        results.append(r)
        print(f"  {name}: triton {r.get('triton',{}).get('median_ms',0):.3f}ms vs torch {r.get('pytorch',{}).get('median_ms',0):.3f}ms speedup {r.get('speedup',0):.2f}x correct={r.get('correct')}")
    from pathlib import Path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(results, out_dir / "tts_kernels_correctness.json")
    rows = []
    for r in results:
        for backend in ("pytorch","triton"):
            if backend in r:
                s = r[backend]
                rows.append({"kernel": r["kernel"], "backend": backend, "median_ms": s.get("median_ms"), "peak_mb": s.get("peak_mb"), "speedup": r.get("speedup") if backend=="triton" else 1.0, "correct": r.get("correct"), "max_abs_error": r.get("correctness",{}).get("max_abs_error")})
    save_csv(rows, out_dir / "tts_kernels_latency.csv")
    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
