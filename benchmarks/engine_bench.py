"""Benchmark script: compare inference engine features vs base — real Qwen only.

Each config runs in an isolated subprocess to avoid cross-run VRAM leakage
(previous in-process loop caused 3rd+ configs to OOM and silently fall back).
Fails loud if runner is not QwenRunner.
"""
import argparse
import gc
import json
import subprocess
import sys
import time

import torch

sys.path.insert(0, ".")

from src.inference import InferenceEngine, EngineConfig, SamplingParams


def build_configs():
    return {
        "base": EngineConfig(
            model="Qwen/Qwen3-0.6B",
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enable_cuda_graph=False,
            enable_fused_attention=False,
            max_batch_size=4,
            num_blocks=16,
            kv_cache_dtype="fp16",
        ),
        "prefix": EngineConfig(
            model="Qwen/Qwen3-0.6B",
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            enable_cuda_graph=False,
            enable_fused_attention=False,
            max_batch_size=4,
            num_blocks=16,
            kv_cache_dtype="fp16",
        ),
        "chunked": EngineConfig(
            model="Qwen/Qwen3-0.6B",
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enable_cuda_graph=False,
            enable_fused_attention=False,
            max_batch_size=4,
            num_blocks=16,
            kv_cache_dtype="fp16",
        ),
        "cudagraph": EngineConfig(
            model="Qwen/Qwen3-0.6B",
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enable_cuda_graph=True,
            enable_fused_attention=False,
            max_batch_size=4,
            num_blocks=16,
            kv_cache_dtype="fp16",
        ),
        "all": EngineConfig(
            model="Qwen/Qwen3-0.6B",
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enable_cuda_graph=True,
            enable_fused_attention=True,
            max_batch_size=4,
            num_blocks=16,
            kv_cache_dtype="fp16",
        ),
    }


def run_one(name, config, num_requests=8, max_tokens=32):
    """Run single benchmark in this process — real Qwen only."""
    print(f"\n=== {name} ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eng = InferenceEngine(config, device=device, runner="qwen")
    runner_name = type(eng.runner).__name__
    print(f"Device: {eng.device}, Runner: {runner_name}", flush=True)
    if runner_name != "QwenRunner":
        raise RuntimeError(f"expected QwenRunner, got {runner_name} — dummy removed")
    if not getattr(eng.runner, "loaded", False):
        raise RuntimeError("QwenRunner weights not loaded — fail loud, no dummy fallback")

    for i in range(2):
        eng.add_request(f"warm{i}", [1, 2, 3], SamplingParams(max_tokens=4))
        while eng.has_unfinished():
            eng.step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(num_requests):
        eng.add_request(f"req{i}", [1, 2, 3, 4, 5], SamplingParams(max_tokens=max_tokens))

    total_tokens = 0
    while eng.has_unfinished():
        outs = eng.step()
        total_tokens += len(outs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    stats = eng.stats()
    print(f"  Time: {dt:.3f}s, Tokens: {total_tokens}, Throughput: {total_tokens/dt:.1f} tok/s", flush=True)
    print(f"  Steps: {stats['steps']}, KV mem: {stats['kv_cache']['memory_mb']:.1f}MB", flush=True)
    if "prefix_cache" in stats:
        print(f"  Prefix cache: {stats['prefix_cache']}", flush=True)
    if "cuda_graph" in stats:
        print(f"  CUDA graph: {stats['cuda_graph']}", flush=True)

    result = {
        "name": name,
        "time_s": dt,
        "tokens": total_tokens,
        "throughput": total_tokens / dt if dt > 0 else 0,
        "runner": runner_name,
        "stats": {k: v for k, v in stats.items() if k in ("steps", "prefix_cache", "cuda_graph")},
    }

    del eng
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run single config in this process (subprocess mode)")
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    configs = build_configs()

    if args.only is not None:
        # subprocess worker: run one config, print JSON result
        if args.only not in configs:
            raise ValueError(f"unknown config {args.only}")
        r = run_one(args.only, configs[args.only], args.num_requests, args.max_tokens)
        print("RESULT_JSON:" + json.dumps(r), flush=True)
        return

    # parent: spawn isolated subprocess per config
    names = ["base", "prefix", "chunked", "cudagraph", "all"]
    results = []
    for name in names:
        cmd = [sys.executable, __file__, "--only", name,
               "--num-requests", str(args.num_requests),
               "--max-tokens", str(args.max_tokens)]
        print(f"\n>>> spawning subprocess for {name}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout[-2000:])
        if proc.returncode != 0:
            print(proc.stderr[-3000:])
            raise RuntimeError(f"benchmark {name} failed (rc={proc.returncode}) — no dummy fallback")
        # parse RESULT_JSON
        result = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line[len("RESULT_JSON:"):])
        if result is None:
            raise RuntimeError(f"no RESULT_JSON from {name}")
        if result.get("runner") != "QwenRunner":
            raise RuntimeError(f"{name} did not use real QwenRunner: {result}")
        results.append(result)

    print("\n=== SUMMARY (real QwenRunner only) ===")
    base_tput = results[0]["throughput"]
    for r in results:
        delta = (r["throughput"] / base_tput - 1.0) * 100 if base_tput > 0 else 0
        print(f"  {r['name']}: {r['throughput']:.1f} tok/s ({r['time_s']:.3f}s) delta {delta:+.1f}% vs base")


if __name__ == "__main__":
    main()
