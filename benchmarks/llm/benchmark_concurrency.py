"""LLM concurrency benchmark: multiple requests concurrently."""
import argparse
import asyncio
import time
from pathlib import Path

import torch
import numpy as np

from benchmarks.common.benchmark import seed_everything, dtype_from_str, ensure_output_dir, save_json
from benchmarks.common.environments import get_environment
from benchmarks.common.metrics import latency_stats
from benchmarks.common.memory import reset_peak, snapshot


async def _one_request(engine, ids, max_tokens):
    # Run in thread to not block event loop
    loop = asyncio.get_running_loop()
    def _run():
        t0 = time.perf_counter()
        out = engine.generate(ids, max_new_tokens=max_tokens)
        # use ttft from out + total wall
        e2e = time.perf_counter() - t0
        ttft = out.get("ttft", 0)
        # TPOT approximated as (e2e - ttft)/(out_len-1)
        n = len(out.get("ids", []))
        tpot = (e2e - ttft)/max(n-1,1) if n>1 else 0
        return e2e, ttft, tpot, n
    return await loop.run_in_executor(None, _run)


def main():
    parser = argparse.ArgumentParser(description="LLM concurrency benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp32","fp16","bf16"])
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    seed_everything(args.seed)
    dtype_torch = dtype_from_str(args.dtype)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    conc_levels = args.concurrency if args.concurrency else [1,2,4,8,16]
    env = get_environment(model_name=args.model, dtype=args.dtype)

    # Load engine once
    from benchmarks.llm._shared import _get_engine, _make_input_ids
    engine, is_synth = _get_engine(args.model, device, dtype_torch, max_new_tokens=args.output_length)
    print(f"Concurrency benchmark backend engine synthetic={is_synth}")

    # tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        has_tok = True
    except Exception:
        has_tok = False

    all_rows = []

    for conc in conc_levels:
        print(f"\nConcurrency {conc} ...")
        # warmup
        ids = _make_input_ids(None, args.input_length, args.seed)
        for _ in range(args.warmup):
            try:
                engine.generate(ids, max_new_tokens=args.output_length)
            except Exception:
                break
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        e2es = []
        ttfts = []
        tpots = []
        peaks = []
        for it in range(args.iterations):
            reset_peak()
            # prepare batch of ids
            batch_ids = [_make_input_ids(None, args.input_length, args.seed + i) for i in range(conc)]

            async def _run_batch():
                t0 = time.perf_counter()
                tasks = [_one_request(engine, bid, args.output_length) for bid in batch_ids]
                results = await asyncio.gather(*tasks)
                total_wall = time.perf_counter() - t0
                return results, total_wall

            results, total_wall = asyncio.run(_run_batch())
            # results: list of (e2e, ttft, tpot, n)
            for e2e, ttft, tpot, n in results:
                e2es.append(e2e)
                ttfts.append(ttft)
                tpots.append(tpot)
            peaks.append(snapshot()["peak_mb"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        stats_e2e = latency_stats(e2es)
        stats_ttft = latency_stats(ttfts)
        stats_tpot = latency_stats(tpots)
        median_e2e = stats_e2e["median_ms"]/1000 if stats_e2e["median_ms"] else 1e-9
        # Throughput: requests per sec = conc / median_total_wall? But we measured per-request e2e median, not total wall.
        # For concurrency, better use total wall per iteration: total_wall per iter is sum? We approximated per-request median.
        # We'll compute throughput as conc * iterations / sum(e2es) * conc? Simplified:
        total_tokens = conc * args.output_length
        # Use sum of e2es across concurrency? Use total_wall approx as median e2e for single? For report, use conc / median_e2e * efficiency
        req_tput = conc / median_e2e if median_e2e else 0
        tok_tput = total_tokens / median_e2e if median_e2e else 0

        row = {
            "backend": "triton",
            "model": args.model,
            "dtype": args.dtype,
            "device": device,
            "gpu_name": env["gpu_name"],
            "gpu_memory_mb": env["gpu_memory_mb"],
            "cuda_version": env["cuda_version"],
            "torch_version": env["torch_version"],
            "triton_version": env["triton_version"],
            "python_version": env["python_version"],
            "concurrency": conc,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "median_ms": stats_e2e["median_ms"],
            "p50_ms": stats_e2e["p50_ms"],
            "p95_ms": stats_e2e["p95_ms"],
            "p99_ms": stats_e2e["p99_ms"],
            "mean_ms": stats_e2e["mean_ms"],
            "min_ms": stats_e2e["min_ms"],
            "max_ms": stats_e2e["max_ms"],
            "ttft_p50_ms": stats_ttft["p50_ms"],
            "ttft_p95_ms": stats_ttft["p95_ms"],
            "tpot_p50_ms": stats_tpot["p50_ms"],
            "tpot_p95_ms": stats_tpot["p95_ms"],
            "e2e_p50_ms": stats_e2e["p50_ms"],
            "e2e_p95_ms": stats_e2e["p95_ms"],
            "requests_per_sec": req_tput,
            "output_tokens_per_sec": tok_tput,
            "throughput": tok_tput,
            "peak_mb": float(np.max(peaks)) if peaks else 0,
            "timestamp": env["timestamp"],
        }
        all_rows.append(row)
        print(f"  conc {conc}: median {row['median_ms']:.1f}ms TTFT p50 {row['ttft_p50_ms']:.1f}ms TPOT p50 {row['tpot_p50_ms']:.2f}ms req/s {row['requests_per_sec']:.2f} tok/s {row['output_tokens_per_sec']:.1f}")

    out_dir = ensure_output_dir(args.output_dir)
    # Save concurrency results to llm_concurrency.csv
    import csv
    path = out_dir / "llm_concurrency.csv"
    if all_rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    save_json(all_rows, out_dir / "llm_concurrency.json")
    print(f"Saved concurrency results to {out_dir}")


if __name__ == "__main__":
    main()
