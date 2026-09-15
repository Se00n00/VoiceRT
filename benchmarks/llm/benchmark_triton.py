"""LLM full-model Triton benchmark (uses Triton where available)."""
import argparse
from pathlib import Path

import src.models.triton_kernels.qwen as qwen_surface

from benchmarks.llm._shared import run_llm_benchmark


def main():
    parser = argparse.ArgumentParser(description="LLM Triton benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp32","fp16","bf16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--seq-lengths", type=int, nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    args = parser.parse_args()
    # Ensure Triton enabled if possible
    # If CUDA not available it will fallback gracefully
    run_llm_benchmark(args, backend="triton")


if __name__ == "__main__":
    main()
