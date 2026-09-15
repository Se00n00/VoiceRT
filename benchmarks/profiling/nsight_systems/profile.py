"""Nsight Systems helper — prints nsys profile commands."""
import argparse
from pathlib import Path

CMDS = {
    "llm_pytorch": "nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas --stats=true -o profiling/nsight_systems/llm_pytorch python -m benchmarks.llm.benchmark_pytorch --warmup 5 --iterations 20",
    "llm_triton": "nsys profile --trace=cuda,nvtx,osrt --stats=true -o profiling/nsight_systems/llm_triton python -m benchmarks.llm.benchmark_triton --warmup 5 --iterations 20",
    "stt_pytorch": "nsys profile --trace=cuda,nvtx,osrt --stats=true -o profiling/nsight_systems/stt_pytorch python -m benchmarks.stt.benchmark_pytorch --warmup 5 --iterations 20",
    "stt_triton": "nsys profile --trace=cuda,nvtx --stats=true -o profiling/nsight_systems/stt_triton python -m benchmarks.stt.benchmark_triton --warmup 5 --iterations 20",
    "tts_pytorch": "nsys profile --trace=cuda,nvtx --stats=true -o profiling/nsight_systems/tts_pytorch python -m benchmarks.tts.benchmark_pytorch --warmup 5 --iterations 10",
    "tts_triton": "nsys profile --trace=cuda,nvtx --stats=true -o profiling/nsight_systems/tts_triton python -m benchmarks.tts.benchmark_triton --warmup 5 --iterations 10",
}

def main():
    parser = argparse.ArgumentParser(description="Nsight Systems profiling commands")
    parser.add_argument("--list", action="store_true", help="list all commands")
    parser.add_argument("--target", choices=list(CMDS.keys()), default=None)
    parser.add_argument("--output", default="profiling/nsight_systems")
    args = parser.parse_args()
    if args.list or args.target is None:
        print("Nsight Systems — complete inference execution (identify kernel launch overhead, CPU/GPU sync, idle gaps, sequencing, H2D/D2H, concurrency):")
        for k, v in CMDS.items():
            print(f"\n  {k}:")
            print(f"    {v}")
        print("\nView with: nsys-ui <report>.qdrep  or  nsys stats <report>.qdrep")
        print("Metrics: kernel launch overhead, sync, idle, sequencing, preprocessing, H2D/D2H, concurrent exec")
    else:
        print(CMDS[args.target])

if __name__ == "__main__":
    main()
