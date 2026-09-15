"""TTS Triton benchmark."""
import argparse
from benchmarks.tts._shared import run_tts_benchmark

def main():
    parser = argparse.ArgumentParser(description="TTS Triton benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32","fp16","bf16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--text-length", type=int, default=50)
    parser.add_argument("--token-lengths", type=int, nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="kokoro-82M")
    args = parser.parse_args()
    run_tts_benchmark(args, backend="triton")

if __name__ == "__main__":
    main()
