"""STT PyTorch baseline."""
import argparse
from benchmarks.stt._shared import run_stt_benchmark

def main():
    parser = argparse.ArgumentParser(description="STT PyTorch baseline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp32","fp16","bf16"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    parser.add_argument("--audio-duration", type=float, default=5)
    parser.add_argument("--durations", type=float, nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="openai/whisper-base")
    args = parser.parse_args()
    run_stt_benchmark(args, backend="pytorch")

if __name__ == "__main__":
    main()
