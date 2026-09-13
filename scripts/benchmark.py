"""Dispatcher for all voice-pipeline benchmarks.

Run: PYTHONPATH=voice-pipeline python scripts/benchmark.py {vad,whisper,qwen,tts,pipeline,capacity} [bench args...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEGS = ("vad", "whisper", "qwen", "tts", "pipeline", "capacity", "e2e")

_MOD = {"capacity": "benchmarks.bench_capacity"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("leg", choices=LEGS)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)

    mod_name = _MOD.get(args.leg, f"benchmarks.benchmark_{args.leg}")
    try:
        mod = __import__(mod_name, fromlist=["main"])
    except Exception as exc:
        print(f"OMITTED: {mod_name} unavailable ({exc})")
        return 2
    # Never pass None: argparse would re-parse sys.argv (this dispatcher's
    # own argv, starting with the leg name) and crash on it.
    return mod.main(args.rest if args.rest else [])


if __name__ == "__main__":
    raise SystemExit(main())
