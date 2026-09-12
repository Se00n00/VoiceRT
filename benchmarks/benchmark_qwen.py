"""Qwen-leg benchmark via the pipeline engine classes.

Ported from VOICE/llm_bench.py (voice-style prompts, TTFT/TPS/VRAM) but
driving VoiceEngine.chat() so numbers match the served path.

Run: PYTHONPATH=voice-pipeline python benchmarks/benchmark_qwen.py
       [--max-tokens N]
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [
    "The user said: 'hello, my name is John'. Reply in one short sentence.",
    "The user said: 'what is the future of AI inference'. Reply in one short sentence.",
    "The user said: 'tell me about cats'. Reply in one short sentence.",
]


def _vram_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args(argv)

    try:
        from engine.engine import VoiceEngine
    except Exception as exc:
        print(f"OMITTED: VoiceEngine unavailable ({exc})")
        return 2
    try:
        eng = VoiceEngine()
    except Exception as exc:
        print(f"OMITTED: could not construct VoiceEngine ({exc})")
        return 2

    ttfts, tpss = [], []
    for p in PROMPTS:
        try:
            r = eng.chat(p, max_tokens=args.max_tokens, stream=False)
        except Exception as exc:
            print(f"OMITTED: chat failed ({exc})")
            return 2
        ttfts.append(float(r.get("ttft", 0.0)))
        tpss.append(float(r.get("tps", r.get("decode_tps", 0.0))))
        print(
            f"TTFT={ttfts[-1] * 1000:.0f}ms TPS={tpss[-1]:.1f} "
            f"out={r.get('text', '')[:80]!r}"
        )
    print(
        f"\nmean TTFT={statistics.fmean(ttfts) * 1000:.0f}ms "
        f"mean TPS={statistics.fmean(tpss):.1f} "
        f"vram={_vram_mb():.0f}MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
