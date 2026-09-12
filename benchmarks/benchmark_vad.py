"""VAD-leg benchmark via the pipeline engine classes.

Ported from VOICE/vad.py (__main__ timing) and the VAD leg of
VOICE/bench_api_e2e.py, but calling the engine directly instead of HTTP.

Metric: per-utterance wall time + RTF (proc_s / audio_dur_s).

Run: PYTHONPATH=voice-pipeline python benchmarks/benchmark_vad.py
       [--wav PATH] [--iters N]
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_audio(path, sr_want=16000):
    """Load mono float audio; falls back to synthetic 3s noise."""
    if path and os.path.exists(path):
        try:
            import soundfile as sf

            audio, sr = sf.read(path)
            if getattr(audio, "ndim", 1) > 1:
                audio = audio.mean(axis=1)
            return audio, sr
        except Exception as exc:
            print(f"soundfile load failed ({exc}); using synth audio")
    import numpy as np

    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(sr_want * 3).astype("float32"), sr_want)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None)
    ap.add_argument("--iters", type=int, default=20)
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

    audio, sr = _load_audio(args.wav)
    dur = len(audio) / float(sr)
    # Warmup (stateful VAD: resets internally per segments() call).
    try:
        eng.vad_segments(audio)
    except Exception as exc:
        print(f"OMITTED: vad_segments failed ({exc})")
        return 2

    dts = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        segs = eng.vad_segments(audio)
        dts.append(time.perf_counter() - t0)
    dts_ms = sorted(d * 1000 for d in dts)
    q = lambda a, p: a[min(int(p * len(a)), len(a) - 1)]  # noqa: E731
    mean_rtf = statistics.fmean(d / dur for d in dts)
    print(
        f"vad iters={len(dts)} audio={dur:.1f}s segs={len(segs)} "
        f"p50={q(dts_ms, .5):.1f}ms p99={q(dts_ms, .99):.1f}ms "
        f"mean_RTF={mean_rtf:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
