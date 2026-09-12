"""Whisper-leg benchmark via the pipeline engine classes.

Ported from STT/triton/bench_api.py + STT/triton/bench_throughput.py
(single-leg timing) and the STT section of VOICE/pipeline.py, but calling
VoiceEngine.transcribe() directly instead of HTTP.

Metrics: TTFS (time to first token), total, RTF.

Run: PYTHONPATH=voice-pipeline python benchmarks/benchmark_whisper.py
       [--wav PATH] [--iters N]
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_audio(path, sr_want=16000):
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
    ap.add_argument("--iters", type=int, default=5)
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
    try:
        r0 = eng.transcribe(audio, sr)
    except Exception as exc:
        print(f"OMITTED: transcribe failed ({exc})")
        return 2
    print(f"warmup text={r0.get('text', '')[:70]!r}")

    tots, ttfs, rtfs = [], [], []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        r = eng.transcribe(audio, sr)
        tots.append(time.perf_counter() - t0)
        ttfs.append(float(r.get("ttfs") or 0.0))
        rtfs.append(float(r.get("rtf") or (tots[-1] / max(dur, 1e-9))))
    tots_ms = sorted(t * 1000 for t in tots)
    q = lambda a, p: a[min(int(p * len(a)), len(a) - 1)]  # noqa: E731
    print(
        f"whisper iters={len(tots)} audio={dur:.1f}s "
        f"p50={q(tots_ms, .5):.0f}ms p99={q(tots_ms, .99):.0f}ms "
        f"mean_TTFS={statistics.fmean(ttfs) * 1000:.0f}ms "
        f"mean_RTF={statistics.fmean(rtfs):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
