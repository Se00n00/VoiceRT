"""TTS-leg benchmark via the pipeline engine classes.

Ported from VOICE/tts_bench.py (Kokoro first-chunk TTFA proxy, RTF, VRAM)
but driving VoiceEngine.speak() so numbers match the served path.

Run: PYTHONPATH=voice-pipeline python benchmarks/benchmark_tts.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENTS = [
    "Hello, my name is John.",
    "The future of AI inference is faster and cheaper than ever before.",
    "Cats are fascinating creatures with a wide range of behaviors.",
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

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    rtfs = []
    for i, text in enumerate(SENTS):
        try:
            r = eng.speak(text)
        except Exception as exc:
            print(f"OMITTED: speak failed ({exc})")
            return 2
        wav, sr = r["wav"], int(r.get("sr", 24000))
        dur = len(wav) / float(sr)
        total = float(r.get("synth_s", 0.0))
        rtf = total / max(dur, 1e-9)
        rtfs.append(rtf)
        try:
            import soundfile as sf

            sf.write(os.path.join(out_dir, f"tts_{i:02d}.wav"), wav, sr)
            saved = "saved"
        except Exception:
            saved = "not saved (soundfile missing)"
        print(
            f"sent={i} chars={len(text):3d} total={total * 1000:6.0f}ms "
            f"audio={dur:.1f}s RTF={rtf:.3f} VRAM={_vram_mb():.0f}MB {saved}"
        )
    print(f"mean RTF={statistics.fmean(rtfs):.3f} wavs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
