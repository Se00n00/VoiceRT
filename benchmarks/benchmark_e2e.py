"""Real-speech end-to-end benchmark: wav files -> full VAD->STT->LLM->TTS turns.

Unlike benchmark_pipeline.py (synthetic sine plumbing) this feeds REAL
speech and reports what a user feels: TTFA, E2E, STT accuracy proxy
(transcript shown), reply text, RTF, VRAM — per file plus p50 summary.

Inputs (first found wins, or pass your own):
  1. --wav PATH / --dir DIR (e.g. LibriSpeech test-clean wavs)
  2. benchmarks/results/tts_*.wav (Kokoro-synthesized real speech prosody)
  3. falls back to synthesizing one prompt via the engine itself.

Comparison: every run writes results/e2e_<ts>.json; pass
  --compare results/e2e_<older>.json
to print per-file and p50 deltas (TTFA/E2E/RTF) vs that baseline — this is
how you compare the whole pipeline before/after a change. To compare
against OTHER systems, quote the TTFA/E2E/RTF rows: same definitions as
docs/benchmarks.md (entry->first-audio, entry->done, proc/audio).

Run: PYTHONPATH=. python benchmarks/benchmark_e2e.py [--dir DIR] [--compare JSON]
     PYTHONPATH=. python scripts/benchmark.py e2e [--dir DIR]
"""
import argparse
import datetime
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OWN_TTS = sorted(glob.glob(os.path.join(RESULTS, "tts_*.wav")))


def load_wav_16k(path):
    """wav file -> (mono float32 @16k, dur_s). soundfile preferred."""
    try:
        import soundfile as sf

        audio, sr = sf.read(path)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
    except Exception:
        import wave

        import numpy as np

        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
        import numpy as _np

        audio = _np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32).ravel()
    if sr != 16000 and len(audio):
        old = np.linspace(0.0, 1.0, num=len(audio))
        new = np.linspace(0.0, 1.0, num=max(int(round(len(audio) * 16000 / sr)), 1))
        audio = np.interp(new, old, audio).astype(np.float32)
    return audio, len(audio) / 16000.0


def collect_inputs(wav=None, dir=None):
    if wav:
        return [wav]
    if dir:
        files = sorted(glob.glob(os.path.join(dir, "*.wav")))
        if not files:
            print(f"no wavs in {dir}")
            return []
        return files
    if OWN_TTS:
        print(f"using Kokoro-synthesized speech: {len(OWN_TTS)} files")
        return OWN_TTS
    return []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--compare", default=None,
                    help="older results/e2e_<ts>.json to diff against")
    args = ap.parse_args(argv)

    files = collect_inputs(args.wav, args.dir)
    try:
        from engine.engine import VoiceEngine
    except Exception as exc:
        print(f"OMITTED: VoiceEngine unavailable ({exc})")
        return 2
    if not files:
        # Last resort: synthesize one prompt, then turn on its own audio.
        try:
            eng0 = VoiceEngine()
        except Exception as exc:
            print(f"OMITTED: could not construct VoiceEngine ({exc})")
            return 2
        import soundfile as sf

        wav, sr = eng0.speak("Benchmark fallback sentence for end to end.")
        fb = os.path.join(RESULTS, "e2e_fallback.wav")
        sf.write(fb, wav, sr)
        files = [fb]
        print(f"synthesized fallback input: {fb}")
        eng = eng0
    else:
        try:
            eng = VoiceEngine()
        except Exception as exc:
            print(f"OMITTED: could not construct VoiceEngine ({exc})")
            return 2

    rows = []
    for path in files:
        audio, dur = load_wav_16k(path)
        r = eng.stream_turn(audio, 16000)
        row = {"file": os.path.basename(path), "dur_s": dur,
               "text": r["text"], "reply": r["reply"],
               "ttfa_ms": r["ttfa_s"] * 1000, "e2e_ms": r["total_s"] * 1000,
               "rtf": r["total_s"] / max(dur, 1e-9),
               "vram_mb": r.get("vram_mb", 0.0)}
        rows.append(row)
        print(f"{row['file']}: TTFA {row['ttfa_ms']:.0f}ms "
              f"E2E {row['e2e_ms']:.0f}ms RTF {row['rtf']:.2f} "
              f"VRAM {row['vram_mb']:.0f}MB")
        print(f"  user : {row['text']!r}")
        print(f"  asst : {row['reply']!r}", flush=True)

    def p50(key):
        return statistics.median([r[key] for r in rows]) if rows else 0.0

    summary = {"ttfa_p50_ms": p50("ttfa_ms"), "e2e_p50_ms": p50("e2e_ms"),
               "rtf_p50": p50("rtf"),
               "vram_mb": max([r["vram_mb"] for r in rows] or [0.0])}
    print(f"p50: TTFA {summary['ttfa_p50_ms']:.0f}ms "
          f"E2E {summary['e2e_p50_ms']:.0f}ms RTF {summary['rtf_p50']:.2f}")

    os.makedirs(RESULTS, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(RESULTS, f"e2e_{ts}.json")
    with open(jp, "w") as f:
        json.dump({"ts": ts, "rows": rows, "summary": summary}, f, indent=1)
    print(f"wrote {jp}")

    if args.compare:
        with open(args.compare) as f:
            base = json.load(f)
        bsum = base.get("summary", {})
        print(f"--- vs baseline {args.compare} ---")
        for k in ("ttfa_p50_ms", "e2e_p50_ms", "rtf_p50"):
            old, new = bsum.get(k, 0.0), summary.get(k, 0.0)
            d = new - old
            print(f"{k}: {old:.1f} -> {new:.1f} ({d:+.1f})")
        brow = {r["file"]: r for r in base.get("rows", [])}
        for r in rows:
            b = brow.get(r["file"])
            if b:
                print(f"{r['file']}: TTFA {b['ttfa_ms']:.0f}->{r['ttfa_ms']:.0f} "
                      f"E2E {b['e2e_ms']:.0f}->{r['e2e_ms']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
