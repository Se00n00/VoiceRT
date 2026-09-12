"""Streaming example: chunked-file simulated streaming turn.

Honest scope (mirrors VOICE/stream.py): utterance-level STT (no partials
yet) + sentence streaming for LLM->TTS. The input file is fed in
--chunk-ms slices to show incremental VAD progress, then one full
engine.stream_turn() produces the reply.

Run: PYTHONPATH=voice-pipeline python examples/streaming.py IN.wav [OUT.wav]
       [--chunk-ms 320]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_wav(path):
    try:
        import soundfile as sf

        audio, sr = sf.read(path)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        import wave

        import numpy as np

        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0, sr


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_in")
    ap.add_argument("wav_out", nargs="?", default="reply_stream.wav")
    ap.add_argument("--chunk-ms", type=int, default=320)
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

    audio, sr = load_wav(args.wav_in)
    step = max(1, int(sr * args.chunk_ms / 1000))
    n_chunks = (len(audio) + step - 1) // step
    for i in range(n_chunks):
        chunk = audio[i * step:(i + 1) * step]
        try:
            segs = eng.vad_segments(chunk)
        except Exception:
            segs = []
        t = (i + 1) * args.chunk_ms / 1000
        print(f"chunk {i + 1}/{n_chunks} t={t:.2f}s segs={len(segs)}", flush=True)

    r = eng.stream_turn(audio, sr)
    try:
        import soundfile as sf

        sf.write(args.wav_out, r["wav"], 24000)
    except Exception:
        import wave

        import numpy as np

        pcm = (np.asarray(r["wav"], dtype="float32") * 32767).astype("<i2")
        with wave.open(args.wav_out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm.tobytes())
    print(f"user : {r['text']!r}\nasst : {r['reply']!r}")
    print(f"TTFA={r['ttfa_s'] * 1000:.0f}ms total={r['total_s'] * 1000:.0f}ms "
          f"-> {args.wav_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
