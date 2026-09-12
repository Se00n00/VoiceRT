"""Offline example: wav file -> reply wav via VoiceEngine.

Loads one utterance, runs a full VAD->STT->LLM->TTS turn, saves the reply.

Run: PYTHONPATH=voice-pipeline python examples/offline.py IN.wav [OUT.wav]
"""
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
            n = w.getnframes()
            raw = w.readframes(n)
        audio = (
            np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        )
        return audio, sr


def save_wav(path, wav, sr):
    try:
        import soundfile as sf

        sf.write(path, wav, sr)
        return
    except Exception:
        pass
    import wave

    import numpy as np

    pcm = (np.asarray(wav, dtype="float32") * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("wav_in")
    ap.add_argument("wav_out", nargs="?", default="reply.wav")
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

    audio, sr = load_wav(args.wav_in)
    r = eng.stream_turn(audio, sr)
    save_wav(args.wav_out, r["wav"], 24000)
    print(f"user : {r['text']!r}\nasst : {r['reply']!r}")
    print(f"TTFA={r['ttfa_s'] * 1000:.0f}ms total={r['total_s'] * 1000:.0f}ms "
          f"-> {args.wav_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
