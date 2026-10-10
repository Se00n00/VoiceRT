"""A/B voice samples: current Kokoro-82M vs KittenTTS-mini-0.8, male + female.

Same 3-sentence script for every voice. Writes 24kHz WAVs to
benchmarks/results/tts_samples/ and prints wall time + RTF per voice.

Usage: ``PYTHONPATH=. .venv/bin/python benchmarks/tts_ab.py``
"""

import os
import sys
import time

SENTENCES = {
    "neutral": "The small harbor town wakes early, with fishing boats already drifting past the lighthouse.",
    "question": "Are you really sure the northern road is still open after last night's storm?",
    "paragraph": ("The archive room smelled of dust and old paper. "
                  "Mara ran her finger along the third shelf until she found the missing ledger. "
                  "Behind her, the door clicked shut on its own."),
}

KOKORO_VOICES = {"female": "af_sarah", "male": "am_adam"}
KITTEN_VOICES = {"female": "Bella", "male": "Jasper"}


def save_wav(path: str, audio, sr: int) -> None:
    import numpy as np
    import soundfile as sf

    a = np.asarray(audio, dtype=np.float32).ravel()
    sf.write(path, a, sr)


def main() -> int:
    outdir = os.path.join("benchmarks", "results", "tts_samples")
    os.makedirs(outdir, exist_ok=True)

    print("== kokoro-82M (current leg) ==", flush=True)
    import asyncio

    from src.models.tts import TtsModel

    kokoro = TtsModel()

    async def kokoro_synth() -> None:
        await kokoro.warm()
        for gender, voice in KOKORO_VOICES.items():
            for key, text in SENTENCES.items():
                t0 = time.perf_counter()
                out = await kokoro.speak(text, voice=voice)
                dt = time.perf_counter() - t0
                dur = len(out.wav) / float(out.sample_rate or 24000)
                path = os.path.join(outdir, f"kokoro_{gender}_{key}.wav")
                save_wav(path, out.wav, int(out.sample_rate or 24000))
                print(f"kokoro/{gender}/{key}: {dur:.1f}s audio in {dt:.1f}s "
                      f"(RTF {dt / max(dur, 1e-6):.2f}) -> {path}", flush=True)

    asyncio.run(kokoro_synth())

    print("== kitten-tts-mini-0.8 ==", flush=True)
    from kittentts import KittenTTS

    t0 = time.perf_counter()
    kitten = KittenTTS("KittenML/kitten-tts-mini-0.8")
    print(f"kitten load: {time.perf_counter() - t0:.1f}s", flush=True)
    for gender, voice in KITTEN_VOICES.items():
        for key, text in SENTENCES.items():
            t0 = time.perf_counter()
            audio = kitten.generate(text, voice=voice)
            dt = time.perf_counter() - t0
            import numpy as np

            a = np.asarray(audio, dtype=np.float32).ravel()
            dur = len(a) / 24000.0
            path = os.path.join(outdir, f"kitten_{gender}_{key}.wav")
            save_wav(path, a, 24000)
            print(f"kitten/{gender}/{key}: {dur:.1f}s audio in {dt:.1f}s "
                  f"(RTF {dt / max(dur, 1e-6):.2f}) -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
