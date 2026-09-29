#!/usr/bin/env python3
"""Per-leg micro-profile: STT / LLM-single / TTS in isolation, warmed.

Prints stage timings so optimization targets the true hot spot.
Usage: PYTHONPATH=. .venv/bin/python benchmarks/leg_profile.py
"""

import asyncio
import time

import numpy as np


async def main_async() -> int:
    from src.models.llm import LlmConfig, LlmModel
    from src.models.stt import SttModel
    from src.models.tts import TtsModel

    llm = LlmModel(LlmConfig(model="Qwen/Qwen3-0.6B", backend="qwen",
                             max_tokens=32))
    stt = SttModel()
    tts = TtsModel()
    t0 = time.perf_counter()
    await llm.warm()
    await stt.warm()
    await tts.warm()
    print(f"warm: {time.perf_counter() - t0:.1f}s", flush=True)

    # input speech: 2s synth
    out = await tts.speak("hello, what time is it")
    from engine.audio import resample
    wav = np.asarray(resample(np.asarray(out.wav, np.float32).ravel(),
                              24000, 16000), dtype=np.float32).ravel()
    print(f"input {len(wav) / 16000:.2f}s", flush=True)

    # -- STT x3 --
    for i in range(3):
        t = time.perf_counter()
        r = await stt.transcribe(wav, 16000)
        print(f"stt[{i}]: {time.perf_counter() - t:.3f}s text={r.text!r}",
              flush=True)

    # -- LLM single generate x3 (short voice prompt, no agent loop) --
    msgs = [{"role": "system",
             "content": "Reply in one very short sentence."},
            {"role": "user", "content": "Hello, what time is it?"}]
    for i in range(3):
        t = time.perf_counter()
        r = await llm.generate(msgs, max_tokens=32)
        dt = time.perf_counter() - t
        n = len(r.output_ids)
        print(f"llm[{i}]: {dt:.3f}s ids={n} ttft={r.ttft_s:.3f}s "
              f"tps={r.tps:.1f} text={r.text!r}", flush=True)

    # -- TTS x3 (short reply) --
    for i in range(3):
        t = time.perf_counter()
        a = await tts.speak("Hello! How can I help?")
        print(f"tts[{i}]: {time.perf_counter() - t:.3f}s "
              f"wav={len(a.wav) / 24000:.2f}s", flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
