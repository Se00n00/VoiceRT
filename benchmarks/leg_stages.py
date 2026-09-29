#!/usr/bin/env python3
"""Stage-split profile: STT mel/encode/decode-loop/decode-text,
LLM prefill vs per-token, TTS fixed vs per-char. No source changes.
Usage: PYTHONPATH=. .venv/bin/python benchmarks/leg_stages.py
"""

import asyncio
import time

import numpy as np


async def main_async() -> int:
    import torch

    from src.models.llm import LlmConfig, LlmModel
    from src.models.stt import SttModel
    from src.models.tts import TtsModel

    llm = LlmModel(LlmConfig(model="Qwen/Qwen3-0.6B", backend="qwen",
                             max_tokens=32))
    stt = SttModel()
    tts = TtsModel()
    await llm.warm()
    await stt.warm()
    await tts.warm()

    out = await tts.speak("hello, what time is it")
    from engine.audio import resample
    wav = np.asarray(resample(np.asarray(out.wav, np.float32).ravel(),
                              24000, 16000), dtype=np.float32).ravel()

    # ---------- STT stages ----------
    leg = stt._backend()
    proc = stt._processor()
    from src.models.engines.whisper import log_mel_spectrogram, pad_or_trim
    t = time.perf_counter()
    mel = pad_or_trim(log_mel_spectrogram(wav, sr=16000), 3000)
    t_mel = time.perf_counter() - t
    mel_t = torch.from_numpy(np.stack([mel], 0)).to(leg.device)
    t = time.perf_counter()
    with torch.no_grad():
        mem = leg.encode(mel_t)
    t_enc = time.perf_counter() - t
    t = time.perf_counter()
    with torch.no_grad():
        cross = leg.cross_kv(mem)
    t_cross = time.perf_counter() - t
    for mt in (64, 16):
        t = time.perf_counter()
        with torch.no_grad():
            r = leg.transcribe([wav], sr=16000, max_tokens=mt)
        t_dec = time.perf_counter() - t
        t = time.perf_counter()
        text = proc.batch_decode([r["ids"][0]], skip_special_tokens=True)[0]
        t_bdec = time.perf_counter() - t
        print(f"stt: mel={t_mel * 1000:.0f}ms enc={t_enc * 1000:.0f}ms "
              f"cross={t_cross * 1000:.0f}ms loop(mt={mt})={t_dec * 1000:.0f}ms "
              f"batch_decode={t_bdec * 1000:.0f}ms text={text!r}", flush=True)

    # ---------- LLM: prefill vs per-token ----------
    msgs = [{"role": "system", "content": "Reply in one short sentence."},
            {"role": "user", "content": "Hello, what time is it?"}]
    t = time.perf_counter()
    ids = await llm.encode(msgs)
    t_tok = time.perf_counter() - t
    print(f"llm: prompt_ids={len(ids)} encode={t_tok * 1000:.0f}ms",
          flush=True)
    qleg = llm._backend()
    for mt in (1, 32):
        t = time.perf_counter()
        with torch.no_grad():
            r = qleg.generate(ids, mt)
        dt = time.perf_counter() - t
        n = r["ids"] if isinstance(r["ids"], list) else []
        print(f"llm: generate(mt={mt})={dt * 1000:.0f}ms out={len(n)}",
              flush=True)

    # ---------- TTS: fixed vs per-char ----------
    for txt in ("Hi.", "Hello! How can I help?", "Hello! How can I help you "
                "today? I can run commands, read files, and answer questions."):
        t = time.perf_counter()
        a = await tts.speak(txt)
        dt = time.perf_counter() - t
        print(f"tts: chars={len(txt):3} synth={dt * 1000:.0f}ms "
              f"wav={len(a.wav) / 24000:.2f}s", flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
