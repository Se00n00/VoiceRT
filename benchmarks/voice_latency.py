#!/usr/bin/env python3
"""Voice E2E latency probe: synth speech in -> full voice turn -> node timings.

Warms VAD + STT + LLM + TTS, synthesizes the input utterance with the
warmed Kokoro leg (reproducible, no mic needed), then runs N voice turns
through VoiceAgent._voice_turns and prints per-node seconds + totals.

First turn is warmup (Triton autotune, first-touch) and excluded from means.

Usage:
  PYTHONPATH=. .venv/bin/python benchmarks/voice_latency.py [--turns 5]
      [--say "hello, what time is it"] [--backend qwen]
      [--model Qwen/Qwen3-0.6B] [--max-tokens 48] [--fast]
"""

import argparse
import asyncio
import time

import numpy as np


def build_agent(args):
    from src.main import VoiceAgent, VoiceAgentConfig
    from src.models.llm import LlmConfig
    from src.models.tts import TtsConfig

    llm = LlmConfig(model=args.model, backend=args.backend,
                    max_tokens=args.max_tokens)
    tts = TtsConfig(speed=args.speed)
    cfg = VoiceAgentConfig(llm=llm, tts=tts,
                           sessions_dir="/tmp/voice-lat-sessions",
                           fast_voice=args.fast)
    return VoiceAgent(cfg)


async def main_async(args) -> int:
    from engine.audio import resample

    agent = build_agent(args)
    t0 = time.perf_counter()
    await agent.warm()
    print(f"warm done in {time.perf_counter() - t0:.1f}s "
          f"missing={agent.missing}", flush=True)

    # reproducible input speech from the warmed TTS leg itself
    out = await agent.tts.speak(args.say)
    wav24 = np.asarray(out.wav, dtype=np.float32).ravel()
    wav = np.asarray(resample(wav24, 24000, 16000), dtype=np.float32).ravel()
    print(f"input: {args.say!r} -> {len(wav) / 16000.0:.2f}s @16k",
          flush=True)

    # clock soak: Kole GPUs ramp from idle over ~30s of load; without this
    # the first measured turns pay idle-clock prices and every run reads
    # differently. Throwaway work on each leg, then measure.
    for _ in range(3):
        await agent.llm.generate(
            [{"role": "user", "content": "say ok"}], max_tokens=4)
    await agent.stt.transcribe(wav[:16000], 16000)
    await agent.tts.speak("ok.")
    print("clock soak done", flush=True)

    if args.idle:
        print(f"idling {args.idle}s to simulate production gaps...",
              flush=True)
        await asyncio.sleep(args.idle)

    rows = []
    for i in range(args.turns):
        sid = f"lat-{i}"
        summary = {}
        n_ids = 0
        async for ev in agent(wav, 16000, sid):
            if ev.node == "turn":
                summary = dict(ev.data or {})
            elif ev.node == "llm" and ev.kind == "done":
                n_ids = int((ev.data or {}).get("ids", 0) or 0)
        ns = summary.get("node_s", {}) or {}
        rows.append({
            "vad": ns.get("vad", 0.0), "stt": ns.get("stt", 0.0),
            "llm": ns.get("llm", 0.0), "tts": ns.get("tts", 0.0),
            "ttfa": summary.get("ttfa_s", 0.0),
            "total": summary.get("total_s", 0.0),
            "ids": n_ids,
            "text": (summary.get("text", "") or "")[:60],
            "reply": (summary.get("reply", "") or "")[:80],
        })
        r = rows[-1]
        tag = "WARMUP" if i == 0 else f"turn{i}"
        print(f"[{tag}] vad={r['vad'] * 1000:6.0f}ms stt={r['stt'] * 1000:6.0f}ms "
              f"llm={r['llm'] * 1000:6.0f}ms(ids={r['ids']}) tts={r['tts'] * 1000:6.0f}ms "
              f"ttfa={r['ttfa'] * 1000:6.0f}ms total={r['total'] * 1000:6.0f}ms",
              flush=True)
        print(f"   stt={r['text']!r} reply={r['reply']!r}", flush=True)

    meas = rows[1:] or rows
    for k in ("vad", "stt", "llm", "tts", "ttfa", "total"):
        vals = sorted(r[k] for r in meas)
        mean = sum(vals) / len(vals)
        med = vals[len(vals) // 2]
        print(f"{k}: mean={mean * 1000:.0f}ms med={med * 1000:.0f}ms "
              f"min={vals[0] * 1000:.0f}ms max={vals[-1] * 1000:.0f}ms "
              f"(n={len(vals)})", flush=True)
    try:
        from src.models.runtime import max_allocated_mb
        print(f"vram peak: {max_allocated_mb():.0f}MB", flush=True)
    except Exception:
        pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="voice E2E latency probe")
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--say", default="hello, what time is it")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="TTS speaking-rate multiplier (Kokoro native)")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="sleep SECS after soak before measured turns")
    ap.add_argument("--fast", action="store_true",
                    help="fast voice path (single LLM call, no agent loop)")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
