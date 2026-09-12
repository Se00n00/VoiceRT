# docs

Technical documentation for `voice-pipeline`. Start here, then go deep.

| Doc | What it covers |
|---|---|
| [architecture.md](architecture.md) | Complete system architecture: components, data flow per endpoint, module map, startup sequence, failure modes |
| [capacity.md](capacity.md) | VRAM capacity model: probe → price → plan, queueing design, worked 4GB example, retuning guide |
| [benchmarks.md](benchmarks.md) | Profiling methodology: what each benchmark measures, JSON schema, graphs, reproducing the README numbers |

```
                    ┌─────────────────────────────────────────────┐
                    │                  clients                     │
                    │   curl / frontend / WebSocket /v1/talk       │
                    └──────────────┬──────────────────────────────┘
                                   │ HTTP + WS :8003
                    ┌──────────────▼──────────────────────────────┐
                    │              server/ (FastAPI)               │
                    │  routes.py  │ schemas.py │ websocket.py      │
                    │  FIFO ticket queue (runtime/scheduler.py)    │
                    └──────────────┬──────────────────────────────┘
                                   │ one VoiceEngine singleton
                    ┌──────────────▼──────────────────────────────┐
                    │           engine/engine.py                   │
                    │  VoiceEngine: VAD→STT→LLM→TTS turns          │
                    │  budget guard │ profiler │ session memory    │
                    └──┬────────┬────────┬────────┬───────────────┘
                       │        │        │        │
                 models/    models/   models/  models/
               silero_vad  whisper    qwen      tts
               (ONNX CPU)  (Triton)  (Triton)  (Kokoro+Triton post)
                       │        │        │        │
                    triton_kernels/qwen.py · whisper.py · tts.py
                    (Triton fast path + exact torch fallback)
                       └────────┴────────┴────────┘
                              runtime/
                 device · memory · profiler · tensor
                 capacity · scheduler
```

Conventions used across these docs:

- `file:line` references are to the repo root (`voice-pipeline/`).
- All commands assume `PYTHONPATH=.` from inside `voice-pipeline/`.
- "Measured" numbers come from `benchmarks/results/capacity.md` (RTX 3050 4GB, 2026-09-12) unless noted.
