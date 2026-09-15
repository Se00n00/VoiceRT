# docs

Technical documentation for `voice-pipeline`. Start here, then go deep.

| Doc | What it covers |
|---|---|
| [architecture.md](architecture.md) | Complete system architecture: components, data flow per endpoint, module map, startup sequence, failure modes |
| [capacity.md](capacity.md) | VRAM capacity model: probe → price → plan, queueing design, worked 4GB example, retuning guide |
| [inference_engine.md](inference_engine.md) | Scheduling, batching, paged KV, memory mgmt, QwenRunner, integration with VoiceAgent |
| [benchmarks.md](benchmarks.md) | All measured numbers: fused kernel microbench, inference engine features, full pipeline, cost, capacity |

```
                    ┌─────────────────────────────────────────────┐
                    │                  clients                     │
                    │   curl / frontend / WebSocket /talk          │
                    └──────────────┬──────────────────────────────┘
                                   │ HTTP + WS :8003
                    ┌──────────────▼──────────────────────────────┐
                    │              server.py (FastAPI)               │
                    │  /health  │  /metrics  │  /talk (two-way WS)   │
                    │  FIFO ticket queue (src/models/runtime/scheduler.py)    │
                    └──────────────┬──────────────────────────────┘
                                   │ one VoiceAgent turn graph
                    ┌──────────────▼──────────────────────────────┐
                    │      src/main.py: VoiceAgent                │
                    │  VAD→STT→LLM→TTS turns (LangGraph nodes)    │
                    │  budget guard │ FIFO tickets │ session memory │
                    └──┬────────┬────────┬────────┬───────────────┘
                       │        │        │        │
                 models/    models/   models/  models/
               silero_vad  whisper    qwen      tts
               (ONNX CPU)  (Triton)  (Triton)  (Kokoro+Triton post)
                       │        │        │        │
                    src/models/triton_kernels/qwen.py · whisper.py · tts.py
                    (Triton fast path + exact torch fallback)
                       └────────┴────────┴────────┘
                              src/models/runtime/
                 device · memory · profiler · tensor
                 capacity · scheduler
```

Conventions used across these docs:

- `file:line` references are to the repo root (`voice-pipeline/`).
- All commands assume `PYTHONPATH=.` from inside `voice-pipeline/`.
- "Measured" numbers come from the 2026-09-12 sweep (RTX 3050 4GB, Qwen2.5-0.5B stack, kept for reference) unless noted.
