# docs

Technical documentation for `voice-pipeline`. Start here, then go deep.

| Doc | What it covers |
|---|---|
| [guide.md](guide.md) | Full walkthrough moved from the root README: run steps, architecture, agent loop, benchmarks, TUI, sandbox |
| [**architecture/**](architecture/README.md) | **The system, split by part: [01 Agent](architecture/01-agent.md) · [02 Tools](architecture/02-tools.md) · [03 Memory](architecture/03-memory.md) · [04 Sessions](architecture/04-sessions.md) · [05 Models](architecture/05-models.md) · [06 Processes](architecture/06-processes.md) · [07 Inference](architecture/07-inference.md) · [08 TUI](architecture/08-tui.md)** |
| [**APIReferences/**](APIReferences/README.md) | **Every `:8004` route, one file per group: [System](APIReferences/System.md) · [Turns](APIReferences/Turns.md) · [Voice](APIReferences/Voice.md) · [Talk](APIReferences/Talk.md) · [Chat](APIReferences/Chat.md) · [Sessions](APIReferences/Sessions.md) · [Keys](APIReferences/Keys.md)** |
| [architecture.md](architecture.md) | Older single-file architecture: components, data flow per endpoint, module map, startup sequence, failure modes |
| [capacity.md](capacity.md) | VRAM capacity model: probe → price → plan, queueing design, worked 4GB example, retuning guide |
| [benchmarks.md](benchmarks.md) | All measured numbers: fused kernel microbench, full pipeline, cost, capacity |

```
                    ┌─────────────────────────────────────────────┐
                    │                  clients                     │
                    │   curl / frontend / WS /chat · /term          │
                    └──────────────┬──────────────────────────────┘
                                   │ HTTP + WS :8004
                    ┌──────────────▼──────────────────────────────┐
                    │          src.server (FastAPI, :8004)           │
                    │  /health · /metrics · /chat WS · /talk/offer  │
                    │  FIFO ticket queue (src/models/runtime/scheduler.py)    │
                    └──────────────┬──────────────────────────────┘
                                   │ one VoiceAgent turn graph
                    ┌──────────────▼──────────────────────────────┐
                    │      src/agent/voice_agent/agent.py: VoiceAgent                │
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
