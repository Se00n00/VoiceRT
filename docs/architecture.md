# Architecture

Complete technical architecture of `voice-pipeline`: every component, every
request path, every cross-module edge. Nothing here is aspirational — each
claim cites the file that implements it.

## 1. Component map

```
voice-pipeline/
├── server.py                # THE server: health, metrics, two-way /talk WS (:8003)
├── engine/
│   ├── session.py           # SessionStore: RAM-only multi-turn memory
│   ├── streaming.py         # SentenceSplitter + SSE framing (format_sse…)
│   ├── audio.py             # wav io / resample / normalize helpers
│   └── __init__.py          # public re-exports only (no logic)
├── requirements.txt         # pinned pip deps (torch cu124, triton, kokoro…)
├── src/
│   ├── models/              # clean async legs (dataclass configs, no YAML)
│   │   ├── vad.py / stt.py / llm.py / tts.py
│   │   ├── runtime/         # device/memory/profiler/tensor + capacity/scheduler
│   │   └── triton_kernels/  # hand-written kernels + per-leg surfaces
│   ├── agent/               # LangGraph turn graph + node events
│   └── main.py              # VoiceAgent: VAD -> STT -> LLM -> TTS loop
├── models/
│   ├── qwen.py              # QwenEngine: weights + KVCache + greedy decode
│   ├── whisper.py           # WhisperEngine: mel + encoder + decoder
│   ├── tts.py               # KokoroEngine: voices + text + synth + post
│   └── silero_vad/          # SileroVAD: pure-ONNX session
├── tests/                   # 83-test unittest suite (mirrors the layout)
└── README.md
```

(`runtime/` lives at `src/models/runtime/`, `triton_kernels/` at
`src/models/triton_kernels/` — same files, see the `src/` block above.)

### Dependency rules (enforced by construction, verified in CI smoke)

- `src/models/runtime/` imports **nothing** from `src/engine/`, `server/`,
  `models/`, `src/models/triton_kernels/` — it is the leaf layer
  (`src/models/runtime/memory.py` only reaches back into
  `src/models/runtime/device.py`).
- `src/models/triton_kernels/*.py` (per-leg surfaces) import only raw-kernel
  siblings + torch — never `models/`.
- `models/*.py` import `src.models.triton_kernels.<leg>` + `src.models.runtime.tensor`
  (host-boundary helper) — never `engine/` or `server.py`.
- `engine/` (audio/session/streaming helpers) imports
  `src.models.runtime.*` for tensor moves — never `models/` legs.
- `src/models/*.py` (clean facades) import their `models.*` leg lazily
  inside `_backend()` only, so `import src.models.vad` never loads
  weights.
- `server.py` (one module, not a package) imports `src.main`, `engine.*`,
  `src.models.runtime.*` — never `models.*` at top level, keeping
  `/health` liveness-safe.

## 2. Startup sequence

`python server.py` → `uvicorn.run(app)` → FastAPI `startup` event
(`server.py:_warm`):

```
1. VoiceAgent() → await agent.warm()              (~2 min first boot)
   ├─ VadModel/SttModel/LlmModel/TtsModel warm, one leg at a time
   ├─ per-leg failure → self.missing (omit-and-report, never fatal)
   └─ print "voice-agent ready" [+ "voice-agent missing: …"]
```

Startup **never** fails for observability: `/health` answers with
`agent_loaded: false` + `missing` until legs exist.

## 3. Request flows

### 3.1 `WS /talk` — two-way socket (the only turn path)

One socket = one session, established at connect:

```
connect ?session_id= ──▶ {"event":"ready","session_id","sr":16000,"nodes"}
binary PCM16 mono chunks (or {"type":"audio","pcm_b64","sr"})
  ──▶ buffered (cap 60s) + per-chunk VAD
  ──▶ {"event":"node","node":"vad","kind":"speech",...} on change
speech + end_silence_s trailing silence ──▶ auto-commit
  (or {"type":"commit"}; {"type":"reset"} drops buffer+VAD state)
  ──▶ FIFO ticket (≤10s else turn `error` event)
  ──▶ async for event in agent(buffer, 16000, sid):   [worker threads]
         ├─ VAD gate: silence ──▶ stt-empty + done summary, no burn
         ├─ STT done       ──▶ {"event":"node","node":"stt",...}  immediately
         ├─ per LLM token  ──▶ {"event":"node","node":"llm",...}  immediately
         ├─ per sentence   ──▶ {"event":"node","node":"tts",      immediately
         │                      "data":{"wav_b64","sr","sentence"}}
         └─ turn done      ──▶ {"event":"done","summary":
                                {"text","reply","node_s","ttfa_s",
                                 "total_s","vram_mb","session_id"}}
budget/guard failures ──▶ {"event":"error","message"} (socket stays open)
```

One `VoiceAgent` serves every socket (a second agent would duplicate
weights in VRAM and OOM the 4 GB card). LLM→TTS overlap: each completed
sentence is synthesized while the next tokens still decode, so TTFA lands
without waiting for the full reply.

Measured (old stack, same legs): TTFA 308 ms, E2E 827 ms, VRAM 1957 MB
(synthetic speech; LibriSpeech 531–858 / 1321–2590 ms).

### 3.2 `GET /health` · `GET /metrics`

- `/health`: liveness-safe (never builds the agent). `{ok, uptime_s,
  agent_loaded, missing, nodes, sessions, vram_mb}`.
- `/metrics`: `{uptime_s, turns: {started, done, errors, mean_s},
  events: {vad, stt, llm, tts, turn}}`.

## 4. Concurrency & admission model

```
turns ──▶ FIFOScheduler(max_concurrency=4) ──▶ VoiceAgent (serial-ish CUDA)
                 │ blocking acquire, timeout=10s
                 ├─ admitted → turn runs, ticket released in finally
                 └─ timeout  → turn `error: server saturated` (counted)
```

Why FIFO, not a semaphore: arrival-order tickets prevent Grabby-client
starvation under burst. Why 4: measured serialization — conc-4 halves
per-request throughput (old `benchmark_pipeline.py` sweep, since removed
with the `/v1/*` API it drove).

Session memory is orthogonal: `SessionStore` (20 msgs, 30-min TTL, cap =
plan `max_sessions`, LRU-evicted) only stores text — sessions never hold
GPU resources, so stored sessions ≫ in-flight turns by design.

## 5. Latency budget (where the 827 ms goes)

Typical 48-token turn, conc 1 (measured):

```
VAD ~50ms  +  STT ~324ms  +  LLM prefill/TTFT ~14ms + decode ~660ms¹
+  TTS first-sentence ~150ms (overlaps decode tail)  ≈  TTFA ~308ms
total ≈ 827ms (queue_wait ≈ 0 unloaded; +queue under burst)
```

¹ 48 tokens × ~14 ms TPO. Longer `max_tokens` stretches decode linearly —
which is exactly why generation length prices sessions (docs/capacity.md).

## 6. VRAM budget (where the 1.9 GB goes)

```
weights resident (measured baseline, Qwen3-0.6B stack)  ~2290 MB
├─ Qwen3-0.6B bf16                       ~1480 MB
├─ Whisper-base                            ~300 MB
├─ Kokoro-82M                              ~330 MB
└─ CUDA context / fragmentation            ~180 MB
per-turn transient (guarded @150 MB)      ≤ 150 MB
hard ceiling (vram_budget_mb)              3800 MB  → turn `error` past it
steady-state peak (measured)               2290 MB
```

## 7. Failure modes (all surfaced as turn `error` events; socket stays open)

| Input | Event | Where |
|---|---|---|
| empty buffer on commit | `error: empty buffer` | `/talk` handler |
| >60 s buffered audio | `error: buffer exceeds 60s` | `/talk` handler |
| empty audio array | `error: empty audio` | `VoiceAgent.__call__` guard |
| >60 s turn audio | `error: audio … exceeds … cap` | `VoiceAgent.__call__` guard |
| weights missing | `error` | leg warm (omit-and-report → `missing`) |
| queue timeout (burst) | `error: server saturated` | FIFO ticket, 10 s |
| VRAM guard tripped | `error` | `check_budget` |

## 8. What was deliberately removed

Dead surface deleted 2026-09-12 (zero callers in prod or tests):
`engine/executor.py`, `engine/generation.py`, `engine/pipeline.py`
(alternate engines nobody ran), `runtime/{stream,graph,event,allocator,
batcher,request}.py` (unwired utilities), `runtime/memory.py:MemoryBudget`
+ unused counters, `triton_kernels/{embedding,matmul,reductions}.py`
(orphan kernels). Deliberately **kept**: eager TTS (measured 8x faster
than dynamic compile), ONNX VAD (RTF 0.01), cuDNN conv (0.00x Triton
speedup) — each with the measurement cited, not just asserted.
