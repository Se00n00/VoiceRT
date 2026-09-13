# Architecture

Complete technical architecture of `voice-pipeline`: every component, every
request path, every cross-module edge. Nothing here is aspirational — each
claim cites the file that implements it.

## 1. Component map

```
voice-pipeline/
├── serve.py                 # entrypoint: builds sys.path, runs uvicorn(:8003)
├── requirements.txt         # pinned pip deps (torch cu124, triton, kokoro…)
├── configs/                 # pipeline.yaml, vad/whisper/qwen/tts.yaml
├── server/
│   ├── app.py               # create_app(): router wiring + startup hook
│   ├── routes.py            # all HTTP endpoints + admission + health/metrics
│   ├── schemas.py           # pydantic validation (ChatReq/SpeakReq/…Resp)
│   ├── websocket.py         # WS /v1/talk (shares the HTTP engine singleton)
│   └── middleware.py        # CORS/logging middleware
├── engine/
│   ├── engine.py            # VoiceEngine: the whole voice loop in one class
│   ├── model.py             # leg registry: YAML → module/class → instance
│   ├── session.py           # SessionStore: RAM-only multi-turn memory
│   ├── streaming.py         # SentenceSplitter + SSE framing (format_sse…)
│   ├── audio.py             # wav io / resample / normalize helpers
│   └── __init__.py          # public re-exports only (no logic)
├── models/
│   ├── qwen.py              # QwenEngine: weights + KVCache + greedy decode
│   ├── whisper.py           # WhisperEngine: mel + encoder + decoder
│   ├── tts.py               # KokoroEngine: voices + text + synth + post
│   └── silero_vad/          # SileroVAD: ONNX session + energy fallback
├── triton_kernels/
│   ├── qwen.py              # rmsnorm/rope(+batched)/swiglu/gqa/fused_qkv_gqa
│   ├── whisper.py           # layernorm/row_softmax/decode/batched/fused_qkv
│   ├── tts.py               # conv1d_silu/in1d_silu/resample/postprocess
│   ├── attention.py         # raw Triton decode/GQA/fused-QKV kernels
│   ├── rope.py · rmsnorm.py · layernorm.py · softmax.py · conv1d.py
│   ├── activation.py        # swiglu/silu/lstm_cell
│   └── utils.py             # grid/block/dtype helpers (pure python)
├── runtime/
│   ├── device.py            # select_device/synchronize/VRAM counters
│   ├── memory.py            # summary/fits/check_budget/MemoryBudgetExceeded
│   ├── profiler.py          # Profiler: named latency samples + p50 report
│   ├── tensor.py            # dtype/device moves + to_host_numpy boundary
│   ├── capacity.py          # probe_vram/estimate/plan_capacity (startup)
│   ├── scheduler.py         # FIFOScheduler: fair admission queue
│   └── __init__.py          # re-exports (every name has a caller or test)
├── scripts/                 # download/validate/benchmark/convert entry points
├── benchmarks/              # per-leg + pipeline + capacity sweeps + results/
├── tests/                   # 54-test unittest suite (mirrors the layout)
└── examples/                # offline.py / streaming.py (VoiceEngine clients)
```

### Dependency rules (enforced by construction, verified in CI smoke)

- `runtime/` imports **nothing** from `engine/`, `server/`, `models/`,
  `triton_kernels/` — it is the leaf layer (`runtime/memory.py:4` only
  reaches back into `runtime/device.py`).
- `triton_kernels/*.py` (per-leg surfaces) import only raw-kernel
  siblings + torch — never `models/`.
- `models/*.py` import `triton_kernels.<leg>` + `runtime.tensor`
  (host-boundary helper) — never `engine/` or `server/`.
- `engine/` imports `runtime.*` + lazily `models.*` via the registry
  (`engine/model.py:86-99`); no static `from models…` at module top, so
  `import engine.engine` works without GPU/weights (`engine/model.py:118-132`
  omit-and-report keeps missing legs as `None` + `self.missing`).
- `server/` imports `engine.*`, `runtime.*`, `server.schemas` — never
  `models.*` at top level (health flags import lazily inside `health()`,
  `server/routes.py:218-221`, keeping `/health` liveness-safe).

## 2. Startup sequence

`serve.py` → `uvicorn.run(create_app())` → FastAPI `startup` event
(`server/app.py:25-60`):

```
1. get_engine() → VoiceEngine()                       (~2 min first boot)
   ├─ load_config(vad/stt/llm/tts) from configs/*.yaml
   ├─ resolve_leg_class() per leg (engine/model.py CANDIDATES)
   ├─ load weights (HF snapshot, safetensors) per leg
   ├─ TTS warmup synth ("warmup.") + cuda.synchronize
   └─ print "voice engine ready vram=1807MB [missing=…]"
2. probe_vram() → {total 4096MB, free, name}          (nvidia-smi → torch → zeros)
3. plan_capacity(total, baseline=max_allocated_mb(), genlen=48, …)
   → {max_sessions 9, max_inflight 4, per_session 206MB, …}
4. configure_capacity(plan)                           (server/routes.py)
   ├─ _sched.max_concurrency = 4
   ├─ _QUEUE_TIMEOUT_S = 10
   └─ engine.sessions.max_sessions = 9
5. print "capacity: 9 sessions @ genlen 48, 4 inflight (…)"
```

Any step-2..4 failure prints `capacity planning skipped (…)` and serving
continues on conservative defaults (`max_sessions 1000` uncapped store,
4 inflight, 10 s queue) — startup **never** fails for observability.

## 3. Request flows

### 3.1 `POST /v1/voice` — full turn (the flagship path)

`server/routes.py:voice` (~line 401):

```
multipart f=wav + session_id ──▶ _parse_upload (400 if unreadable)
 ──▶ _guard_wav (400 empty / 413 >60s)
 ──▶ ticket = _acquire_or_503()            # FIFO wait ≤10s else 503+Retry-After
 ──▶ asyncio.to_thread(eng.stream_turn)    # off the event loop (blocking CUDA)
        │ engine/engine.py:stream_turn
        ├─ check_budget(150MB, 3800MB)      # 503 past the hard guard
        ├─ transcribe()                     # mel frontend → WhisperEngine
        │    → 64-token greedy decode, KV-cached cross-attn
        ├─ prompt_ids()                     # system + session history + user
        ├─ llm.generate_stream()            # prefill (rope_batched, 1 launch)
        │    per decode step: fused_qkv_gqa → rope → gqa_decode_attn
        │    per completed sentence ──▶ speak() → Kokoro → postprocess()
        │    TTFA stamped at first synthesized chunk
        └─ remember(session) → wav concat → {text,reply,wav,ttfa,total,vram}
 ──▶ _release(ticket) ──▶ {text, reply, wav_b64, ttfa_s, total_s, vram_mb, session_id}
```

Measured: TTFA 308 ms, E2E 827 ms, VRAM 1957 MB (synthetic speech;
LibriSpeech 531–858 / 1321–2590 ms).

### 3.2 `POST /v1/chat` (+SSE) — text leg

Non-stream: `eng.chat()` → `generate()` → `{"text","ttft_s","tps","session_id"}`.
Stream: `prompt_ids()` + `llm.generate_stream()` token loop, framed by
`engine/streaming.py:format_sse` / `done_frame` (`server/routes.py:sse`),
`[DONE]` terminator, history remembered after the stream. Stop ids
`151645/151643` never leak into output (regression-tested,
`tests/models/test_qwen.py:test_eos_stops`).

### 3.3 `POST /v1/transcribe` / `/v1/vad` / `/v1/speak` — single legs

Same admission wrapper (parse → guard → ticket → `to_thread` → release →
`_record`). VAD runs ONNX-on-CPU (`models/silero_vad/model.py:prob`,
32 ms windows); STT returns `{text, rtf, ttfs, dur}`; TTS returns raw
`audio/wav` bytes with `X-Synth-S` header.

### 3.4 `WS /v1/talk` — incremental streaming session

`server/websocket.py` shares the HTTP engine singleton via `get_engine()`
(a second `VoiceEngine` would OOM the 4 GB card) and the same FIFO ticket
queue. One socket = one session, established at connect:

```
connect ?session_id= ──▶ {"event":"ready","session_id","sr":16000}
binary PCM16 mono chunks ──▶ buffered (cap 60s) + per-chunk VAD
                             ──▶ {"event":"vad","speech","buffer_s"} on change
                             ──▶ {"event":"stt","text","partial":true}
                                 live partials (throttled buffer re-decode,
                                 best-effort, text-change gated)
speech + 0.8s trailing silence ──▶ auto-commit (or {"type":"commit"})
 ──▶ ticket ──▶ eng.talk_turn(buffer, on_event=emit)   [worker thread]
        ├─ STT done       ──▶ {"event":"stt","text"}            immediately
        ├─ per LLM token  ──▶ {"event":"llm","token"}           immediately
        ├─ LLM done       ──▶ {"event":"llm","done":true,"text"}
        ├─ per sentence   ──▶ {"event":"tts","wav_b64","sr","sentence"}
        └─ turn done      ──▶ {"event":"turn","text","reply","ttfa_s",
                               "total_s","vram_mb","session_id"}
{"type":"reset"} drops the buffer; {"type":"config","sr","end_silence_s"}
retunes the session; "close" ends it. A RIFF/wav binary message takes the
legacy one-shot path (single summary JSON, unchanged).
```

Event bridging: `talk_turn` invokes `on_event` synchronously in the worker
thread; the handler re-posts each event to an `asyncio.Queue` via
`loop.call_soon_threadsafe` and a drain loop forwards them to the socket
in stage order — the client sees STT text ~60 ms after commit while the
LLM is still decoding.

### 3.4b `POST /v1/talk` — same staged turn over plain HTTP (SSE)

No WebSocket client? `POST /v1/talk` (multipart wav `f` + optional
`session_id` form field, `server/routes.py:talk_stream`) runs the identical
`talk_turn(..., on_event=...)` path and streams it back as
`text/event-stream`: `event: stt → event: llm (per token) → event: llm done
→ event: tts (per sentence, wav_b64) → event: turn → [DONE]`, with
`X-Session-Id` echoed. Same FIFO ticket, same budget guard, same 400/413/
503 semantics as `/v1/voice` — consume with `curl -N`.

### 3.5 `GET /health` · `GET /metrics` · `GET /v1/sessions`

- `/health`: liveness-safe (no engine construction). Publishes `capacity`
  plan + `triton` flags + `missing` legs + `vram_mb`.
- `/metrics`: per-endpoint `{count, errors, mean_s}` (incl. `queue_wait`,
  `queue_timeout`, `chat_stream`), `inflight` (= scheduler `running`),
  `queue` = `FIFOScheduler.stats()` (`max_concurrency/pending/running/
  admitted_total`), `vram_mb`.
- `/v1/sessions`: `{sessions, turns, max_sessions, generation_length}`.

## 4. Concurrency & admission model

```
requests ──▶ FIFOScheduler(max_concurrency=4) ──▶ engine (serial-ish CUDA)
                │ blocking acquire, timeout=10s
                ├─ admitted → turn runs, ticket released in finally
                └─ timeout  → 503 + Retry-After: 2  (recorded: queue_timeout)
```

Why FIFO, not a semaphore: arrival-order tickets prevent Grabby-client
starvation under burst; `pending/running/admitted_total` make the queue
observable. Why 4: measured serialization — conc-4 halves per-request
throughput (`benchmarks/benchmark_pipeline.py` sweep). Queue waits are
timed into `queue_wait` so p99-latency math includes admission, honestly.

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
weights resident (measured baseline)     ~1807 MB
├─ Qwen2.5-0.5B bf16                      ~1000 MB
├─ Whisper-base                            ~300 MB
├─ Kokoro-82M                              ~330 MB
└─ CUDA context / fragmentation            ~180 MB
per-turn transient (guarded @150 MB)      ≤ 150 MB
hard ceiling (vram_budget_mb)              3800 MB  → 503 past it
steady-state peak (measured)               1957 MB
```

## 7. Failure modes (all tested live: `tests/server/test_routes.py`)

| Input | Code | Where |
|---|---|---|
| unreadable upload | 400 | `_parse_upload` |
| empty audio | 400 | `_guard_wav` |
| >60 s audio | 413 | `_guard_wav` |
| prompt/tokens/text bounds | 422 | `server/schemas.py` |
| weights missing | 503 | `_leg_error` (omit-and-report) |
| queue timeout (burst) | 503 + Retry-After | `_acquire_or_503` |
| VRAM guard tripped | 503 + Retry-After | `check_budget` → `_leg_error` |

## 8. What was deliberately removed

Dead surface deleted 2026-09-12 (zero callers in prod or tests):
`engine/executor.py`, `engine/generation.py`, `engine/pipeline.py`
(alternate engines nobody ran), `runtime/{stream,graph,event,allocator,
batcher,request}.py` (unwired utilities), `runtime/memory.py:MemoryBudget`
+ unused counters, `triton_kernels/{embedding,matmul,reductions}.py`
(orphan kernels). Deliberately **kept**: eager TTS (measured 8x faster
than dynamic compile), ONNX VAD (RTF 0.01), cuDNN conv (0.00x Triton
speedup) — each with the measurement cited, not just asserted.
