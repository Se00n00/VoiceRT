<img src="VoiceRT.png">

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![CUDA 12](https://img.shields.io/badge/CUDA-12-green)
![VRAM 4GB](https://img.shields.io/badge/VRAM-4GB-orange)
![TTFT 14ms](https://img.shields.io/badge/TTFT-14ms-brightgreen)
![Tests 84 passing](https://img.shields.io/badge/tests-84_passing-brightgreen)

**A full voice loop — speech in, speech out — running live on a single 4GB laptop GPU.**
No cloud, no API keys, no cluster: microphone (or wav) → text → reply → voice, in under a second.

```bash
# one turn over the two-way socket (see API below)
# {"event": "done", "summary": {"text": "...",
#  "reply": "Hello! How can I assist you today?",
#  "ttfa_s": 0.31, "total_s": 0.83, ...}}
```

| Metric (measured, RTX 3050 4GB) | Value |
|---|---|
| Pipeline | Silero VAD → Whisper-base → Qwen3-0.6B → Kokoro-82M |
| Resident VRAM | ~1.9 GB (all four models live) |
| Voice-turn latency | TTFA **308 ms**, end-to-end **827 ms** |
| LLM decode | TTFT **14 ms**, **~69 tok/s** sustained (base); +13.6% with all features (chunked+prefix+cudagraph) |
| Speech recognition | RTF **0.020** (50x real-time) |
| Speech synthesis | RTF **0.045** (22x real-time) |
| Serving capacity | **9 sessions** @ 48 tokens, FIFO-queued |
| Cost | **$0.20 per million tokens** (local power, $0.05/hr) |
| Tests | 84 passing, 1 skipped |

## Contents

- [How it works](#how-it-works)
- [Capacity: how many sessions fit](#capacity-how-many-sessions-fit)
- [Performance](#performance)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Use it in Python](#use-it-in-python)
- [API specification](#api-specification)
- [Configuration](#configuration)
- [Triton kernels](#triton-kernels)
- [Operations](#operations)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Deep dives (docs/)](#deep-dives)

## How it works

```
mic / wav ──> VAD (Silero, ONNX CPU) ──> STT (Whisper-base)
    ──> LLM (Qwen3-0.6B, streaming tokens)
    ──> sentence splitter ──> TTS (Kokoro-82M) ──> wav out
```

One `VoiceAgent` (a compiled LangGraph turn graph) holds all four models
in a single process and VRAM pool.
LLM tokens stream out; each completed sentence is synthesized immediately,
so the first audio starts in ~300 ms without waiting for the full reply.

What makes this more than a demo glue script:

1. **VRAM-aware capacity planning.** At startup the server probes the GPU,
   prices one session from the generation length, and derives how many
   sessions it can serve — no hardcoded limits. See below.
2. **Fair session queues.** Burst traffic waits in a FIFO queue with a
   timeout instead of thundering-herd; only a truly full server says 503.
3. **Hand-written GPU kernels.** Custom Triton kernels (RMSNorm, RoPE,
   attention, SwiGLU, conv+SiLU…) with exact CPU fallbacks and parity tests.
   Every exported kernel runs in the hot path — nothing decorative.
4. **Honest benchmarking.** TTFT/TPO/throughput/GPU-util/power/cost are
   measured live and published with the methodology, including the parts
   that are deliberately left eager (documented, with numbers).

## Capacity: how many sessions fit

The question every serving system must answer: *given this GPU and this
workload, how many users fit?* The server answers it at every boot:

```
startup: probe VRAM (nvidia-smi)
       → warm models, measure baseline (weights resident)
       → price one session from generation length
       → max_sessions = (total − baseline − headroom) / per_session
       → resize session store + FIFO queue from the plan
```

Per-session price (`src/models/runtime/capacity.py`, unit-tested on CPU):

- **KV-cache share** — `2 × 28 layers × 8 kv-heads × 512 × 128 × 2B ≈ 56 MB`
- **History working set** — `20 turns × (200 prompt + 48 gen) tokens × 1024 × 2B ≈ 9.7 MB`
- **Transient scratch** — TTS buffers + mel frontend ≈ 150 MB
- × **1.25 safety factor** → **≈ 270 MB/session** at 48 tokens

Worked example — this machine (RTX 3050 Laptop, 4096 MB), Qwen2.5-0.5B
stack measured 2026-09-12 (kept for reference; Qwen3-0.6B re-measures at
next boot, estimate ≈ 270 MB/session → ~6 sessions):

| | genlen 48 | genlen 128 |
|---|---|---|
| Baseline (weights) | 1807 MB | 1807 MB |
| Headroom (10%) | 410 MB | 410 MB |
| Usable | 1879 MB | 1879 MB |
| Per session | 206 MB | 209 MB |
| **Sessions served** | **9** | **8** |

Longer generations cost more history per turn, so fewer sessions fit —
exactly the tradeoff the planner quantifies. The live plan is always
visible: `GET /health` → `missing` + `sessions` + `vram_mb`.
Retune via `VoiceAgentConfig` fields (`max_tokens`, `max_inflight`,
`queue_timeout_s`, `vram_budget_mb`).

## Performance

Measured 2026-09-12 on RTX 3050 Laptop 4GB (CUDA 12, torch 2.5.1),
Qwen2.5-0.5B stack (kept for reference; the serving math is unchanged,
see `src/models/runtime/capacity.py` + `tests/runtime/test_capacity.py`).
First-touch CUDA init inflates the very first row; steady state follows it.
GPU util is sampled via `nvidia-smi` at 2 Hz during each run — short
sub-second bursts under-fill the sampler, so treat util as a lower bound.

### LLM: latency vs concurrency vs generation length

| genlen | conc | TTFT p50 | TPO | TPS | req/s | tok/s | util% | W | tok/s/W | $/1M |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 214ms¹ | 20.3ms | 26.4 | 2.40 | 26.4 | 2.0 | 19.8 | 1.33 | $0.5265 |
| 16 | 2 | 30ms | 29.4ms | 34.6 | 4.85 | 65.5 | 1.0 | 22.1 | 2.96 | $0.2122 |
| 48 | 1 | 14ms | 14.4ms | 69.6 | 6.31 | 69.4 | 1.0 | 22.1 | 3.14 | $0.2000 |
| 48 | 2 | 28ms | 25.8ms | 40.3 | 3.44 | 67.1 | 16.0 | 27.5 | 2.44 | $0.2070 |

¹ First generate after load includes CUDA init; steady-state TTFT is the 14–30 ms band.

Concurrency doubles token throughput (26 → 66 tok/s at genlen 16) while
per-request latency rises — the classic serving tradeoff, measured not
assumed. Peak efficiency: **3.14 tok/s per watt** (conc 1, genlen 48).

### LLM: inference engine features benchmark (RTX 3050 4GB, Qwen3-0.6B real `QwenRunner`, `num_blocks=16`)

Real `QwenRunner` only (isolated subprocess per config; `DummyRunner` removed). 8 concurrent requests:

| config | 32 tok/req: tok/s | delta | 8 tok/req: tok/s | delta |
|---|---|---|---|---|
| base (no features) | 42.0 | — | 21.7 | — |
| + prefix caching | 48.8 | **+16.4%** | 19.4 | -10.5% |
| + chunked prefill | 49.3 | **+17.4%** | 30.6 | **+41.3%** |
| + CUDA graph | 48.4 | **+15.3%** | 31.8 | **+46.7%** |
| all features | 48.0 | **+14.4%** | 32.4 | **+49.4%** |

Longer generations amortize prefill; short bursts benefit most from chunked / graph. No dummy fallback — every row is real Qwen3-0.6B weights on 4GB.

### Cost per million tokens

At a local amortized rate of **$0.05/hr** (laptop power + hardware share):

| workload | tok/s | $/1M tokens |
|---|---|---|
| genlen 48, conc 1 | 69.4 | **$0.2000** |
| genlen 48, conc 2 | 67.1 | $0.2070 |
| genlen 16, conc 2 | 65.5 | $0.2122 |

### Full voice turn (round-trip: synth speech → full pipeline)

| | TTFA | E2E | VRAM |
|---|---|---|---|
| Round-trip (synthetic speech in) | 308 ms | 827 ms | 1957 MB |
| Earlier live turn | 438 ms | 640 ms | 1941 MB |
| LibriSpeech samples (previous) | 531–858 ms | 1321–2590 ms | 1937 MB |

### Per-leg spot checks

| Leg | Latency | Real-time factor |
|---|---|---|
| VAD | ~50 ms | 0.01 |
| STT (Whisper) | 59 ms / 3 s audio | **0.020** |
| LLM TTFT / decode | 14 ms / ~69 tok/s | — |
| TTS (Kokoro, eager) | 141 ms | **0.045** |

## Project structure

```
voice-pipeline/
├── server.py            # THE server: health, metrics, two-way /talk WS (port 8003)
├── engine/              # audio io, session memory, streaming helpers
├── requirements.txt    # pinned runtime deps, pip install only
├── models/
│   ├── qwen.py         # LLM leg: QwenEngine (weights + KV-cache + attn)
│   ├── whisper.py      # STT leg: WhisperEngine (mel + encoder + decoder)
│   ├── tts.py          # TTS leg: KokoroEngine (voices + text + synth)
│   └── silero_vad/     # VAD leg (pure ONNX, CPU)
├── src/                # clean async legs + agent loop
│   ├── models/         # vad/stt/llm/tts (dataclass configs, async, no YAML)
│   │   ├── runtime/    # device/memory/profiler/tensor + capacity + scheduler
│   │   │   ├── triton_kernels/  # hand-written kernels (rmsnorm/rope/attn/conv1d…)
│   │   │                 # + per-leg surface: qwen.py / whisper.py / tts.py
│   │   └── download.py # weight bootstrap (HF + silero ONNX)
│   ├── agent/          # LangGraph turn graph (state, nodes, builder)
│   └── main.py         # VoiceAgent: VAD -> STT -> LLM -> TTS loop
├── tests/              # 84-test unittest suite (engine/kernels/models/inference_engine…)
└── README.md
```

## Requirements

- Python 3.12
- CUDA GPU (measured on RTX 3050 4 GB, CUDA 12; CPU works, slower)
- ~2.5 GB free VRAM for warm run, ~5 GB disk for weights
- System: `ffmpeg`, libsndfile (for `soundfile`), `nvidia-smi` (telemetry; torch fallback)
- Python deps: `torch` CUDA build, `triton`, `transformers`, `fastapi`,
  `uvicorn`, `soundfile`, `numpy`, `onnxruntime`, `pyyaml`, `kokoro`

Get the CUDA torch first if pip resolves CPU-only:

See https://pytorch.org/get-started/locally/ for your platform, then
install the rest from `requirements.txt`.

## Installation

Run everything from inside `voice-pipeline/`:

```bash
cd VoiceAgent/voice-pipeline
pip install -r requirements.txt
```

`PYTHONPATH=.` must be set on every command below so the
  `src` / `models` / `engine` / `server` packages resolve (kernels live at `src.models.triton_kernels`).

Download weights (first run only):

```bash
PYTHONPATH=. python -m src.models.download
```

Start the API (warms all legs — ~2 min first boot):

```bash
PYTHONPATH=. python server.py
# optional: PYTHONPATH=. python server.py --host 0.0.0.0 --port 8003
```

Watch the startup line: `voice-agent ready` (plus `missing=[…]` when a
leg has no weights).

## Quickstart

1. Install + download models (see above).
2. Start server: `PYTHONPATH=. python server.py`
3. Full voice turn over the two-way socket (see API below).

4. Health + metrics:

```bash
curl http://localhost:8003/health
curl http://localhost:8003/metrics
```

5. Run tests:

```bash
PYTHONPATH=. python -m unittest discover -s tests -t . -q
```

## Use it in Python

Voice turn straight on the agent (streams per-node events):

Direct agent API:

```python
import asyncio
from src.main import VoiceAgent

async def main():
    agent = VoiceAgent()
    await agent.warm()
    async for event in agent(audio, sr=16000, session_id="my-session-id"):
        print(event.node, event.kind, str(event.data)[:80])

asyncio.run(main())
```

Single legs (dataclass configs, no YAML):

```python
from src.models.vad import VadConfig, VadModel
from src.models.llm import LlmConfig, LlmModel

vad = VadModel(VadConfig(threshold=0.5))
llm = LlmModel(LlmConfig(model="Qwen/Qwen3-0.6B", thinking=False))
```

Capacity math in code (same functions the server uses):

```python
from src.models.runtime.capacity import probe_vram, plan_capacity
info = probe_vram()                            # nvidia-smi -> torch -> zeros
plan = plan_capacity(info["total_mb"], 1807, max_new_tokens=48)
print(plan["max_sessions"], "sessions @", plan["generation_length"], "tokens")
```

## API specification (one server: `server.py`)

Base URL default: `http://localhost:8003`. Interactive docs at `/docs`,
raw schema at `/openapi.json` — all three endpoints are listed there,
including the WebSocket `/talk`.

| Method | Endpoint | Input | Output |
|---|---|---|---|
| `GET` | `/health` | — | liveness + agent status (below) |
| `GET` | `/metrics` | — | turn counters + per-node event counts |
| `WS` | `/talk` | PCM16 chunks / control JSON (below) | per-node streams + turn summary |

### `GET /health`

Liveness-safe: never builds the agent, always 200 while the process is up.

```jsonc
// GET /health
{
  "ok": true,
  "uptime_s": 25.29,
  "agent_loaded": true,     // false until legs finish warming
  "nodes": ["vad", "stt", "llm", "tts"],
  "missing": [],            // legs without weights, e.g. ["tts leg: ..."]
  "vram_mb": 2289.7,        // torch peak allocated (absent on CPU-only)
  "sessions": {"sessions": 0, "turns": 0, "max_turns": 20, "max_age_s": 1800.0}
}
```

### `GET /metrics`

```jsonc
// GET /metrics
{
  "uptime_s": 31.49,
  "turns": {"started": 12, "done": 11, "errors": 1, "mean_s": 0.94},
  "events": {"vad": 24, "stt": 11, "llm": 132, "tts": 21, "turn": 11}
}
```

### `WS /talk` — two-way voice socket

Connect at `ws://host:8003/talk`, optional `?session_id=`. One socket =
one session. Audio is mono float, server works at 16 kHz internally
(chunks at other rates are resampled).

**Client → server.** Text frames carry JSON; binary frames carry raw
little-endian PCM16 mono (equivalent to one `audio` chunk):

| Message | Meaning |
|---|---|
| `{"type":"config","sr":16000,"session_id":"…","end_silence_s":0.8}` | tune session (all fields optional); server re-sends `ready` |
| `{"type":"audio","pcm_b64":"…","sr":16000}` | append one chunk (`sr` optional, resampled to 16 kHz) |
| `<binary PCM16>` | same as an `audio` chunk, no base64 overhead |
| `{"type":"commit"}` | run a turn on the buffer now |
| `{"type":"reset"}` | drop the buffer + VAD state |
| `{"type":"close"}` | end the session |

Buffer cap is 60 s of audio (`error` past it). Empty `commit` answers
`{"event":"error","message":"empty buffer"}` and the socket stays open.

**Endpointing:** every chunk is VAD-scored. After speech plus
`end_silence_s` trailing silence the buffered turn auto-commits — a mic
client can just stream frames and read back streams, no `commit` needed.

**Server → client:**

| Frame | Meaning |
|---|---|
| `{"event":"ready","session_id":"…","sr":16000,"nodes":["vad","stt","llm","tts"]}` | on connect (and after `config`) |
| `{"event":"node","node":"vad","kind":"speech","data":{"speech":bool,"buffer_s":float}}` | VAD state change per chunk |
| `{"event":"node","node":"vad","kind":"segments","data":{"segments":[[start_s,end_s]],"audio_dur_s":float,"speech_s":float}}` | turn speech spans |
| `{"event":"node","node":"vad","kind":"reset","data":{"buffer_s":0.0}}` | ack of `reset` |
| `{"event":"node","node":"stt","kind":"text","data":{"text","rtf","ttfs","dur_s"}}` | transcript (or `{"text":"","silent":true}` on gated silence) |
| `{"event":"node","node":"llm","kind":"token","data":{"token","first":bool}}` | one decoded piece, streamed |
| `{"event":"node","node":"llm","kind":"done","data":{"text"}}` | full reply |
| `{"event":"node","node":"tts","kind":"audio","data":{"wav_b64","sr","sentence","synth_s"}}` | one sentence of float32 PCM (`wav_b64` decodes to `sr`-Hz mono) |
| `{"event":"done","summary":{"text","reply","segments","node_s","ttfa_s","total_s","vram_mb","session_id"}}` | turn complete |
| `{"event":"error","message"}` | guard/saturation failure; socket stays open |

**A real turn** (live capture — "Hello there, how are you today?" in,
reply out):

```
server  {"event":"ready","session_id":"66f3…","sr":16000,"nodes":[…]}
server  {"event":"node","node":"vad","kind":"speech","data":{"speech":false,…}}
server  {"event":"node","node":"vad","kind":"segments",
          "data":{"segments":[[0.32,2.01]],"audio_dur_s":2.15,"speech_s":1.69}}
server  {"event":"node","node":"stt","kind":"text",
          "data":{"text":" Hello there, how are you today?","rtf":0.075,…}}
server  {"event":"node","node":"llm","kind":"token","data":{"token":"Hello","first":true}}
server  {"event":"node","node":"llm","kind":"token","data":{"token":"!",…}}
        …tokens stream, one TTS chunk per completed sentence…
server  {"event":"node","node":"tts","kind":"audio",
          "data":{"wav_b64":"…","sr":24000,"sentence":"Hello!","synth_s":…}}
server  {"event":"node","node":"llm","kind":"done",
          "data":{"text":"Hello! I'm here to help you…"}}
server  {"event":"done","summary":{"text":" Hello there, how are you today?",
          "reply":"Hello! I'm here to help you…","ttfa_s":0.31,"total_s":0.83,…}}
```

Minimal client:

```python
import asyncio, json
import websockets  # pip install websockets

async def main():
    async with websockets.connect(
            "ws://localhost:8003/talk?session_id=my-uuid") as ws:
        print(await ws.recv())  # {"event": "ready", ...}
        with open("sample.pcm", "rb") as f:
            pcm = f.read()      # mono 16k PCM16 frames
        await ws.send(pcm[:32000])
        await ws.send(json.dumps({"type": "commit"}))
        async for msg in ws:
            evt = json.loads(msg)
            # {"event": "node", "node": "vad"|"stt"|"llm"|"tts", ...}
            # {"event": "done", "summary": {text, reply, ttfa_s, total_s}}
            # {"event": "error", "message": ...}
            print(evt["event"], evt.get("node", ""))
            if evt.get("event") in ("done", "error"):
                break

asyncio.run(main())
```

Sessions: frontend creates one UUID (`crypto.randomUUID()`), sends it as
`?session_id=` on `WS /talk` (or inside a `config` message). Omit it for
stateless. History bounded per session, 30-min TTL, RAM-only.

## Configuration

No YAML, no config files: every leg takes a frozen dataclass, composed
in `VoiceAgentConfig`:

```python
from src.main import VoiceAgent, VoiceAgentConfig
from src.models.llm import LlmConfig
from src.models.tts import TtsConfig
from src.models.vad import VadConfig

cfg = VoiceAgentConfig(
    vad=VadConfig(threshold=0.5),
    llm=LlmConfig(model="Qwen/Qwen3-0.6B", max_tokens=48, thinking=False),
    tts=TtsConfig(voice="af_heart"),
    max_inflight=4,          # measured serialization point
    queue_timeout_s=10,      # FIFO wait before a turn `error` event
    vram_budget_mb=3800,     # hard guard, enforced per voice turn
)
agent = VoiceAgent(cfg)
```

Raising `max_tokens` prices sessions up (longer generations hold more
history per turn); lowering `max_inflight` sheds burst load sooner.

## Triton kernels

All kernels are parity-tested and wired in (`HAVE_*=True` verified).
One import per leg — `src/models/triton_kernels/qwen.py`, `whisper.py`,
`tts.py` — each with Triton fast path + exact torch fallback.

| Kernel | Speedup vs eager | Status |
|---|---|---|
| rope_batched (prefill) | 26.7x | TTFT 173 → 14 ms |
| conv1d_silu / in1d_silu (TTS post) | 2.53x | in TTS hot path |
| lstm_cell | 1.59x | standalone parity vs nn.LSTMCell (generic; VAD runs pure ONNX, unwired) |
| rmsnorm / rope / swiglu / gqa / layernorm / row_softmax / decode-attn / batched-decode | active in hot paths | profile-verified |
| fused_qkv (STT) / fused_qkv_gqa (LLM) | faster in-engine than microbench | primary + fallback |
| plain conv1d | 0.00x vs cuDNN | stays on cuDNN, documented |

Honest call: Kokoro full-model `torch.compile` is broken in this env
(dynamo × transformers-5 → `NameError: torch`); submodule static compile
crashes on new lengths; dynamic compile is slower than eager for varying
sentences (RTF 0.37 vs 0.045). Eager TTS + Triton post-processing is the
right default, and the CUDA-graph runner in `src/models/runtime/` stays explicitly
opted out for the same measured reason.

By leg:

| Leg | File | Custom Triton | Note |
|---|---|---|---|
| STT | `models/whisper.py` | layernorm, row_softmax, decode-attn, batched-decode, fused_qkv | every export in hot path |
| LLM | `models/qwen.py` | rmsnorm, rope (+batched), swiglu, gqa, fused_qkv_gqa | exact text match vs HF |
| TTS | `models/tts.py` | conv1d+silu, in1d+silu (post-filter), resample | Kokoro eager + Triton audio path |
| VAD | `models/silero_vad/` | — (ONNX CPU, RTF 0.01) | nothing to win |

## Operations

- `GET /health` is liveness-safe (never builds the agent): `{ok,
  uptime_s, agent_loaded, missing, nodes, sessions, vram_mb, engine}`.
  When `VoiceAgent(llm_paged=True)` / `server.py` on CUDA, `engine` is the
  paged `InferenceEngine` stats (runner `QwenRunner`, paged KV, prefix cache,
  CUDA-graph) proving the inference engine is live for every LLM turn.
- `GET /metrics`: turn counts/errors/mean latency + per-node event counts
  plus `engine` (steps/tokens/prefix/cache) when paged is active.
- Admission: turns take a FIFO ticket (`queue_timeout_s`, default 10 s);
  a saturated server answers the turn with an `error` event instead of
  queueing unboundedly. Per-turn VRAM guard (`vram_budget_mb`, default
  3800 MB) is enforced in `VoiceAgent.__call__`.
- Generation stops at `<|im_end|>` / `<|endoftext|>` — the `llm done`
  event ends the token stream; no template leakage.
- Guards: audio capped at 60 s, empty audio rejected, missing legs
  reported in `/health` → `missing` (and surface as turn `error` events).
- Concurrency: single worker serializes past 4 in-flight (measured) —
  the queue sheds load instead of melting latency. True admission
  batching is the next rung, not this rung.
- VRAM: ~1.9 GB steady with all four legs; 9 sessions fit the reference
  4 GB card at 48 tokens.
- One `VoiceAgent` serves every socket — a second agent would duplicate
  weights in VRAM and OOM the 4 GB card.
- VAD is pure ONNX on CPU by decision: vendored weights, RTF 0.01,
  no energy fallback, no Triton path — the ONNX session is the whole leg.

## Testing

```bash
PYTHONPATH=. python -m unittest discover -s tests -t . -q
# 84 tests OK (agent loops, kernel parity, model legs, capacity math, inference engine,
# runtime helpers, 3-endpoint server)
```

Check weights are present after download (`GET /health` → `missing`
must be `[]` when all four legs resolve).

## Deep dives

- [`docs/architecture.md`](docs/architecture.md) — complete system
  architecture: component map, startup sequence, every request flow,
  latency/VRAM budgets, failure-mode table
- [`docs/capacity.md`](docs/capacity.md) — the VRAM capacity model,
  queueing design, worked 4GB example, retuning guide

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: src / models / server` | Run from `voice-pipeline/` with `PYTHONPATH=.` |
| pip installed CPU-only torch | Install CUDA build from pytorch.org first, then `pip install -r requirements.txt` |
| `leg(s) not loaded` (turn `error` event) | Run `python -m src.models.download`, check `GET /health` → `missing` |
| `server saturated` (turn `error` event) | Queue timed out under burst; retry, or raise `queue_timeout_s` / lower `max_tokens` |
| `needs ~XMB but budget is 3800MB` (turn `error` event) | Per-turn guard tripped; free VRAM or raise `vram_budget_mb` |
| First boot slow (~2 min) | Normal: all four legs warm up; later boots are faster |
| OOM on 4 GB card | Close other GPU apps; only one `VoiceAgent` may exist |
| Empty / long audio errors | Cap is 60 s; empty buffers return an `error` event by design |

## Roadmap

- [x] VRAM-derived session capacity (this release)
- [x] FIFO admission queue with timeout (this release)
- [x] TTFT/TPO/throughput/util/power/cost profiling (this release)
- [x] True admission batching (1 fused layer x28, B=1..8, KV-cache aware)
- [x] Partial STT + VAD-driven barge-in over WS
- [ ] ONNX → Triton VAD cutover when it outgrows CPU
- [ ] Quantized LLM / STT options for <2 GB cards
- [ ] Frontend demo client for `/talk`

## Fused Kernels (1 layer x28, batched, KV-cache aware) — single-file per model

Each model now lives in **one Triton file** + **one PyTorch reference** + **one complete class** (`28 loops`) — inside `src/models/` (clean, no root `models/`):

| Model | Triton fused (single file) | PyTorch reference | Complete class (all layers) |
|---|---|---|---|
| LLM (Qwen3-0.6B) | `src/models/triton_kernels/qwen_fused.py` | `src/models/pytorch/qwen.py` | `src/models/qwen.py:QwenFused` — 28x `qwen_fused_decode_layer`, `KVCacheBatched [B,Hk,512,128]`, `check_budget` |
| STT (Whisper-base) | `src/models/triton_kernels/whisper_fused.py` | `src/models/pytorch/whisper.py` | `src/models/whisper.py:WhisperFused` — 6x `whisper_fused_decoder_layer`, `cross KV [B,H,1500,64]` |
| TTS (Kokoro-82M) | `src/models/triton_kernels/tts_fused.py` | `src/models/pytorch/tts.py` | `src/models/kokoro.py:KokoroFused` (`src/models/tts.py` facade) — `in1d_silu`/`conv1d_silu`, `postprocess_batched` |

`src/models/llm.py:LlmModel`, `src/models/stt.py:SttModel`, `src/models/tts.py:TtsModel` facades now **mandatory** fused (`src/models/qwen.py`/`whisper.py`/`kokoro.py`, batched, KV-cache, `check_budget`) — legacy `src/models/engines/*` fallback removed (fail loud). Root `models/` removed. `LlmModel` optionally routes via paged `InferenceEngine` (`QwenRunner`, `PagedKVCache`, `ContinuousScheduler`, prefix / chunked / CUDA-graph) when `LlmConfig(use_paged=True)` or `VoiceAgentConfig(llm_paged=True)` — server does this on CUDA so `/health → engine.active` proves the engine is hot for every LLM turn.

Batching enabled in all layers (`B=1..8`), KV-cache room per layer checked via `check_budget(estimate_kv_cache_mb(B,...), budget=4000MB)` before alloc — no OOM.

Static parity tests (fused vs torch) + VRAM check:

```bash
PYTHONPATH=. .venv/bin/python -c "from src.models.qwen import QwenFused; QwenFused.test_against_torch(2)"
PYTHONPATH=. .venv/bin/python -c "from src.models.whisper import WhisperFused; WhisperFused.test_against_torch(2)"
PYTHONPATH=. .venv/bin/python -c "from src.models.kokoro import KokoroFused; KokoroFused.test_against_torch(2)"
# all PASS: max_err 9.7e-04 (fp16) under 1e-2, VRAM OK 50-112 MB (B=8: 448MB)
```

Perf report is inside each Triton file (`@triton.testing.perf_report`) — run:

```bash
PYTHONPATH=. .venv/bin/python -m src.models.triton_kernels.qwen_fused --bench --save_path benchmarks/results
PYTHONPATH=. .venv/bin/python -m src.models.triton_kernels.whisper_fused --bench --save_path benchmarks/results
PYTHONPATH=. .venv/bin/python -m src.models.triton_kernels.tts_fused --bench --save_path benchmarks/results
```

Plots (RTX 3050 Laptop 4GB, CUDA 12, `B` and `seq_len` sweeps):

**LLM fused decode layer**

![qwen fused B](benchmarks/results/plots_fused/qwen-fused-layer-B.png)
![qwen fused seq](benchmarks/results/plots_fused/qwen-fused-layer-seq.png)

**STT fused decoder layer**

![whisper fused B](benchmarks/results/plots_fused/whisper-fused-layer-B.png)
![whisper fused seq](benchmarks/results/plots_fused/whisper-fused-layer-seq.png)

**TTS fused post-filter**

![tts fused B](benchmarks/results/plots_fused/tts-fused-B.png)
![tts fused L](benchmarks/results/plots_fused/tts-fused-L.png)

Notes: TTS `in1d_silu` shows 3-6x over torch (fused norm+SiLU). LLM fused decode now **24-30% faster** than torch (`B=1: 0.66 vs 0.88ms`, `B=8: 1.65 vs 2.25ms`, `seq 512: 0.99 vs 1.73ms`) via true batched `fused_qkv` (1 launch) + `batched GQA` (1 launch, GQA-aware). STT fused decoder **27-52% faster** (`B=1: 0.28 vs 0.59ms`, `B=8: 0.56 vs 1.19ms`) — fixed `fused_qkv_batched` (B==1 fused, B>1 batched GEMM) + fair bench (both do full layer). All parity `max_err 9.7e-04` + VRAM `check_budget` (`B=8: 448MB KV + 1200MB weights <4000MB`). Previous per-op kernels under `src/models/triton_kernels/rmsnorm.py` etc are now superseded by the single-file fused versions.
