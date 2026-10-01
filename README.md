<img src="VoiceRT.png">

---

```
─── 01 / HIGH LEVEL ARCHITECTURE ───────────────────────────────────────────────────────────

      ┌──────────────┐   ┌─────────────────────┐   ┌─────────────────┐   ┌───────────────┐
wav |>  | Silero VAD | |> | Tiny Whisper (STT) | |> | Mini CPM (LLM) | |> | Kokoro (TTS) | |> wav
      └──────────────┘   └─────────────────────┘   └─────────────────┘   └───────────────┘
      
─── 01 ~ 1 / CONTEXT & MEMORY ───────────────────────────────────────────────────────────────────────

  BUDGET 16K - pack_prompt cuts obs → hist, never system/facts
  ┌────────────┬───────────┬──────────────────────┬───────────────┬────────────┐
  │ system +   │ tool      │ history              │ observations  │ reserve    │
  │ preamble   │ specs     │ elastic - newest-    │ 1500 → 500    │ 256 / 512  │
  │ ~2.5K      │ ~1K       │ first fill           │ → drop        │ floor      │
  │ FIXED      │ FIXED     │ ELASTIC              │ ELASTIC       │ FIXED      │
  └────────────┴───────────┴──────────────────────┴───────────────┴────────────┘
  count: /tokenize exact, else chars/3.5 - per-backend rows in budget.py:BUDGETS

  MEMORY - L1 evict -> L2 store - L2/L3 recall -> prompt (every turn)
  ┌────────────────────┐
  │ L1 WORKING         │
  │ sessions/*.json    │
  │ window+tok-TTL-LRU │
  └─────────┬──────────┘
            │ evict → summarize_turns → store()
            ▼
  ┌────────────────────┐                 ┌──────────────────────┐
  │ L2 EPISODIC bge    │ ──────────────▶ │ ASSEMBLED PROMPT     │
  └────────────────────┘ [Past episodes] │ facts → episodes →   │
  ┌────────────────────┐                 │ hist newest-first    │
  │ L3 FACTS s-P-o     │ ──────────────▶ │ + CWD trailer        │
  └────────────────────┘ [User facts]    └──────────────────────┘
  L2 = memory/episodic.db (bge-small cos+kw+rec) - L3 = memory/facts.md (newer wins)

  TURNS - middleware in order, then pack
  ┌──────────┐    ┌────────────┐    ┌────────────┐    ┌─────────────┐
  │ write_   │───▶│ TrimObs    │───▶│ InjectTool │───▶│ Summarizer  │──▶ pack_prompt
  │ todos    │    │ head 1500  │    │ sem top-k  │    │ trg .9xctx  │
  │ (plan)   │    │ + omitted  │    │ ~1K specs  │    │ keep 6      │
  └──────────┘    └────────────┘    └────────────┘    └─────────────┘

  COMPACTION - cross-turn
  ┌───────────────┐                     ┌────────────────┐
  │ model-switch  │ ──────────────────▶ │ compacted sess │
  │ (bridge)      │  <=150w + last 2    │ summary + 2    │
  └───────────────┘  per-target skip    └────────────────┘
  ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐    ┌───────────┐
  │ plan         │───▶│ stash frame    │───▶│ lean subtask x N │───▶│ envelope  │
  │ 1 cheap call │    │ sim_<sid>.json │    │ sliver+preamble  │    │ > parent  │
  │ numbered     │    │                │    │ scratch session  │    │ + restore │
  └──────────────┘    └────────────────┘    └──────────────────┘    └───────────┘

```

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![CUDA 12](https://img.shields.io/badge/CUDA-12-green)
![VRAM 4GB](https://img.shields.io/badge/VRAM-4GB-orange)
![TTFT 14ms](https://img.shields.io/badge/TTFT-14ms-brightgreen)
![Tests 84 passing](https://img.shields.io/badge/tests-84_passing-brightgreen)

**Full voice loop — speech in, speech out — on a single 4GB laptop GPU.** Mic/wav → text → reply → voice in <1s, no cloud.

## Run in 2 minutes

```bash
cd VoiceAgent/voice-pipeline
pip install -r requirements.txt          # get CUDA torch first: https://pytorch.org/get-started/locally/
PYTHONPATH=. python -m src.models.download   # first time: ~5GB weights
PYTHONPATH=. python server.py                # warms 4 legs (~2 min first boot) → voice-agent ready
curl http://localhost:8003/health            # {ok, agent_loaded, missing:[], vram_mb, engine:{active}}
curl http://localhost:8003/metrics
# talk: ws://localhost:8003/talk  (see docs/architecture.md for protocol)
```

`PYTHONPATH=.` is required — kernels live at `src.models.triton_kernels`.

Docs: [`docs/architecture.md`](docs/architecture.md) · [`docs/capacity.md`](docs/capacity.md) · [`docs/inference_engine.md`](docs/inference_engine.md) · [`docs/benchmarks.md`](docs/benchmarks.md) · [`docs/README.md`](docs/README.md)

Project structure → [`docs/architecture.md#1-component-map`](docs/architecture.md)

## Architecture — VoiceAgent + Inference Engine + Kernels

```
                    ┌─────────────────────────────────────────┐
                    │              VoiceAgent                 │
  mic/wav ──► VAD ──┤  Silero ONNX (CPU, RTF 0.01)            │
              │     │       │                                 │
              │     │       ▼                                 │
              │     │  STT  Whisper-base ─┐                   │
              │     │  fused decoder x6   │ fused Triton      │
              │     │  batched B=1..8     │ + torch fallback  │
              │     │       │             │ parity-tested     │
              │     │       ▼             │                   │
                      │     │  front LLM Qwen3-0.6B ─┐ yes/no:   │
                      │     │  is this worker work?  │ escalate? │
                      │     │       │                 │ (no tool) │
                      │     │       ▼ (escalated)    │           │
                     │     │  worker LLM Bonsai 27B ─┤           │
                     │     │  ┌────────────────┴──────────────────┐  │
              │     │  │      Inference Engine (paged)     │  │
              │     │  │  scheduling  ContinuousScheduler  │  │
              │     │  │  batching    make_batch + token  │  │
              │     │  │  KV cache    PagedKVCache/Pool   │  │
              │     │  │  memory      BlockManager +      │  │
              │     │  │              auto num_blocks     │  │
              │     │  │  features: prefix cache / chunked│  │
              │     │  │           prefill / CUDA-graph / │  │
              │     │  │           fused paged attention  │  │
              │     │  │  runner    QwenRunner (real     │  │
              │     │  │            weights, no dummy)   │  │
              │     │  └────────────┬────────────────────┘  │
              │     │       │       │ streaming tokens      │
              │     │  sentence splitter (overlap)          │
              │     │       │                                 │
              │     │  TTS  Kokoro-82M ─┐                   │
              │     │  batched post     │ fused in1d_silu / │
              │     │  filter           │ conv1d_silu       │
              │     │       │           │ + resample       │
              └─────┘       ▼           └───────────────────┘
                         wav out (24kHz)
```

**How a kernel is built (same for Qwen/Whisper/Kokoro):**
`src/models/triton_kernels/qwen_fused.py` — *one file* = all fused kernels + `@triton.testing.perf_report` bench + torch exact fallback. `src/models/pytorch/qwen.py` is the PyTorch reference. `src/models/qwen.py:QwenFused` loops it `28×` with `KVCacheBatched` and `check_budget`. Same pattern for `whisper_fused` (6×) and `tts_fused`. Facades `src/models/llm.py:SttModel/TtsModel` are mandatory fused (no legacy `engines/*` fallback). `LlmModel(use_paged=True)` wraps the same weights in the paged engine; `VoiceAgent(llm_paged=True)` / `server.py` on CUDA activates it — `GET /health → engine.active` proves it.

Verbose specs → docs: capacity model, queue, Triton details, API and config.

## Agent — text in, tools out, reply back

### Two brains: the front routes, the worker works

`server.py /talk` and `bridge.py` run **delegation on by default**. A small
front model (Qwen3-0.6B) owns the conversation and is given **no tools at
all**: it is asked one yes/no question — *does this need the worker?* — and
answers with a single word. On `YES` the turn runs on the worker leg (Bonsai
27B) with the whole existing agentic harness. Delegation does not build a
second harness — it routes into the one already there, so tools, memory and
sessions stay shared.

```
user ──► front (Qwen3-0.6B, no tools, one-word decision)
            │
            ├── NO ──► front writes a short reply ──► TTS  (never touches the 27B)
            │
            └── YES ─► worker (Bonsai 27B, full tool set) ──► reply → TTS
```

Four deliberate properties:

- **Chat never loads the worker.** "hi" costs a 0.6B generate, not a 27B
  one. This is the whole latency argument.
- **The worker gets the user's own words.** The front writes no task
  string, so there is nothing for it to garble, drop or invent — Bonsai
  re-reads the raw utterance and the history and extracts the intent
  itself. A delegated result also goes straight to TTS; the front does not
  get to reword it, so a correct result can't be mangled on the way out.
- **Failure is asymmetric.** A dead front leg, an unparseable answer or a
  narration-shaped reply all route to the worker, because that is the leg
  that can actually do anything. A worker that dies **says so** instead of
  going silent, since silence there reads as "done".
- **The decision is asked first.** The flag is the *first* call of the
  turn, so a delegated task costs one 6-token generate and never produces a
  chat reply nobody will hear. Asking for the flag and the reply in one
  generation was measured and drops real tasks (31/37 vs 34/37).

### Why a boolean and not a tool

The front leg used to hold exactly one tool, `delegate`, and pick a
function envelope with a task string as its argument. Measured on real
task prompts with Qwen3-0.6B, that was the bottleneck, not the routing
idea: **0 of 8** prompts produced a valid call. The model had the intent
and no trouble narrating it ("Sure, I renamed it"), and just could not be
made to fill in the envelope. True token-level prefill, which bypassed the
envelope entirely, delegated 10/10 — but it also escalated every bit of
chit-chat, because a 1B-scale model cannot express "not this one" without
being handed the alternative.

A single yes/no question is the one form of that discrimination a 0.6B
does reliably, and the asymmetry lets it be biased hard toward `YES`: a
false positive costs one slow worker turn, a false negative silently drops
a task the user asked for. With a 5-pair few-shot prompt, on real weights:

| | result |
| --- | --- |
| task prompts escalated | **19 / 19** (zero dropped) |
| chit-chat kept on the front leg | **15 / 18** |
| chat latency | **0.25 s** mean, 0.13–0.47 s (was 0.7–1.4 s with the tool prompt) |

Turn cost: `YES` = 1 call, 6 tokens. `NO` = 2 calls (flag, then reply).

`configs/delegate.yaml` picks the front's placement — `fused` (CUDA,
default), `cpu` (0 VRAM), or `sidecar` (a 0-VRAM llama.cpp CPU server) —
plus the backstop and route-event switches. The backstop is a scored regex
pass with no second LLM call, kept **on** as the last override before a
chat reply: a 0.6B model's default failure is narrating work it never did,
and that is the one bug this design must never ship. It barely fires — on
the 37-prompt run above the model never said `NO` to anything the regex
scored as work — so it costs a regex match and buys insurance against the
worst outcome.

The same regex runs on **every** turn as telemetry either way. Where it
disagrees with the model, the count surfaces on `GET /health`
(`delegate.route_skew`): `regex_no_model_yes` means the model escalated
something the regex reads as chatter (the regex is usually wrong — it
scored "how do i center a div" 0), and `model_no_regex_yes` is the
backstop catching a miss. Skew is recorded, never used to decide.

| Switch | Default | Effect |
| --- | --- | --- |
| `VOICE_DELEGATE=0` | on | single brain — the llm leg answers everything |
| `VOICE_DELEGATE_CONFIG` | `configs/delegate.yaml` | path to the placement/routing config |
| `VoiceAgentConfig(delegate=...)` | **off** | library default: no second set of weights unless asked |

The library default is off on purpose: delegation builds a *second* set of
weights, so a bare `VoiceAgent()` that never asked for two brains should
not pay for it. The app entry points are where that decision belongs.

### The deep-agent loop (worker leg)

```
text in ──► VoiceAgent.run_text ──► deep-agent loop (≤6 steps, lock+queue per session)
                                          │ think → toolcall (one action per step)
                                          ▼
                                   exec/read/write/edit/list/grep (host backend)
                                   python_exec/fetch/web_search (MCP stdio)
                                          │ policy deny/confirm + y/n gate
                                          ▼
                                   chat reply (+ thinking + token stream)
sessions/*.json: 8-turn window + token budget (see Context Engineering)
```

## Context Engineering — one budget per backend, enforced at assembly

```
16K window
├─ system + preamble ......... fixed (~2.5K, never cut)
├─ tool specs ................ fixed (~1K narrowed, never cut)
├─ history window ............ elastic (newest-first fill to remainder)
├─ observations .............. elastic per-call (1500 → 500 → drop)
└─ generation reserve ........ fixed floor (256/512, never spent by input)
```

Rule: step floor ≥ think cap + min answer (a think cap above the floor
truncates mid-thought). Per-backend rows (`src/agent/budget.py:BUDGETS`):

```
backend    ctx    think cap   floor   history ≈
bonsai     16K    128         512     ~9K
qwen17     32K    512         512     ~22K
gemma270   32K    256         256     ~22K
gemma      4K     256         256     0 by caps (measured system ~0.5K → ~2K real room)
```

`pack_prompt` cuts obs → history, never system/facts. L1 window, L2
episodic (`memory/episodic.db`), L3 facts (`memory/facts.md`) distill
from packed context — see PLAN.md (local-only).

## Benchmarks — measured on RTX 3050 Laptop 4GB (CUDA 12, torch 2.5.1)

All commands assume `PYTHONPATH=.` from inside `voice-pipeline/`.

### Reproduce

```bash
# fused kernel microbenchmarks (perf_report inside each Triton file)
.venv/bin/python -m src.models.triton_kernels.qwen_fused --bench --save_path benchmarks/results
.venv/bin/python -m src.models.triton_kernels.whisper_fused --bench --save_path benchmarks/results
.venv/bin/python -m src.models.triton_kernels.tts_fused --bench --save_path benchmarks/results

# inference engine features (real QwenRunner, isolated subprocess per config)
.venv/bin/python benchmarks/engine_bench.py --num-requests 8 --max-tokens 32
```

### LLM: inference engine features (Qwen3-0.6B, real QwenRunner, `num_blocks=16`)

Real `QwenRunner` only — isolated subprocess per config; `DummyRunner` removed.

| config | 32 tok/req: tok/s | delta | 8 tok/req: tok/s | delta |
|---|---|---|---|---|
| base (no features) | 42.0 | — | 21.7 | — |
| + prefix caching | 48.8 | **+16.4%** | 19.4 | -10.5% |
| + chunked prefill | 49.3 | **+17.4%** | 30.6 | **+41.3%** |
| + CUDA graph | 48.4 | **+15.3%** | 31.8 | **+46.7%** |
| all features | 48.0 | **+14.4%** | 32.4 | **+49.4%** |

Longer generations amortize prefill; short bursts benefit most from chunked / CUDA-graph.

### Fused kernel microbenchmarks (single layer vs eager torch)

Parity: `max_err 9.7e-04` (fp16) under `1e-2`. VRAM `check_budget` passes at B=8 (448MB KV + 1200MB weights < 4000MB).

**LLM — `qwen_fused_decode_layer` (28 layers, B=1..8, seq 1..512)**

| | B=1 | B=4 | B=8 | seq 512 |
|---|---|---|---|---|
| fused | 0.66ms | 1.05ms | 1.65ms | 0.99ms |
| torch | 0.88ms | 1.62ms | 2.25ms | 1.73ms |
| **speedup** | **24%** | **35%** | **27%** | **43%** |

![qwen fused B](benchmarks/results/plots_fused/qwen-fused-layer-B.png)
![qwen fused seq](benchmarks/results/plots_fused/qwen-fused-layer-seq.png)

**STT — `whisper_fused_decoder_layer` (6 layers, B=1..8)**

| | B=1 | B=4 | B=8 |
|---|---|---|---|
| fused | 0.28ms | 0.42ms | 0.56ms |
| torch | 0.59ms | 0.89ms | 1.19ms |
| **speedup** | **52%** | **53%** | **53%** |

![whisper fused B](benchmarks/results/plots_fused/whisper-fused-layer-B.png)
![whisper fused seq](benchmarks/results/plots_fused/whisper-fused-layer-seq.png)

**TTS — `in1d_silu` / `conv1d_silu` / `postprocess_batched`**

| | B=1 | B=4 | B=8 | L=2048 |
|---|---|---|---|---|
| `in1d_silu` | 0.12ms | 0.31ms | 0.58ms | 3-6× torch |
| `conv1d_silu` | 0.08ms | 0.22ms | 0.41ms | cuDNN parity |

![tts fused B](benchmarks/results/plots_fused/tts-fused-B.png)
![tts fused L](benchmarks/results/plots_fused/tts-fused-L.png)

### LLM: latency vs concurrency vs generation length (reference Qwen2.5-0.5B stack)

| genlen | conc | TTFT p50 | TPO | TPS | req/s | tok/s | util% | W | tok/s/W | $/1M |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 214ms¹ | 20.3ms | 26.4 | 2.40 | 26.4 | 2.0 | 19.8 | 1.33 | $0.5265 |
| 16 | 2 | 30ms | 29.4ms | 34.6 | 4.85 | 65.5 | 1.0 | 22.1 | 2.96 | $0.2122 |
| 48 | 1 | 14ms | 14.4ms | 69.6 | 6.31 | 69.4 | 1.0 | 22.1 | 3.14 | $0.2000 |
| 48 | 2 | 28ms | 25.8ms | 40.3 | 3.44 | 67.1 | 16.0 | 27.5 | 2.44 | $0.2070 |

¹ First generate after load includes CUDA init; steady-state TTFT 14–30 ms band.

Concurrency doubles token throughput (26 → 66 tok/s at genlen 16) while per-request latency rises — measured, not assumed. Peak efficiency: **3.14 tok/s per watt** (conc 1, genlen 48).

### Cost per million tokens (local $0.05/hr)

| workload | tok/s | $/1M tokens |
|---|---|---|
| genlen 48, conc 1 | 69.4 | **$0.2000** |
| genlen 48, conc 2 | 67.1 | $0.2070 |
| genlen 16, conc 2 | 65.5 | $0.2122 |

### Full voice turn (round-trip: synthetic speech → full pipeline)

| | TTFA | E2E | VRAM |
|---|---|---|---|
| Fast path `--fast` (Qwen3-0.6B, VAD→STT→1 LLM call→TTS) | 329 ms | 336 ms | 3034 MB |

Measured on the two-brain route (RTX 3050, fused front, real weights):
chat turn **0.7–1.4 s** front-to-reply, with the worker leg never
loaded. See `tests/src/test_delegate.py` for the routing contract.
| Round-trip (synthetic speech in) | 308 ms | 827 ms | 1957 MB |
| Earlier live turn | 438 ms | 640 ms | 1941 MB |
| LibriSpeech samples (previous) | 531–858 ms | 1321–2590 ms | 1937 MB |

### Per-leg spot checks

| Leg | Latency | Real-time factor |
|---|---|---|
| VAD | ~50 ms | 0.01 |
| STT (Whisper) | 59 ms / 3 s audio | **0.020** (50× real-time) |
| LLM TTFT / decode | 14 ms / ~69 tok/s | — |
| TTS (Kokoro, eager) | 141 ms | **0.045** (22× real-time) |

### Capacity: VRAM-probed sessions (RTX 3050 4096 MB, Qwen3-0.6B)

| | genlen 48 | genlen 128 |
|---|---|---|
| Baseline (weights) | ~1900 MB | ~1900 MB |
| Headroom (10%) | 410 MB | 410 MB |
| Usable | ~1700 MB | ~1700 MB |
| Per session (270 MB @48 tok) | **~6–7 sessions** | **~5–6 sessions** |

Server auto-derives this at boot (`src/models/runtime/capacity.py`, `auto_engine_config`).

---

All plots in `benchmarks/results/plots_fused/` · engine bench in `benchmarks/engine_bench.py` · capacity math in `src/models/runtime/capacity.py`.

## References

### Transformers

- [1]: https://www.aleksagordic.com/blog/transformer | Transformer from scratch - by Aleksa Gordić
- [2]: https://www.youtube.com/playlist?list=PLqO45Dg1pMhlDBZTMqVL2GU-14xYip2y2 | Transformer playlist - by Umar Jamil

### Inference

- [3]: https://learn-inference.com/ | learn-inference.com - by learn-inference.com

### Kernels

- [4]: https://triton-lang.org/ | OpenAI Triton - by OpenAI

### Harness

- [5]: https://martinfowler.com/articles/harness-engineering.html | Harness engineering for coding agent users - by Birgitta Böckeler
- [6]: https://youtu.be/C_GG5g38vLU?si=RaptRWbS-27f7r78 | Harnesses in AI: A Deep Dive - by Tejas Kumar (IBM)
- [7]: https://youtu.be/tYchws8hpd8?si=pcI1PIS1qqmmSLIX | AI Harness Engineering - Forward Deployed Engineering Tutorial - by YouTube

### Agentic Design

- [8]: https://github.com/ombharatiya/ai-system-design-guide/tree/main | AI system design guide - by Om Bharatiya

### Context Engineering

- [9]: https://www.langchain.com/blog/context-engineering-for-agents | Context Engineering for Agents - by LangChain
- [10]: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | Effective context engineering for AI agents - by Anthropic (Applied AI team)

More: [`docs/architecture.md`](docs/architecture.md) startup + WS flow, [`docs/inference_engine.md`](docs/inference_engine.md) scheduling/batching/KV/memory, [`docs/capacity.md`](docs/capacity.md) VRAM math, plots in `benchmarks/results/plots_fused/`.

---

## 🚧 TUI version — work in progress

I'm building an opencode-style **terminal voice agent** on top of this pipeline: full-screen split UI — conversation + input on the left, live USER/AGENT audio visualizer on the right — with the same single Qwen model chatting or running shell commands (`v` voice turn, `y`/`n` confirm gate, `/` commands, per-session LangChain memory). Two frontends exist: `tui.py` (Textual) and the current **OpenTUI (Solid) app in `src/tui/`** backed by `bridge.py` (`:8004`, `server.py` untouched). Not done yet: visual polish and edge cases are still being worked on.

See it live (one command — the TUI spawns its own local agent backend; `server.py` stays out of it):

```bash
cd src/tui && npm install && npm run dev
```

First boot warms legs (~1-2 min). Headphones (or speakers down) avoid the mic re-ingesting replies; without them the app still works — it pauses listening while speaking and discards its own echo. Advanced: run the backend separately (`PYTHONPATH=. python bridge.py`) and point the TUI at it with `VOICE_BRIDGE=http://127.0.0.1:8004`.

On quit (`ctrl+c` or `/quit`) a farewell card prints a two-word name for the session and the command to resume it:

```bash
voicert -s ses_f09c7b175ffews1QW321k2D9An
```

Session ids are `ses_` + 26 base62 chars, written to `sessions/<id>.json`, so a resume survives a restart of both the TUI and `bridge.py`. `voicert -s <id>` also accepts a bare id (no prefix); anything that isn't 6–64 alphanumerics is rejected.

### Monotonic shades

The palette is black and white plus **one** hue. `/shade <name>` collapses every chromatic role (accent, warn, danger, voice label) onto that single color; the backdrop, ink and borders stay exactly where they were. `/shade` lists them, `/shade none` restores the original multi-hue palette, and `VOICE_SHADE=blue|yellow|red|orange|green|none` picks the starting shade (default **green**). The wordmark's `RT` half is painted in the active shade too.

The trade is deliberate: errors and confirm prompts stop being red and yellow, so they're told apart by their glyph and label rather than color.

## Sandboxed tool execution (docker)

Terminal tools run on the host by default. Pass a `SandboxConfig` to run
`exec`/`exec_bg`/`poll` inside a per-turn container instead
(`src/sandbox/docker.py:DockerSandbox`, a deepagents `BaseSandbox` over
the docker CLI — no new dependencies):

```python
from src.main import VoiceAgent, VoiceAgentConfig
from src.sandbox.docker import SandboxConfig
agent = VoiceAgent(VoiceAgentConfig(sandbox=SandboxConfig()))
# or: PYTHONPATH=. .venv/bin/python llm_chat.py --sandbox
```

What it is (honest version): separate pid/network/mount namespaces,
`--cap-drop ALL`, memory/cpu/pids limits, no privileged flag. The turn
cwd is bind-mounted at `/work`, so file ops work on host files unchanged
— containment covers processes, network, devices and resources, not
filesystem secrecy. Policy deny-list + `y/n` confirm stay on top.
`read`/`write`/`edit`/`grep`/`list` stay host-side on the same tree.
`python_exec`/`docker_exec` always run containerized (MCP or local).

Defaults: image `python:3.12-slim` (needs `bash`, `python3`, GNU
`timeout`), `--network none`, `--memory 1g`, `--cpus 2`. Requirements:
docker daemon access (`sudo usermod -aG docker $USER` + re-login) —
without it the turn fails fast with the fix printed.
