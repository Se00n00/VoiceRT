# 05 — Models

Four legs, one config object, and the backends behind them.

## 5.1 The legs

| Leg | Module | What it does |
|---|---|---|
| VAD | `src/models/vad.py` | speech/silence, per chunk |
| STT | `src/models/stt.py` | audio → text |
| LLM | `src/models/llm.py` | text → text (+ tools) |
| TTS | `src/models/tts.py` | text → audio |

Each is a thin async facade over a concrete engine. The rule that makes them
cheap: a facade imports its backend **lazily inside `_backend()`**, so
`import src.models.llm` never loads weights.

Voice and STT implementations: `src/models/vad.py`, `stt.py`, plus
`whisper.py` for the Whisper-base leg and `kokoro.py` for Kokoro-82M.

## 5.2 `LlmConfig`

`src/models/llm.py:144-241`. Defaults:

| Field | Default | Line |
|---|---|---|
| `model` | `Qwen/Qwen3-0.6B` | `llm.py:147` |
| `backend` | `qwen` | `llm.py:165` |
| `max_tokens` | 48 | `llm.py:168` |
| `max_seq` | 8192 | `llm.py:169` |
| `thinking` | `False` | `llm.py:172` |

`max_tokens: 48` is the single most consequential default in the file. A voice
reply is a sentence, not an essay, and the tail of a decode is where small
models start rambling.

`thinking` controls whether Qwen3 `<think>` traces are kept; they are stripped
from output by `split_thinking` (`llm.py:23`).

## 5.3 Backends

Two survive: `qwen` (fused in-process) and `bonsai` (llama.cpp sidecar on
a port, speaking HTTP). `SIDECAR_BACKENDS = ("bonsai",)` (`llm.py:16`).

| Backend | Default port | Weights |
|---|---|---|
| `qwen` | — (fused) | `Qwen/Qwen3-0.6B` on CUDA/CPU |
| `bonsai` | 8081 | `prism-ml/Ternary-Bonsai-2-27B` GGUF |

Retired 2026-10-05: `gemma`/`gemma270`/`qwen17`/`qwen06` sidecars,
`minicpm`/`minicpm_q4k`, and Qwen2.5 bench defaults (now Qwen3-0.6B).
STT is pure-torch Whisper-base, TTS is Kokoro + torch post — neither
uses custom kernels anymore.

## 5.4 Readiness

Two guards, both used by the eval runners so a run cannot silently pass on CPU:

- `assert_cuda_leg` (`llm.py:75`) — aborts if the leg is not on CUDA.
- `assert_ready_leg` (`llm.py:91`) — for sidecars, checks the server actually
  answers.

Every runner calls the first before generating. That is why the whole-dataset
Kaggle runs are trustworthy: a CPU fallback fails fast instead of producing
plausible-looking numbers at 30× the latency.

## 5.5 Streaming

`LlmModel.stream` is an async generator yielding `LlmToken` (`llm.py:250`).
`LlmResult` (`llm.py:242`) carries the assembled text.

`LocalChatModel._stream_raw` (`src/models/chat_model.py`) consumes it
and does two things worth noting:

1. **Groups consecutive think pieces into one `<think>…</think>` block**
   (`chat_model.py:314-328`) rather than emitting a fragment per token.
2. **Re-decodes the whole id list** when the stream exposed ids
   (`chat_model.py:299-308`), so the final text is authoritative rather than an
   accumulation of deltas.

Non-streaming `generate` is the fallback (`chat_model.py:332-342`).

## 5.6 Memory as a model concern

`src/models/runtime/` is the leaf layer — device, memory, profiler, tensor
helpers, capacity and scheduler. It imports nothing from `engine/`, `models/`,
or the Triton kernels.

`src/models/triton_kernels/` holds hand-written kernels and per-leg surfaces;
they import only raw-kernel siblings plus torch, never `models/`.

`PYTHONPATH=.` is required because the kernels live at `src.models.triton_kernels`.

## 5.7 VRAM

On a 4 GB card, one GPU process at a time is a hard rule, not a preference — a
second model instance is an OOM, which is what killed a GAIA run.

`LlmConfig.backend` picks the local leg: `qwen` is Qwen3-0.6B fused
Triton; `bonsai` is Ternary Bonsai 2 27B via a Prism-fork llama.cpp
sidecar on `bonsai_port` (8081, `bonsai_ngl="auto"` probes free VRAM).
Cloud backends (`groq`, `gemini`, `anthropic`, `openai`, `openrouter`,
`deepseek`, `openai_compat`) use stock LangChain chat models and hold no
local weights.

## 5.8 See also

- [01-agent.md](01-agent.md) — how the planner uses the model
- [07-inference.md](07-inference.md) — retired runtime note
