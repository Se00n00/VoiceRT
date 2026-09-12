# Benchmarks & methodology

What each benchmark measures, the exact schema of the machine-readable
results, and how to reproduce every number in the README. No number in
this repo is hand-waved — each traces to a command below.

## 1. Suite map

| Benchmark | Path driven | Workload | Key metrics | Output |
|---|---|---|---|---|
| `qwen` | `VoiceEngine.chat` | 3 fixed prompts | TTFT, TPS, VRAM | stdout |
| `whisper` | `VoiceEngine.transcribe` | synth noise or `--wav`, `--iters 5` | total p50/p99, TTFS, RTF | stdout |
| `tts` | `VoiceEngine.speak` | 3 fixed sentences | RTF, VRAM | stdout + `results/tts_*.wav` |
| `vad` | `VoiceEngine.vad_segments` | synth/file, `--iters 20` | p50/p99, RTF | stdout |
| `pipeline` | live `POST /v1/voice` | synth sine, conc sweep 1/2/4 | TTFA/E2E p50/p99, req/s | stdout |
| `capacity` | engine + `stream_turn` round-trip | genlen×conc matrix + telemetry | TTFT/TPO/TPS/thr/util/W/cost | json + md + SVG |

Dispatcher: `PYTHONPATH=. python scripts/benchmark.py {vad,whisper,qwen,tts,pipeline,capacity}`.

## 2. Definitions (used consistently)

- **TTFT** — call entry → first output token id (LLM `generate()["ttft"]`).
- **TPO** — mean inter-token time: `(wall − ttft) / max(n−1, 1)`.
- **TPS** — `n_tokens / wall` per request; **tok/s** — sustained
  `total_tokens / wall` across concurrent requests.
- **TTFA** — `stream_turn` entry → first synthesized audio chunk.
- **E2E** — full `stream_turn` wall time.
- **RTF** — processing time / audio duration (<1 = faster than real-time).
- **req/s** — completed requests / wall across the sweep.
- **Efficiency** — `tok/s per watt`, `tok/s per GB VRAM`.
- **Cost/1M** — `1e6 / tok_per_s / 3600 × rate_usd_per_hr` (default $0.05).

## 3. `bench_capacity.py` internals

```
VoiceEngine() warm (~2 min first boot)
└── NvTelemetry (nvidia-smi poll @2Hz: util%, power W, mem MB)
    ├── bench_llm: genlen×conc ThreadPool chats (llm.generate directly,
    │              same kernels as served path) → TTFT/TPO/TPS/thr
    ├── bench_e2e: Kokoro synth → 24k→16k → stream_turn → TTFA/E2E/VRAM
    └── bench_stt_tts: noise transcribe + sentence speak → RTF spot
with_cost(rate) → cost/1M + tok/s/W + tok/s/GB
write_results → results/capacity_<ts>.json + results/capacity.md
write_graphs  → matplotlib PNG if present, else stdlib SVG (always)
```

Telemetry caveats (read before quoting util%): 2 Hz polling under-fills
on sub-second bursts, so `gpu_util_mean` is a **lower bound**; power and
VRAM-peak readings are robust. The first matrix row includes one-time
CUDA init — steady state is the rows after it.

## 4. Result schema (`results/capacity_<ts>.json`)

```json
{
  "meta": {"ts": "…", "gpu": "…", "vram_total_mb": 4096.0,
           "cost_per_hour_usd": 0.05, "telemetry": {…},
           "capacity_plan": {"sessions": 0, …}},
  "llm": [{"genlen": 48, "conc": 1, "reqs": 1,
           "ttft_p50_ms": 14.0, "tpo_ms": 14.4, "tps_mean": 69.6,
           "thr_req_s": 6.31, "thr_tok_s": 69.4, "vram_mb": 1957.0,
           "gpu_util_mean": 1.0, "gpu_util_p95": 1.0, "power_w_mean": 22.1,
           "cost_per_1m_usd": 0.20, "toks_per_watt": 3.14,
           "toks_per_gb": 36.3}],
  "e2e": {"ttfa_ms": 308.0, "e2e_ms": 827.0, …},
  "spot": {"stt_ms": 59.0, "stt_rtf": 0.020, "tts_ms": 141.0, …}
}
```

## 5. Reproduce the README numbers

```bash
# Full matrix used in the README (genlens {16,48} x conc {1,2})
PYTHONPATH=. python scripts/benchmark.py capacity --quick

# Wider sweep (genlens {16,48,128} x conc {1,2,4})
PYTHONPATH=. python scripts/benchmark.py capacity --full

# Single legs
PYTHONPATH=. python scripts/benchmark.py qwen -- --max-tokens 8
PYTHONPATH=. python scripts/benchmark.py whisper
PYTHONPATH=. python scripts/benchmark.py tts

# Endpoint sweep (needs the server up in another shell)
PYTHONPATH=. python serve.py &
PYTHONPATH=. python scripts/benchmark.py pipeline
```

Graphs land in `benchmarks/results/cap_*.svg` and are embedded in the
README; `capacity.md` is regenerated every run so tables never go stale
without anyone noticing the JSON underneath changed.
