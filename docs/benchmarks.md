# Benchmarks — RTX 3050 Laptop 4GB (torch 2.5.1+cu124, driver 615.71)

Legs, turns and routing re-measured 2026-10-02 (means over n=4 steady-state
turns unless noted; first-call warmup excluded). Rows marked historical
were not re-run: kernel microbenches need a long exclusive GPU session,
and the concurrency/cost tables are a Qwen2.5-era reference stack. Engine
benches were re-run 2026-10-03 after the `src.inference` parser fix.

All commands assume `PYTHONPATH=.` from inside `voice-pipeline/`.

## Reproduce

```bash
# fused kernel microbenchmarks (perf_report inside each Triton file)
.venv/bin/python -m src.models.triton_kernels.qwen_fused --bench --save_path benchmarks/results
.venv/bin/python -m src.models.triton_kernels.whisper_fused --bench --save_path benchmarks/results
.venv/bin/python -m src.models.triton_kernels.tts_fused --bench --save_path benchmarks/results

# inference engine features (real QwenRunner, isolated subprocess per config)
.venv/bin/python benchmarks/engine_bench.py --num-requests 8 --max-tokens 32
```

## LLM: inference engine features (Qwen3-0.6B, real QwenRunner, `num_blocks=16`)

Re-run 2026-10-03 after fixing the `engine.py:431` indentation (the file
imports again). Real `QwenRunner` only — isolated subprocess per config;
`DummyRunner` removed. Workload: 8 requests × 32 tokens (256 total).

| config | tok/s | delta vs base |
|---|---|---|
| base (no features) | 39.5 | — |
| + prefix caching | 35.9 | -9.1% |
| + chunked prefill | 34.4 | -12.8% |
| + CUDA graph | 37.8 | -4.2% |
| all features | 34.7 | -12.0% |

Honest read: on this all-distinct-prompts workload the features do not pay —
prefix cache records 0 hits and CUDA graphs stay enabled-but-never-captured,
so each feature is pure overhead and base wins.

<details><summary>Earlier run (different workload mix)</summary>

| config | 32 tok/req: tok/s | delta | 8 tok/req: tok/s | delta |
|---|---|---|---|---|
| base (no features) | 42.0 | — | 21.7 | — |
| + prefix caching | 48.8 | **+16.4%** | 19.4 | -10.5% |
| + chunked prefill | 49.3 | **+17.4%** | 30.6 | **+41.3%** |
| + CUDA graph | 48.4 | **+15.3%** | 31.8 | **+46.7%** |
| all features | 48.0 | **+14.4%** | 32.4 | **+49.4%** |

</details>

## Fused kernel microbenchmarks (single layer vs eager torch)

Historical — not re-run 2026-10-02 (needs a long exclusive GPU session;
plots in `benchmarks/results/plots_fused/` are from the last full run).

Parity: `max_err 9.7e-04` (fp16) under `1e-2`. VRAM `check_budget` passes at B=8 (448MB KV + 1200MB weights < 4000MB).

### LLM — `qwen_fused_decode_layer` (28 layers, B=1..8, seq 1..512)

| | B=1 | B=4 | B=8 | seq 512 |
|---|---|---|---|---|
| fused | 0.66ms | 1.05ms | 1.65ms | 0.99ms |
| torch | 0.88ms | 1.62ms | 2.25ms | 1.73ms |
| **speedup** | **24%** | **35%** | **27%** | **43%** |

Plots:

![qwen fused B](../benchmarks/results/plots_fused/qwen-fused-layer-B.png)
![qwen fused seq](../benchmarks/results/plots_fused/qwen-fused-layer-seq.png)

### STT — `whisper_fused_decoder_layer` (6 layers, B=1..8)

| | B=1 | B=4 | B=8 |
|---|---|---|---|
| fused | 0.28ms | 0.42ms | 0.56ms |
| torch | 0.59ms | 0.89ms | 1.19ms |
| **speedup** | **52%** | **53%** | **53%** |

Plots:

![whisper fused B](../benchmarks/results/plots_fused/whisper-fused-layer-B.png)
![whisper fused seq](../benchmarks/results/plots_fused/whisper-fused-layer-seq.png)

### TTS — `in1d_silu` / `conv1d_silu` / `postprocess_batched`

| | B=1 | B=4 | B=8 | L=2048 |
|---|---|---|---|---|
| `in1d_silu` | 0.12ms | 0.31ms | 0.58ms | 3-6× torch |
| `conv1d_silu` | 0.08ms | 0.22ms | 0.41ms | cuDNN parity |

Plots:

![tts fused B](../benchmarks/results/plots_fused/tts-fused-B.png)
![tts fused L](../benchmarks/results/plots_fused/tts-fused-L.png)

## LLM: latency vs concurrency vs generation length (reference Qwen2.5-0.5B stack — historical, not re-run)

| genlen | conc | TTFT p50 | TPO | TPS | req/s | tok/s | util% | W | tok/s/W | $/1M |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 214ms¹ | 20.3ms | 26.4 | 2.40 | 26.4 | 2.0 | 19.8 | 1.33 | $0.5265 |
| 16 | 2 | 30ms | 29.4ms | 34.6 | 4.85 | 65.5 | 1.0 | 22.1 | 2.96 | $0.2122 |
| 48 | 1 | 14ms | 14.4ms | 69.6 | 6.31 | 69.4 | 1.0 | 22.1 | 3.14 | $0.2000 |
| 48 | 2 | 28ms | 25.8ms | 40.3 | 3.44 | 67.1 | 16.0 | 27.5 | 2.44 | $0.2070 |

¹ First generate after load includes CUDA init; steady-state TTFT 14–30 ms band.

Concurrency doubles token throughput (26 → 66 tok/s at genlen 16) while per-request latency rises — measured, not assumed. Peak efficiency: **3.14 tok/s per watt** (conc 1, genlen 48).

## Cost per million tokens (local $0.05/hr)

| workload | tok/s | $/1M tokens |
|---|---|---|
| genlen 48, conc 1 | 69.4 | **$0.2000** |
| genlen 48, conc 2 | 67.1 | $0.2070 |
| genlen 16, conc 2 | 65.5 | $0.2122 |

## Full voice turn (round-trip: synthetic speech → full pipeline)

Re-measured 2026-10-02 via `benchmarks/voice_latency.py` (single-brain
Qwen3-0.6B, 48 max-tokens, 1.95 s synth input, first turn excluded):

| | TTFA | E2E | VRAM |
|---|---|---|---|
| Fast path `--fast` (VAD→STT→1 LLM call→TTS, 7 ids) | 551 ms | 551 ms | 3038 MB |
| Default path (same input, 48 max-tokens) | 859 ms | 1021 ms | 3065 MB |

Earlier rows, kept for history: synthetic round-trip 308/827 ms,
live turn 438/640 ms, LibriSpeech samples 531–858 / 1321–2590 ms
(VRAM ~1940 MB — lighter legs than today's full stack).

### Lazy worker turns (2026-10-02, Bonsai CPU sidecar `ngl=0`)

| | measured |
|---|---|
| Boot to ready (lazy) | **~40 s**, 2283 MB VRAM, 0 worker procs |
| First YES (cold worker) | route 0.1 s → first output **44 s** → done **208 s** |
| Steady YES (warm worker) | route ~1 s → done **~340 s**; direct `:8081`: 13.7 s prompt, decode **1.4 tok/s** |
| Footprint | RAM 8→6 GB avail; VRAM 2283→3676 MB (sidecar CUDA context ~886 MB despite `ngl=0`) |

Observed failures (loud, no hangs): sidecar death minutes after boot
(probable kernel OOM) → next YES fails fast `Connection refused`; one
456 s turn died at TTS with CUDA OOM (64 MB alloc, 62 MB free).

![turn anatomy](../benchmarks/results/plots_voice/turn_anatomy.png)
![decode tok/s by leg](../benchmarks/results/plots_voice/toks_comparison.png)
![footprint](../benchmarks/results/plots_voice/footprint.png)

## Per-leg spot checks

Re-measured 2026-10-02 via `benchmarks/leg_profile.py` (steady-state of
3, 1.95 s speech in / 3.02 s audio out) and `benchmarks/voice_latency.py`
(n=4 turns, ranges in brackets):

| Leg | Latency | Real-time factor |
|---|---|---|
| VAD (Silero ONNX, CPU) | 23–25 ms / ~2 s audio [19–31] | **≈0.012** |
| STT (Whisper-base fused) | 113 ms / 1.95 s audio [113–113 steady; 143–221 in-turn] | **≈0.06–0.09** |
| LLM TTFT / decode (batch-1 fused) | 178 ms TTFT (~40-tok prompt) / 22.7 tok/s over 48 ids | — |
| TTS (Kokoro-82M, eager) | ~210 ms / 3.02 s audio [198–434 in-turn] | **≈0.07–0.14** |

Historical steady-state claims kept for reference: TTFT 14–30 ms band
(tiny prompts), decode ~69 tok/s at genlen 48 (batched harness), STT
59 ms / 3 s (RTF 0.020), TTS 141 ms (RTF 0.045). Today's box reads
slower across the board than those rows.

### Larger legs (2026-10-02)

| Leg | Latency / throughput |
|---|---|
| Qwen3-1.7B Q4_K_M CPU sidecar (`backend=qwen17`, :8084, greedy 48 ids) | warm 7 s · server TTFT ~80 ms steady (557 ms first call) · decode **12.5 tok/s** stable over 3 reps |
| Bonsai 27B worker | not runnable here: RAM guard refused (5.6 GB GGUF vs 3.9 GB available) — no tok/s |
| Gemma-4-E4B sidecar | not runnable here: needs ~6 GB RAM |

CUDA-graphs, engine features and kernel plots are blocked behind the
2026-10-02 GPU wedge (`ERR!`, cold boot required) plus the pre-existing
`src/inference` breakage — see `docs/plans/gpu-memory-guard.md` for the
pre-execution guard plan written up after the incident.

## Capacity: VRAM-probed sessions (RTX 3050 4096 MB, Qwen3-0.6B)

Recomputed 2026-10-02 via `src/models/runtime/capacity.py`
(baseline 2381 MB legs-resident, up from ~1900 MB):

| | genlen 48 | genlen 128 |
|---|---|---|
| Baseline (weights) | ~2380 MB | ~2380 MB |
| Headroom (10%) | 410 MB | 410 MB |
| Usable | ~1300 MB | ~1300 MB |
| Per session (~270 MB) | **~4 sessions** | **~4 sessions** |

Server auto-derives this at boot (`src/models/runtime/capacity.py`, `auto_engine_config`).

---

All plots in `benchmarks/results/plots_fused/` · engine bench in `benchmarks/engine_bench.py` · capacity math in `src/models/runtime/capacity.py`.