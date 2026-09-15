# Benchmarks

Rigorous PyTorch vs Triton benchmark suite for the voice-pipeline inference engine.

Covers **LLM (Qwen2.5-0.5B-Instruct)**, **STT (Whisper-base)** and **TTS (Kokoro-82M)** + individual Triton kernels, with kernel-level parity checks, CUDA-event timing, peak-VRAM, concurrency, roofline and Nsight profiling.

## 1. Hardware

Measured on RTX 3050 Laptop 4 GB (3768 MB, CUDA 12.4, `nvidia-smi` 12.4) or fallback CPU.  
`benchmarks/common/environments.py:1` captures `GPU name`, `GPU memory`, `CUDA`, `PyTorch`, `Triton`, `Python`, `model`, `dtype`, `batch`, `shapes`, `warmup/iterations`, `timestamp` for every run.

## 2. Software environment

- `torch==2.5.1+cu124`, `triton==3.1.0`, `transformers==5.17.0`, `Python 3.12`
- All benchmarks pin `PYTHONPATH=.` and `seed 42` (`benchmarks/common/benchmark.py:12`). Deterministic where possible; stochastic sampling disabled/fixed.

## 3. Benchmark methodology

- **Timing**: `benchmarks/common/timing.py:12` — `torch.cuda.Event` with `synchronize()`; never wall-clock around async CUDA without sync. `measure_latencies(fn)` does `warmup` (default 20, Triton compile excluded) then `iterations` timed launches.
- **Warmup / iterations**: `--warmup`/`--iterations` CLI (kernels 100, models 30 default). Enough for stable median/p95/p99.
- **Exclusions**: model load, tokenizer init, CUDA init, Triton JIT compile excluded (warmup). Only inference timed.
- **Output**: machine-readable `benchmarks/results/*.csv` + `*.json` per spec with `median/mean/p50/p95/p99/min/max/peak_throughput/timestamp`.

## 4. PyTorch baseline

For every Triton kernel there is an exact PyTorch reference in `src/models/triton_kernels/{qwen,whisper,tts}.py` falling back when `HAVE_TRITON_KERNELS==False` or `not _cuda(x)`:

- LLM: `rmsnorm=torch.rsqrt(var+eps)*w`, `rope=cat(x1*c-x2*s, x1*s+x2*c)`, `swiglu=silu*up`, `gqa=repeat_interleave+softmax`, `fused_qkv=3×F.linear`
- STT: `layernorm=F.layer_norm`, `row_softmax=F.softmax`, `decode_attn=einsum+softmax`, `fused_qkv=F.linear×3`
- TTS: `conv1d_silu=F.silu(F.conv1d)`, `in1d_silu=F.silu(instance_norm)`

`benchmarks/llm/benchmark_pytorch.py:4` forces `qwen_surface.HAVE_TRITON_KERNELS=False` before engine import; `benchmarks/stt/*` and `tts/*` do the same. Triton variants leave `HAVE_TRITON_KERNELS=True`.

## 5. Triton implementation

Hand-written kernels `src/models/triton_kernels/{activation,attention,conv1d,layernorm,rmsnorm,rope,softmax}.py` imported via surfaces `src/models/triton_kernels/{qwen,whisper,tts}.py:27`. All kernels have `HAVE_TRITON_KERNELS` guard and `try/except` fallback to torch.

Only kernels that exist are benchmarked (`benchmarks/{llm,stt,tts}/benchmark_kernels.py` registries).

## 6. Correctness methodology

Before timing, each benchmark runs numerical validation `benchmarks/common/metrics.py:22`:

- `max_abs_error`, `mean_abs_error`, `relative_error`, `cosine_similarity`, `has_nan/has_inf`
- Tolerance `--atol/--rtol` (default `1e-3`). Fails benchmark if outside tolerance (`fail_if_not_correct`).

Kernels: direct tensor compare (same dtype/shape/weights). Full models: deterministic logits / waveform shape compare; LLM sampling disabled (greedy), STT/TTS fixed seed.

Results: `*correctness.json`.

## 7. LLM metrics

Model `Qwen/Qwen2.5-0.5B-Instruct` (`benchmarks/llm/_shared.py:1`).

- **Prefill vs Decode** measured separately via `engine.generate_stream` per-token timestamps. Not mixed.
- Input `64,128,256,512,1024,2048`; batch `1,2,4,8`; output `32,64,128`. Default sweep `--seq-lengths`/`--batch-sizes`.
- Per run: `TTFT` (request→first token), `TPOT` (`(E2E-TTFT)/(N-1)`, not `E2E/N`), `ITL` distribution, `E2E` (request→last token), `output tok/s`, `total tok/s`, `req/s`, `peak VRAM` (`torch.cuda.max_memory_allocated` after `reset_peak_memory_stats`, `benchmarks/common/memory.py:6`).
- CSVs: `results/llm_latency.csv` (`median/p50/p95/p99/min/max, TTFT/TPOT/E2E`), `llm_throughput.csv`, `llm_memory.csv`, `llm_correctness.json`.

Concurrency (`benchmarks/llm/benchmark_concurrency.py:1`): `1,2,4,8,16` concurrent requests via `asyncio.gather` on `engine.generate` in thread executor; reports `concurrency, request/output-token throughput, TTFT/TPOT/E2E p50/p95, peak VRAM` → `llm_concurrency.csv`.

## 8. STT metrics

Whisper-base (`benchmarks/stt/_shared.py:1`).

- Not autoregressive: no TTFT/TPOT. Measures `preprocessing` (mel frontend, CPU, reported separately), `encoder`, `decoder`, `postprocessing`, `total`, `RTF=infer/audio_dur` (lower better), `audio_s_per_s=total_audio/total_time`, `peak VRAM`.
- Audio `1,5,10,30,60 s` synthetic (`_synthetic_audio`), batch `1,2,4,8`.
- Same audio/preprocessing for PyTorch/Triton compare; CPU preprocessing not hidden.
- CSVs: `stt_latency.csv` (`median/p50/p95/p99, per_audio_ms, RTF`), `stt_throughput.csv` (`audio_s_per_s`), `stt_memory.csv`, `stt_correctness.json`.
- Kernels: `layernorm, row_softmax, decode_attn, batched_decode_attn, fused_qkv`.

## 9. TTS metrics

Kokoro-82M (`benchmarks/tts/_shared.py:1`).

- Measures `text preprocessing`, `token encoding`, `model inference`, `vocoder/waveform`, `total`, `peak VRAM`, `audio throughput`, `RTF=synth_time/audio_dur`.
- Text `10,25,50,100,200` tokens, batch `1,2,4,8`. Identical text/weights/config/dtype; stochastic sampling disabled (greedy, fixed seed).
- Reports `tokens/s`, `audio s/s`, `p50/p95/p99`, `RTF`.
- CSVs: `tts_latency.csv`, `tts_throughput.csv`, `tts_memory.csv`, `tts_correctness.json`.
- Kernels: `conv1d_silu, in1d_silu` (post-filter); plain `conv1d` stays cuDNN (0×).

## 10. Kernel benchmarks

`benchmarks/{llm,stt,tts}/benchmark_kernels.py` — one launch per kernel, `warmup 20, iterations 100` (small kernels) for stable median. Compares `PyTorch` vs `Triton` same inputs/dtype/shapes.

Likely kernels (only those present): `RMSNorm, RoPE/batched, SwiGLU, GQA-attention, fused QKV, layernorm, row_softmax, decode_attn, conv1d_silu, in1d_silu, lstm_cell`.

## 11. Profiling methodology

Scripts, not auto-run on every bench:

- **Nsight Systems** (`benchmarks/profiling/nsight_systems/profile.py:1`): `nsys profile --trace=cuda,nvtx,osrt --stats=true -o profiling/nsight_systems/<model> python -m benchmarks.<model>.benchmark_*`. Profiles complete inference — kernel launch overhead, CPU/GPU sync, idle gaps, sequencing, H2D/D2H, concurrency, preprocessing.
- **Nsight Compute** (`benchmarks/profiling/nsight_compute/profile.py:1`): `ncu --set full -o profiling/nsight_compute/<kernel> python -c "...kernel call..."` plus metrics `--metrics sm__warps_active,sm__throughput,dram__throughput,l1tex, FLOP, mem, regs, shared, stalls, instr`. Detects unsupported metrics per GPU.

Commands listed via `python -m benchmarks.profiling.nsight_systems.profile --list` etc. See `benchmarks/profiling/*/profile.py`.

## 12. Roofline methodology

`benchmarks/common/roofline.py:1`:

- For each kernel: `Arithmetic Intensity = FLOPs / bytes`, `FLOPs`/`bytes` estimated analytically (e.g., `rmsnorm: B·N·5`, `gqa: H·(4·N·D+5N)`), `achieved TFLOP/s = FLOPs / median_s /1e12`, `achieved BW = bytes / s /1e9`.
- Produces `results/roofline.csv` `{kernel,flops,bytes,AI,achieved_tflops,achieved_bw}` and `results/plots/roofline_*.png`.
- Distinguishes `memory-bound` vs `compute-bound` via `min(peak_flops, peak_bw·AI)`; labels theoretical vs measured. Specs: `HW_SPECS` for `rtx3050_laptop (14 TFLOPs fp16, 224 GB/s)` and `a100`.

## 13. Plots

`benchmarks/common/plotting.py:1` via `matplotlib` (Agg), auto from CSVs → `results/plots/`:

- LLM: `TTFT vs input`, `TPOT vs batch`, `throughput vs concurrency`, `peak VRAM vs seq len`
- STT: `latency vs duration`, `RTF vs duration`, `throughput vs batch`, `peak VRAM vs duration`
- TTS: `latency vs tokens`, `RTF vs tokens`, `throughput vs batch`, `peak VRAM vs tokens`
- Kernels: `PyTorch vs Triton latency`, `speedup`, `roofline`

Run: `python -m benchmarks.common.plotting` or after any bench.

## 14. CLI

```bash
python -m benchmarks.llm.benchmark_pytorch  --input-length 512 --output-length 64 --batch-size 1 --warmup 20 --iterations 30 --output-dir benchmarks/results --seed 42
python -m benchmarks.llm.benchmark_triton   --input-length 512 --output-length 64 --warmup 20 --iterations 30
python -m benchmarks.llm.benchmark_kernels  --dtype fp16 --warmup 20 --iterations 100
python -m benchmarks.llm.benchmark_concurrency --concurrency 1 2 4 8 16

python -m benchmarks.stt.benchmark_pytorch  --audio-duration 5 --batch-size 1
python -m benchmarks.stt.benchmark_triton
python -m benchmarks.stt.benchmark_kernels

python -m benchmarks.tts.benchmark_pytorch  --text-length 50 --batch-size 1
python -m benchmarks.tts.benchmark_triton
python -m benchmarks.tts.benchmark_kernels

# Plots & summary
python -c "from benchmarks.common.plotting import generate_all; from pathlib import Path; generate_all(Path('benchmarks/results'))"
python -c "from benchmarks.common.summary import generate_summary; from pathlib import Path; generate_summary(Path('benchmarks/results'))"

# Profiling
python -m benchmarks.profiling.nsight_systems.profile --list
python -m benchmarks.profiling.nsight_compute.profile --list
# Example:
# nsys profile --trace=cuda,nvtx,osrt --stats=true -o profiling/nsight_systems/llm_triton python -m benchmarks.llm.benchmark_triton --warmup 5 --iterations 20
# ncu --set full -o profiling/nsight_compute/rmsnorm python -c "import torch; from src.models.triton_kernels.rmsnorm import rmsnorm; x=torch.randn(4,1024,device='cuda'); w=torch.randn(1024,device='cuda'); rmsnorm(x,w)"
```

Model-specific: `--model`, `--dtype {fp32,fp16,bf16}`, `--device {cuda,cpu}`, `--seq-lengths`, `--durations`, `--token-lengths`, `--atol/--rtol`.

## 15. Results

All under `benchmarks/results/`:

- `llm_latency.csv`, `llm_throughput.csv`, `llm_memory.csv`, `llm_correctness.json`, `llm_kernels_*`, `llm_concurrency.csv`
- `stt_*`, `tts_*`, `roofline.csv`, `summary.json/csv`, `plots/*.png`

Raw measurements retained for reproducibility.

## 16. Known limitations

- Triton compile excluded (warmup), but first-time `torch.compile` (Kokoro) is broken in this env (eager by decision, `src/models/engines/tts.py:181`).
- GPU util via `nvidia-smi 2Hz` lower bound for sub-second bursts.
- Synthetic weights used if `HF` snapshot not cached/offline (marked `synthetic` in correctness); real weights require `python -m src.models.download`.
- CPU fallback loses GPU kernel win (speedup ~1×).
- Roofline FLOPs/bytes are analytical estimates, not profiler-measured; peak specs approximated for laptop 3050.
