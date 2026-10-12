# 07 — Inference

Retired. The custom paged runtime (`src/inference/`: paged KV cache,
FCFS admission, continuous batching, prefix cache, chunked prefill,
CUDA graphs, `QwenRunner`) is deleted, with it `benchmarks/engine_bench.py`,
the `use_paged` / `llm_paged*` flags and the `paged_attention` kernel.

Legs run fused kernels (`src/models/triton_kernels/`, torch fallback)
or llama.cpp sidecars; batching is one-request-at-a-time. The last
measured engine numbers stay in [README Benchmarks](../../README.md)
and [benchmarks.md](../benchmarks.md) for history. Numbering kept so
the series index does not shift.

## See also

- [05-models.md](05-models.md) — what actually runs
- [06-processes.md](06-processes.md) — where the legs are hosted
