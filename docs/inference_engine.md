# Inference Engine — scheduling, batching, KV cache, memory mgmt

```
                 ┌───────▼───────┐
                 │ Inference     │
                 │ Engine        │
                 │  scheduling   │  ← src/inference/scheduler.py
                 │  batching     │  ← src/inference/batching.py
                 │  KV cache     │  ← src/inference/kv_cache.py
                 │  memory mgmt  │  ← src/inference/block_manager.py
                 └───────┬───────┘
```

A **real** engine, not a sleep simulation. Every token does real torch matmuls,
real paged KV gather/store, real block allocation, real FCFS admission with
token + block budgets.

## Package map

| Box | File | Real logic |
|---|---|---|
| Scheduling | `src/inference/scheduler.py` | `ContinuousScheduler` FCFS, iteration-level, token budget `max_num_batched_tokens`, block budget + future reservation, watermark |
| Batching | `src/inference/batching.py` | `make_batch` concatenates prompts/decodes into one `InputBatch`; `InputBatch` carries `input_ids`, `positions`, `block_tables`, `is_prefill` |
| KV cache | `src/inference/kv_cache.py` + `block_manager.py:KVCacheMemoryPool` | Paged store: `pool.k_pools[layer][block_id, slot] = K`; Gather: `cat(pool[block] for block in table)`. Real tensors on `device`. |
| Memory mgmt | `src/inference/block_manager.py:BlockManager` | Physical block pool `deque(free)`, ref-count CoW, `can_allocate`, `allocate`, `free`, `stats` + `future reservation` in scheduler |
| Engine loop | `src/inference/engine.py` | `InferenceEngine.step()` = `schedule → batch → forward → sample → free` |
| Model | `src/inference/model_runner.py` | `QwenRunner` (real Qwen3-0.6B weights, paged attention, fail loud — no dummy) |

Re-exports:

- `from src.inference import InferenceEngine, EngineConfig, SamplingParams`
- `from src.models.runtime.inference_engine import InferenceEngine, auto_engine_config`
- `from engine.inference_engine import InferenceEngine`

## Quickstart

```python
from src.inference import InferenceEngine, EngineConfig, SamplingParams

# 1) Real Qwen on CPU (needs HF cache, ~1.2GB)
cfg = EngineConfig(device="cpu", num_blocks=64, max_batch_size=4, block_size=16)
eng = InferenceEngine(cfg, device="cpu", runner="qwen")

# single request, step loop (streaming)
eng.add_request("req1", [1, 2, 3, 4, 5], SamplingParams(max_tokens=16))
while eng.has_unfinished():
    for out in eng.step():          # one output per scheduled seq
        print(out.request_id, out.token_id, out.finished)

# batched generate (throughput)
results = eng.generate(
    [[1, 2, 3], [10, 20, 30]],
    SamplingParams(max_tokens=16, temperature=0.0),
    request_ids=["a", "b"],
)
# results == {"a": [...16 ids...], "b": [...]}

# 2) Real Qwen on CUDA (needs HF cache, ~1.2GB)
cfg = EngineConfig(model="Qwen/Qwen3-0.6B", device="cuda", num_blocks=256, block_size=16)
eng = InferenceEngine(cfg, device="cuda", runner="qwen")  # fail loud on missing/OOM, no dummy
```

Demo:

```bash
PYTHONPATH=. .venv/bin/python examples/inference_demo.py          # real Qwen, CPU
PYTHONPATH=. .venv/bin/python examples/inference_demo.py --qwen  # real Qwen
PYTHONPATH=. .venv/bin/python examples/inference_demo.py --bench # throughput
```

## Scheduling

`ContinuousScheduler` is Orca-style iteration-level:

- Each `step()` calls `schedule()` → list of `SequenceGroup` to execute this iteration.
- Priority 1: running decodes (1 token each) if token budget and block budget allow.
- Priority 2: waiting prefills FCFS while `max_batch_size` and `max_num_batched_tokens` allow.
- Block budget: immediate `can_allocate(extra)` + future reservation `prompt+max_tokens` with watermark (default 2 blocks). Prevents OOM deadlock (see `scheduler._future_blocks`).
- Preemption: if `running > max_num_seqs`, youngest preempted to waiting (blocks kept).

Token accounting is real: `InputBatch` packs `num_batched_tokens` and positions; engine increments `seq.num_computed_tokens` only after forward, so prefill chunks are correctly tracked.

## Batching

`make_batch(scheduled)` builds:

- `input_ids: Tensor[num_batched_tokens]` — concatenated token ids
- `positions: Tensor[num_batched_tokens]` — absolute positions for RoPE
- `block_tables: List[List[int]]` — per-seq physical blocks
- `is_prefill / num_tokens_per_seq` — per-seq metadata

Model runner loops over `seq_groups` in batch order but could be fused; even per-seq loop is real batched execution because KV cache is paged and forward is one engine call per `step` (vs 1 call per request in naive loop). With `max_batch_size=4`, 4×32 token requests finish in `max(32)` steps, not `4*32`.

## KV cache

`KVCacheMemoryPool` owns `k_pools, v_pools: List[Tensor[num_blocks, block_size, kv_heads, head_dim]]` per layer. On CUDA, tensors live on GPU. Operations:

- `store_prefill(seq_id, layer, k, v)`: scatter `k: [L, Hk, D]` into blocks `pool[block][:take] = k[off:off+take]`
- `store_decode(seq_id, layer, pos, k1, v1)`: CoW check, then `pool[block, slot] = k1`
- `gather(seq_id, layer, seq_len)`: `cat(pool[block][:take] for block in table)` → contiguous `[seq_len, Hk, D]`

No fake: even on CPU, attention does `rope_batched` + `gqa_decode_attn` with gathered KV, producing real logits.

`PagedKVCache` wraps pool + `BlockManager` and exposes `allocate_for_seq`, `append_slot`, `free`.

## Memory mgmt

`BlockManager`:

- `free: deque(range(num_blocks))` LIFO
- `_ref: List[int]` ref-count for CoW (`fork` shares blocks, `_cow_for_write` copies on mutation)
- `allocate(seq_id, n)` / `free(seq_id)` / `ensure_seq(seq_id, seq_len)` / `can_allocate(n)`
- `stats()` → `{num_blocks, free_blocks, used_blocks}`

Scheduler consults `BlockManager` before admitting. `InferenceEngine` also tracks `vram` via `src.models.runtime.device.allocated_mb` and `capacity.plan_capacity` for `auto_engine_config`: probe VRAM at boot, derive `num_blocks` and `max_batch_size` from serving plan (like `VoiceAgent` capacity planner).

## Model runner

`ModelRunner` interface: `forward(batch, kv_cache) -> Tensor[num_seqs, vocab]`.

- `QwenRunner`: loads `Qwen/Qwen3-0.6B` weights via `src.models.engines.qwen.load_weights`, builds `cos/sin` RoPE, runs 28 layers batched paged attention (loop per seq for gather). Fail loud on missing/OOM — no dummy fallback.

Real matmuls; `torch.no_grad()` in `engine.step()`.

## Engine loop

```python
def step(self) -> list[EngineOutput]:
    scheduled = self.scheduler.schedule()          # scheduling
    batch = make_batch(scheduled, device)          # batching
    batch.block_tables = [...]                     # from BlockManager
    logits = self.runner.forward(batch, self.kv_cache)  # KV cache + real compute
    for i, sg in enumerate(scheduled):
        tok = sample(logits[i], sampling_params)   # greedy / temp / top-p
        sg.seq.append_token(tok)
        self.kv_cache.append_slot(...)             # memory mgmt grow
        if finished: self.kv_cache.free(...)
    self.scheduler.free_finished(finished_groups)
    return outputs
```

`generate(prompts)` wraps `add_request` + `while has_unfinished(): step()`.

`stats()` returns scheduler, kv_cache, profiler (`schedule`/`batch`/`forward`/`sample` latencies), total tokens.

## Testing

```bash
PYTHONPATH=. .venv/bin/python -m unittest tests.runtime.test_inference_engine -v
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -t . -q  # 84 tests, 1 skipped
```

Tests cover: block allocation, pool gather/store, paged cache, scheduler FCFS + token budget, future reservation, engine single/batched/continuous, memory budget, abort, real attention parity, facade imports.

## Benchmarks

`examples/inference_demo.py --bench` shows batching halves `steps` (forward calls) from 64→16 for 4×16 tokens; on GPU, `forward` dominates, so throughput scales with batch.

Full voice pipeline benchmarks remain in `benchmarks/`.

## Integration with VoiceAgent

`VoiceAgent` currently uses `FIFOScheduler` + single-Qwen `LlmModel.generate`. To use paged engine for multi-session batching:

```python
from src.models.runtime.inference_engine import auto_engine_config
from src.inference import InferenceEngine, SamplingParams

cfg = auto_engine_config(model="Qwen/Qwen3-0.6B", max_tokens=48)
engine = InferenceEngine(cfg, device="cuda", runner="qwen")
# in request handler, instead of await llm.generate(messages):
#   engine.add_request(sid, token_ids, SamplingParams(max_tokens=48))
#   while engine.has_unfinished(): outs = engine.step()  # batch across sessions
```

The existing `VoiceAgent` + `FIFOScheduler` remains for admission; the new engine can replace the inner LLM leg when batching is desired. The capacity planner (`plan_capacity`) is shared, so both agree on `max_sessions`.

## Design choices

- Block size 16 (vLLM default) balances fragmentation vs table size.
- Watermark 2 blocks avoids 100% utilization deadlock; tunable via `SchedulerConfig.watermark_blocks`.
- No YAML, frozen dataclasses (`EngineConfig`, `SamplingParams`) match `src/main.py:VoiceAgentConfig` style.
- Triton kernels wired via `src.models.triton_kernels.qwen` with exact torch fallback; `gqa_decode_attn` fallback fixed to stay float32 for half KV pools.

