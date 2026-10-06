# 07 — Inference

The custom runtime. Its parser-blocking indentation bug was fixed 2026-10-03;
the suite is green and the engine bench runs again.

## 7.1 Status: fixed 2026-10-03

`src/inference/engine.py` failed to parse — the `for o in outs:` body at
lines 432-441 sat dedented at the `for`'s own level:

```
IndentationError: expected an indented block after 'for' statement on line 431
```

The failing tests, all 14 of them:

```
test_abort                          test_kv_cache_real_tensors
test_allocate_and_free              test_memory_budget_enforced
test_batched_generate_throughput    test_memory_pool_gather_store
test_block_table_for_tokens         test_paged_store_gather
test_continuous_batching_staggered  test_runtime_facade_import
test_fcfs_admission                 test_single_request_generate
test_step_produces_real_attention   test_token_budget
```

That one import failure produced all 14 errors in the Python suite
(baseline was 564 run, 550 pass, 14 error). It also blocked
`benchmarks/engine_bench.py`, which starts with
`from src.inference import InferenceEngine, EngineConfig, SamplingParams`.
The fix was the predicted 11-line re-indent (body to 20 spaces):
`import src.inference` succeeds and the suite went 634 run, 0 errors
(4 intentional skips: CUDA/live-gated tests).

## 7.2 What the re-run showed

`benchmarks/engine_bench.py --num-requests 8 --max-tokens 32` on the real
`QwenRunner`, 8 requests × 32 tokens (re-run 2026-10-05 post cold boot):

| config | tok/s | delta vs base |
|---|---|---|
| base (no features) | 58.9 | — |
| + prefix caching | 58.9 | +0.0% |
| + chunked prefill | 59.1 | +0.3% |
| + CUDA graph | 56.6 | -3.9% |
| all features | 56.3 | -4.5% |

Honest read: on this all-distinct-prompts workload the features do not pay —
prefix cache records 0 hits and CUDA graphs stay enabled-but-never-captured,
so each feature is pure overhead and base wins. See README Benchmarks for
the earlier short-burst run where all-features was +49.4%.

## 7.3 The bug (fixed)

`src/inference/engine.py:431-441` used to read:

```python
431:                for o in outs:
432:                # append to results via internal sg lookup
433:                results[o.request_id].append(o.token_id)
434:                if o.finished and o.request_id in self._req_to_sg:
...
441:                            results[o.request_id] = results[o.request_id][:-1]
```

The `for` and its entire body sat at the same indent — 16 spaces. The body
needed 20. Eleven lines, one re-indent, and the intended structure was
unambiguous: append the token, then on finish capture the sequence's final
`output_token_ids` and strip a trailing EOS when `stop_token_ids` contains it
and `ignore_eos` is false.

`AGENTS.md` §4.6 marked this file **do not touch unless asked**; the fix was
requested 2026-10-03 and applied as predicted.

## 7.4 What the runtime does

For the record, since the code is the design:

**Paged KV cache.** Blocks are pooled and a block table maps sequence →
physical blocks. `MemoryPool` gather/store, `test_memory_budget_enforced` and
`test_paged_store_gather` cover it.

**FCFS admission.** `test_fcfs_admission` — first come first served, matching
the turn scheduler's philosophy ([06-processes.md](06-processes.md)).

**Continuous batching.** `test_continuous_batching_staggered`,
`test_batched_generate_throughput`, `test_step_produces_real_attention` — steps
run over whatever is live, and a new request joins the next step rather than
waiting for a drain.

**Stall detection.** The loop that owns line 431 raises after 500 consecutive
empty steps rather than spinning forever, and names the likely cause in the
message:

```python
raise RuntimeError(
    "inference stalled: scheduler admitted nothing for 500 steps "
    f"(prompt lens {[len(p) for p in prompts]}, max_tokens={sp.max_tokens}, "
    f"blocks={self.kv_cache.config.num_blocks}); "
    "prompt+max_tokens likely exceeds the KV block pool"
)
```

That is the design in miniature: a wedged thread plus poisoned request ids would
take out every later call, so it fails loudly instead. The `finally` block at
`engine.py:442+` guarantees blocks are freed and request ids cleaned on every
path, including that one.

## 7.5 Deliberately not used

The custom runtime is **not** on the live path. `LlmModel` runs fused or as a
llama.cpp sidecar ([05-models.md](05-models.md)).

Kept and documented as dead surface in `docs/architecture.md` §8:
`runtime/{stream,graph,event,allocator,batcher,request}.py`, the unused
`MemoryBudget`, and three orphan Triton kernels.

Kept on the live path because each was measured: eager TTS (8× faster than
dynamic compile), ONNX VAD (RTF 0.01), cuDNN conv (0.00× Triton speedup).

## 7.6 See also

- [05-models.md](05-models.md) — what actually runs
- [06-processes.md](06-processes.md) — where the runtime would be hosted
