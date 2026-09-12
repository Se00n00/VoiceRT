# Capacity model

How the server decides, at every boot, how many sessions it can serve —
and how overflow is queued. Implementation: `runtime/capacity.py`
(estimator, CPU-testable) + `server/app.py` (startup wiring) +
`server/routes.py` (queue + exposure). Tests:
`tests/runtime/test_capacity.py`.

## 1. The pipeline

```
probe_vram()  →  {total_mb, free_mb, cuda, source, name}
     │              nvidia-smi ──▶ torch.cuda ──▶ zeros(CPU)
     ▼
warm engine   →  baseline_mb = max_allocated_mb()   (~1807 MB here)
     ▼
estimate_session_mb(genlen) → per_session_mb        (~206 MB @48)
     ▼
plan_capacity(total, baseline, …) → plan
     │  usable = total − baseline − max(10% × total, 400MB)
     │  max_sessions = clamp(usable / per_session, 1, 1000)
     ▼
configure_capacity(plan) → scheduler + session store + /health
```

`probe_vram` never raises; unknown GPUs yield `total 0` → a conservative
plan (`max_sessions 1`, `conservative: true`) instead of a crash.

## 2. Pricing one session

`estimate_session_mb(max_new_tokens=48, …)` returns `{total_mb, breakdown}`:

| Component | Formula | @48 tokens |
|---|---|---|
| KV-cache share | `2 × 24 × 2 × 512 × 64 × 2B` | 6.0 MB |
| History working set | `20 turns × (200 + 48) tok × 896 × 2B` | 8.5 MB |
| Transient scratch | `per_turn_mb` (TTS + mel) | 150.0 MB |
| **Total** | `(sum) × 1.25 safety` | **≈ 205.6 MB** |

Notes for the skeptical reader:

- The Qwen KV-cache is statically preallocated per engine
  (`models/qwen.py:KVCache`), not per session — pricing it 1:1 per busy
  session is a deliberate over-estimate (safety > cleverness on 4 GB).
- History assumes the worst case: full 20-turn sessions at max tokens.
  Real usage is usually lighter, so the plan under-promises.
- `prompt_tokens=200` covers system prompt + history headroom.

## 3. Worked example (RTX 3050 Laptop, 4096 MB)

```
total 4096 − baseline 1807 − headroom max(409.6, 400) = usable 1879.4
genlen  48: 1879.4 / 205.6 = 9.14 → 9 sessions
genlen 128: 1879.4 / 209.0 = 8.99 → 8 sessions
genlen 256: fewer still (history term grows linearly)
```

Property tests pin the shape of this curve, not just points:
longer generations never yield *more* sessions
(`test_longer_genlen_fewer_sessions`), unknown VRAM yields exactly 1
(`test_plan_unknown_vram_conservative`).

## 4. Queueing

`FIFOScheduler(max_concurrency=4)` (`runtime/scheduler.py`):

- Tickets preserve arrival order; a thread proceeds only when its ticket
  is head-of-queue **and** a slot is free — no thundering-herd.
- `acquire(blocking=True, timeout=queue_timeout_s)`; timeout → HTTP 503 +
  `Retry-After: 2`, recorded as `queue_timeout` in `/metrics`.
- Every admission records its wait in `queue_wait` — advertised latency
  includes queueing, not just GPU time.
- Stats (`pending/running/admitted_total`) stream into `/metrics`.

Two different caps, two different resources:

| Cap | Value here | Guards | Resource |
|---|---|---|---|
| `max_sessions` | 9 | stored conversations (text, RAM) | RAM + relevance |
| `max_inflight` | 4 | concurrent turns (weights + scratch) | VRAM + SMs |

## 5. Retuning (no code changes)

All knobs live in `configs/pipeline.yaml → capacity:`:

```yaml
capacity:
  per_turn_mb: 150
  headroom_frac: 0.10
  headroom_min_mb: 400
  safety_factor: 1.25
  sessions_cap: 1000
  max_inflight: 4       # re-measure before raising (c=4 halves throughput)
  queue_timeout_s: 10
```

And `streaming.max_tokens` (generation length) re-prices sessions
automatically at next boot. Verify any change with:

```bash
PYTHONPATH=. python -m unittest tests.runtime.test_capacity -v
PYTHONPATH=. python scripts/benchmark.py capacity --quick
```
