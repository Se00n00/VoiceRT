# Capacity model

How many sessions fit on the GPU — and how overflow is queued.
Implementation: `src/models/runtime/capacity.py` (estimator, CPU-testable;
 knobs are `VoiceAgentConfig` fields, no YAML). Admission itself lives in
`src/main.py` (`FIFOScheduler` + `check_budget` per turn). Tests:
`tests/runtime/test_capacity.py`.

## 1. The pipeline

```
probe_vram()  →  {total_mb, free_mb, cuda, source, name}
     │              nvidia-smi ──▶ torch.cuda ──▶ zeros(CPU)
     ▼
warm legs     →  baseline_mb = max_allocated_mb()   (~2290 MB here)
     ▼
estimate_session_mb(genlen) → per_session_mb        (~270 MB @48)
     ▼
usable = total − baseline − max(10% × total, 400MB);
turns beyond max_inflight=4 wait in the FIFO queue (timeout → turn
`error` event, counted in /metrics)
```

`probe_vram` never raises; unknown GPUs yield `total 0` → a conservative
plan (`max_sessions 1`, `conservative: true`) instead of a crash.

## 2. Pricing one session

`estimate_session_mb(max_new_tokens=48, …)` returns `{total_mb, breakdown}`:

| Component | Formula | @48 tokens |
|---|---|---|
| KV-cache share | `2 × 28 × 8 × 512 × 128 × 2B` | 56.0 MB |
| History working set | `20 turns × (200 + 48) tok × 1024 × 2B` | 9.7 MB |
| Transient scratch | `per_turn_mb` (TTS + mel) | 150.0 MB |
| **Total** | `(sum) × 1.25 safety` | **≈ 269.6 MB** |

Notes for the skeptical reader:

- The Qwen KV-cache is statically preallocated per engine
  (`models/qwen.py:KVCache`), not per session — pricing it 1:1 per busy
  session is a deliberate over-estimate (safety > cleverness on 4 GB).
- History assumes the worst case: full 20-turn sessions at max tokens.
  Real usage is usually lighter, so the plan under-promises.
- `prompt_tokens=200` covers system prompt + history headroom.

## 3. Worked example (RTX 3050 Laptop, 4096 MB, Qwen3-0.6B stack)

```
total 4096 − baseline 2290 − headroom max(409.6, 400) = usable 1396.4
genlen  48: 1396.4 / 269.6 = 5.18 → ~5 sessions
genlen 128: history term grows linearly → fewer still
```

Property tests pin the shape of this curve, not just points:
longer generations never yield *more* sessions
(`test_longer_genlen_fewer_sessions`), unknown VRAM yields exactly 1
(`test_plan_unknown_vram_conservative`).

## 4. Queueing

`FIFOScheduler(max_concurrency=4)` (`src/models/runtime/scheduler.py`):

- Tickets preserve arrival order; a thread proceeds only when its ticket
  is head-of-queue **and** a slot is free — no thundering-herd.
- `acquire(blocking=True, timeout=queue_timeout_s)`; timeout → a turn
  `error` event (`server saturated (…)`, counted in `/metrics`).
- Every admission records its wait — advertised latency includes
  queueing, not just GPU time.
- Stats (`pending/running/admitted_total`) stream into `/metrics`.

Two different caps, two different resources:

| Cap | Value here | Guards | Resource |
|---|---|---|---|
| session store | 1000 max, 30-min TTL | stored conversations (text, RAM) | RAM + relevance |
| `max_inflight` | 4 | concurrent turns (weights + scratch) | VRAM + SMs |

## 5. Retuning (no YAML — dataclass fields)

All serving knobs are `VoiceAgentConfig` fields (`per_turn_mb`,
`vram_budget_mb`, `max_inflight`, `queue_timeout_s`, `max_sessions`,
`session_ttl_s`); estimator constants (`headroom_*`, `safety_factor`)
live in `src/models/runtime/capacity.py` next to the math:

```python
from src.main import VoiceAgent, VoiceAgentConfig

agent = VoiceAgent(VoiceAgentConfig(max_inflight=4, queue_timeout_s=10))
```

And `llm.max_tokens` (generation length) re-prices sessions. Verify any
change with:

```bash
PYTHONPATH=. python -m unittest tests.runtime.test_capacity -v
```
