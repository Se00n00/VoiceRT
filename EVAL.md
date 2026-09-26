# EVAL.md — full-dataset eval log (newest first)

## 1. BFCL V3 single-turn subset — 2026-09-25 (first full-dataset eval)

Command (Kaggle T4, greedy):

```bash
!PYTHONPATH=. python3 -m eval.evaluate bfcl --version v3 \
  --categories multiple,parallel,parallel_multiple,irrelevance,chatable \
  --k 1 --tag kaggle-v3single
```

Model: `openbmb/MiniCPM5-1B` [minicpm], `max_tokens=320`, raw results in
`bfcl_kaggle-v3single_20260925-130723.json` (repo root, 3014 rows).

### Headline metrics

| metric | value |
|---|---|
| n | 3014 |
| pass@1 | **0.559** (1686/3014) |
| tool precision / recall / F1 | 0.559 / 0.636 / **0.595** |
| irrelevance acc (n=1322) | **0.725** |
| arg validity | 0.962 (n=1878) |
| retries | 5 (rate 0.002) |
| mean steps | 1.0 |
| median s/case | 9.29 |

### Per-category

| category | passed | rate |
|---|---|---|
| multiple (1252) | 728 | 0.581 |
| parallel (216) | 0 | **0.000** |
| parallel_multiple (224) | 0 | **0.000** |
| irrelevance (1122) | 836 | 0.745 |
| chatable (200) | 122 | 0.610 |

### Issues found (diagnosed from rows, not guessed)

1. **`parallel` / `parallel_multiple` are structurally unwinnable (0/440).**
   The contract is one call per step (preamble "exactly one … action",
   `stop=["</function>"]`, single-capture parsers in
   `src/tools/terminal.py`), the runner takes one greedy step
   (`mean_steps=1.0`), and 0/440 raws contain 2+ call envelopes
   (203/216 parallel cases emit exactly 1 call, 13 emit 0).
   These categories require 2+ calls in one turn, so all 440 fail with
   fp=1/fn=1. Sample `parallel_0`: model narrates "I'll make both
   calls", emits only the Taylor Swift call, Maroon 5 never comes.
   Fix direction: multi-call envelope or sequential call-again loop in
   the runner; until then report single-call categories separately so
   440 auto-fails don't mask real progress.
2. **False-positive calls on negative controls (364 total).**
   286 irrelevance + 78 chatable fails, all with a parsed call.
   `irrelevance_5` (rectangle perimeter): the thinking trace itself
   says "I should not use the quadratic equation function" — then
   calls `solve_quadratic_equation` anyway. Chatable sample:
   confabulated `math_sqrt` with garbage bare-tail args. Sampled-57
   irrelevance was 10/10; at n=1322 the acc is 0.725 and precision
   sinks to 0.559 (1328 FP vs 964 FN). The anti-FP side needs work
   (abstention prompt and/or stricter no-call threshold).
3. **`multiple` misses are no-calls + wrong picks (58%).**
   122/1252 emit zero calls (narration), the rest mostly wrong-tool /
   wrong-args. Same narration-without-calling seen in the GAIA agent
   loop.
4. **Retry path is dead (5/3014).** `_needs_retry` fires only on
   garble/echo — wrong-tool and no-call never get a second chance.
5. **Manifest drift vs HF `main`.** The run pulled
   `BFCL_v3_live_irrelevance.json` (882 cases) which is NOT in the
   committed `benchmarks/bfcl_v3_full.jsonl.gz` (my pre-run estimate
   was 2132; actual n=3014, delta = +882 irrelevance). 1 GT-less
   orphan dropped from `BFCL_v3_live_multiple.json`
   (`live_multiple_1052-79-0`). Refresh the full manifest before the
   next big run or keep pinning `--hf-rev`.
