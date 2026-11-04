# TORUN — Bonsai 2 one-example evals

Run from the repo root. Bonsai 2 uses the local Prism-fork
`llama-server` and cached GGUF; `--backend bonsai` starts and stops the
sidecar automatically.

## Before running

Bonsai needs roughly 7GB RAM plus VRAM. The live VoiceAgent server
usually holds ~1GB, so stop it first if you need headroom:

```bash
pkill -f '[s]rc.server'
```

Optional preflight (binary + weights):

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from src.models.bonsai import resolve_server_bin, resolve_gguf
print("server:", resolve_server_bin())
print("gguf:  ", resolve_gguf())
PY
```

## Both suites, one example each

`--limit` applies per suite: one GAIA row and one TB 2.1 row.

```bash
PYTHONPATH=. .venv/bin/python -u benchmarks/harness/eval.py \
  --suite all --limit 1 --backend bonsai \
  --timeout 1800 --tag bonsai-one-each
```

This picks the first GAIA row (L2) and the first TB 2.1 row (easy).
GAIA is graded locally; TB 2.1 is trajectory-only locally.

## GAIA, known-simple L1 example

Runs `gaia-2d83110e` (expected `Right.`):

```bash
PYTHONPATH=. .venv/bin/python -u benchmarks/eval_agent_tasks.py \
  --tasks benchmarks/harness/gaia.jsonl \
  --offset 34 --limit 1 --backend bonsai \
  --timeout 1800 --tag bonsai-gaia-one
```

## Terminal-Bench 2.1, one easy example

```bash
PYTHONPATH=. .venv/bin/python -u benchmarks/eval_agent_tasks.py \
  --tasks benchmarks/harness/tb21.jsonl \
  --limit 1 --backend bonsai \
  --timeout 1800 --tag bonsai-tb21-one
```

This records the trajectory and reports `verifier_pending`: the local
harness cannot grade TB 2.1. For an official score, run the task’s
`docker_image` verifier under Harbor (`benchmarks/kaggle.md` §3).

For a real local pass/fail without Harbor, use the 5-task terminal
smoke instead:

```bash
PYTHONPATH=. .venv/bin/python -u benchmarks/eval_agent_tasks.py \
  --tasks benchmarks/terminal_bench.json \
  --limit 1 --backend bonsai \
  --timeout 1800 --tag bonsai-tb-smoke
```

## Results

- `benchmarks/results/harness_all_<tag>_<ts>.json`
- `benchmarks/results/harness_gaia_<tag>_<ts>.json`
- `benchmarks/results/harness_tb21_<tag>_<ts>.json`
- `benchmarks/results/agent_tasks_<tag>_<ts>.json`

Bonsai on this laptop will be much slower than Groq; raise `--timeout`
if a task is still progressing.
