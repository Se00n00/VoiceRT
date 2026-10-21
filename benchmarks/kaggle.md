# Agent benchmarks on Kaggle (Bonsai leg, T4)

Groq runs are cheap and fast but rate-limited and nondeterministic across
runs; Bonsai (local 27B ternary sidecar) is the leg worth measuring once.
Run it on Kaggle (T4 16GB, free tier): local GPU is down and the GGUF +
server binary need a fresh download anyway.

## 0. Notebook setup

- Accelerator: GPU T4 x2, internet ON.
- Cells run top to bottom, one at a time (single tenant per card).
- Use the Kaggle **system python3** — never a `.venv` (venvs embed
  absolute paths and don't survive copying; a broken one errors in
  `sitecustomize` before your code runs — if one exists: `rm -rf .venv`).

```bash
# cell 1: code
!git clone https://github.com/Se00n00/VoiceRT.git
%cd VoiceRT/voice-pipeline
# cell 2: env (torch CUDA build is the big download; pyproject.toml +
# uv.lock pins are verified for Python 3.13: kokoro splits by version
# marker, torch pair is 2.6.0/2.6.0, mcp-adapters is 0.3.2)
!pip install -q uv
!uv export --frozen --no-hashes --no-emit-project -o /tmp/uv-req.txt && uv pip install --system -r /tmp/uv-req.txt
# cell 3: preconditions (all must print OK)
!PYTHONPATH=. python3 -c "import torch; print(torch.cuda.is_available())"
```
All eval commands below use system `python3` (`PY=python3`), never `.venv`.

## 1. Bonsai preconditions (do these before any eval)

1. `torch.cuda.is_available()` is True (fail fast otherwise — no CPU runs).
2. GGUF auto-downloads on first warm (`prism-ml/Ternary-Bonsai-2-27B-gguf`,
   ~6GB, HF cache). Keep the notebook alive across runs.
3. Prism-fork `llama-server` binary present (`bonsai_bin="auto"` resolves
   it; stock llama.cpp refuses the weights — error message says so).
4. `VOICE_GUARD` left enforcing (default): refuses a wedged driver or a
   second tenant instead of hanging.

Smoke (one task, proves the whole eval path on Bonsai):

```bash
!PYTHONPATH=. VOICE_GUARD=0 python3 -u benchmarks/harness/eval.py \
    --suite gaia --limit 1 --tag kaggle-bonsai-smoke \
    --backend bonsai --timeout 900
```

`--model` is inert for sidecar legs (Bonsai model is fixed by the GGUF).
`--timeout` is per task; Bonsai steps run ~1 tok/s at partial offload,
so allow 900s+ per task.

## 2. GAIA full validation (L1+L2+L3, 165 tasks) + TerminalBench

GAIA validation is access-gated (one-time human step, NOT runnable here):

```bash
# 0. in a browser: accept terms at
#    https://huggingface.co/datasets/gaia-benchmark/GAIA
export HF_TOKEN=<read token with access>
# 1. download + adapt (writes benchmarks/harness/gaia.jsonl)
!PYTHONPATH=. python3 -u benchmarks/harness/eval.py --adapt gaia
```

Then (task attachments copy per task, oracles are GAIA-exact reply_match):

```bash
!PYTHONPATH=. python3 -u benchmarks/harness/eval.py \
    --suite gaia --tag kaggle-bonsai-gaia-full \
    --backend bonsai --timeout 900
# per level, e.g. L1 only:
!PYTHONPATH=. python3 -u benchmarks/harness/eval.py \
    --suite gaia --tag kaggle-bonsai-gaia-l1 \
    --backend bonsai --timeout 900 --level 1
!PYTHONPATH=. python -u benchmarks/eval_agent_tasks.py \
    --tasks benchmarks/terminal_bench.json --tag kaggle-bonsai-tb \
    --backend bonsai --timeout 900
```

165 tasks at Bonsai speed: run per level (L1/L2/L3) in separate sessions
(12h cap). Results carry `by_level` (L1/L2/L3 pass counts) in `summary`.

Reference (Groq, 2026-10-10, local smokes, not comparable):
terminal_bench 5/5. Full-validation numbers are unmeasured —
record them from `benchmarks/results/harness_*_<tag>_*.json`.

## 3. Terminal-Bench 2.1 (89 terminal tasks, verifier-graded)

Manifest is vendored from the Harbor repo (prompts only — scoring
stays with each task's verifier under harbor/docker, never local):

```bash
# 0. vendor (needs git + network; writes benchmarks/harness/tb21.jsonl:
#    89 tasks, easy-first: 4 easy / 55 medium / 30 hard)
!PYTHONPATH=. python3 benchmarks/harness/eval.py --adapt tb21
```

Sample (proves the whole eval path on Bonsai; rows report
verifier_pending while trajectories are recorded):

```bash
!PYTHONPATH=. python3 -u benchmarks/harness/eval.py \
    --suite tb21 --tag kaggle-bonsai-tb21-sample \
    --backend bonsai --limit 3 --timeout 900
```

Then, only if the sample looks sane (steps > 0, no timeouts):

```bash
!PYTHONPATH=. python3 -u benchmarks/harness/eval.py \
    --suite tb21 --tag kaggle-bonsai-tb21 \
    --backend bonsai --limit 25 --offset 0 --timeout 900
!PYTHONPATH=. python3 -u benchmarks/harness/eval.py \
    --suite tb21 --tag kaggle-bonsai-tb21 \
    --backend bonsai --limit 25 --offset 25 --timeout 900
# ... offsets 50, 75 (12h session cap: one chunk per session max)
```

Notes: no 429 risk on Bonsai (local), but wall-clock is ~10-20x Groq
(measure per-task seconds on the sample first). Local rows stay
verifier_pending by design — for true scores, run each task's
verifier from its `docker_image` (oracle field) under harbor. Scores
are setup-dependent — compare only within this setup, never against
harbor-leaderboard numbers.

## 4. After the runs

- Download `benchmarks/results/agent_tasks_*.json`.
- Record: pass rates (graded tasks), mean steps-to-success, median
  seconds, TB 2.1 verifier_pending count + trajectory stats — all in
  the results JSON `summary`.
- Groq-side reference files live next to them (`*-groq*` tags).
