# Agent benchmarks on Kaggle (Bonsai leg, T4)

Groq runs are cheap and fast but rate-limited and nondeterministic across
runs; Bonsai (local 27B ternary sidecar) is the leg worth measuring once.
Run it on Kaggle (T4 16GB, free tier): local GPU is down and the GGUF +
server binary need a fresh download anyway.

## 0. Notebook setup

- Accelerator: GPU T4 x2, internet ON.
- Cells run top to bottom, one at a time (single tenant per card).

```bash
# cell 1: code
!git clone https://github.com/Se00n00/VoiceRT.git
%cd VoiceRT/voice-pipeline
# cell 2: env (~10 min, torch CUDA build dominates)
!pip install -r requirements.txt
# cell 3: preconditions (all must print OK)
!PYTHONPATH=. python -c "import torch; print(torch.cuda.is_available())"
```

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
!PYTHONPATH=. VOICE_GUARD=0 python -u benchmarks/eval_agent_tasks.py \
    --tasks benchmarks/gaia_l1.json --tag kagdhi-bonsai-smoke \
    --backend bonsai --limit 1 --timeout 900
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
# 1. download + adapt (writes benchmarks/gaia.jsonl)
!PYTHONPATH=. python -u benchmarks/gaia_adapter.py
```

Then (task attachments copy per task, oracles are GAIA-exact reply_match):

```bash
!PYTHONPATH=. python -u benchmarks/eval_agent_tasks.py \
    --tasks benchmarks/gaia.jsonl --tag kaggle-bonsai-gaia-full \
    --backend bonsai --timeout 900
# per level, e.g. L1 only:
!PYTHONPATH=. python -u benchmarks/eval_agent_tasks.py \
    --tasks benchmarks/gaia.jsonl --tag kaggle-bonsai-gaia-l1 \
    --backend bonsai --timeout 900 --level 1
!PYTHONPATH=. python -u benchmarks/eval_agent_tasks.py \
    --tasks benchmarks/terminal_bench.json --tag kaggle-bonsai-tb \
    --backend bonsai --timeout 900
```

165 tasks at Bonsai speed: run per level (L1/L2/L3) in separate sessions
(12h cap). Results carry `by_level` (L1/L2/L3 pass counts) in `summary`.

Reference (Groq, 2026-10-10, 5-task L1-style manifests, not comparable):
GAIA-L1 5/5 effective, TB 5/5. Full-validation numbers are unmeasured —
record them from `benchmarks/results/agent_tasks_<tag>_*.json`.

## 3. DRACO (100 deep-research tasks, Bonsai-graded)

Grading uses the same Bonsai leg (`--judge-backend` defaults to the agent
backend), so the score measures one setup — never agent-vs-judge mixing.

```bash
!PYTHONPATH=. python -u benchmarks/draco_eval.py \
    --backend bonsai --judge-backend bonsai --limit 3 \
    --tag kaggle-bonsai-draco-sample \
    --timeout 2400 --max-tokens 8192 --pause 10
```

Then, only if the sample looks sane (reports >500 chars, judge parses):

```bash
!PYTHONPATH=. python -u benchmarks/draco_eval.py \
    --backend bonsai --limit 25 --offset 0 --tag kaggle-bonsai-draco \
    --timeout 2400 --max-tokens 8192 --pause 10
!PYTHONPATH=. python -u benchmarks/draco_eval.py \
    --backend bonsai --limit 25 --offset 25 --tag kaggle-bonsai-draco \
    --timeout 2400 --max-tokens 8192 --pause 10
# ... offsets 50, 75 (12h session cap: one chunk per session max)
```

Notes: no 429 risk on Bonsai (local), but wall-clock is ~10-20x Groq
(measure per-task seconds on the sample first). Judge runs on the same
leg. Scores are judge- and setup-dependent — compare only within this
setup, never against Groq-judged numbers.

## 4. After the runs

- Download `benchmarks/results/agent_tasks_*.json` and `draco_*.json`.
- Record: pass rates, mean steps-to-success, median seconds, DRACO mean
  score + per-domain breakdown — all in the results JSON `summary`.
- Groq-side reference files live next to them (`*-groq*` tags).
