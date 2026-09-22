#!/usr/bin/env bash
# Full eval sweep in one command: own tools -> BFCL V1/V2/V3 -> GAIA -> TB,
# then a consolidated results table at the end.
#
# Usage (from voice-pipeline/, GPU idle, ONE model process at a time):
#   PYTHONPATH=. bash benchmarks/run_all_evals.sh [TAG]
#   PY=/path/to/python bash benchmarks/run_all_evals.sh mytag   # custom interp
#
# Each step runs even if an earlier one fails; exit code is nonzero if
# anything failed. Every runner aborts loudly on CPU fallback (no silent
# hours-long CPU runs). Results land in benchmarks/results/.
set -u
TAG="${1:-kaggle}"
# PY override wins; else .venv if present (local), else system python3 (Kaggle).
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
export PYTHONPATH=.

FAILED=0
step() { echo; echo "===== $1 ====="; }

step "1/4 own toolset (18 cases)"
$PY -u benchmarks/toolcall_eval.py --tag "$TAG" || { echo "toolcall FAILED"; FAILED=1; }

step "2/4 BFCL V1+V2+V3, 57 cases, pass@3"
$PY -u benchmarks/bfcl_eval.py --k 3 --tag "$TAG" || { echo "bfcl FAILED"; FAILED=1; }

step "3/4 GAIA L1 agent tasks"
$PY -u benchmarks/eval_agent_tasks.py --tasks benchmarks/gaia_l1.json --tag "$TAG-gaia" || { echo "gaia FAILED"; FAILED=1; }

step "4/4 TerminalBench agent tasks"
$PY -u benchmarks/eval_agent_tasks.py --tasks benchmarks/terminal_bench.json --tag "$TAG-tb" || { echo "tb FAILED"; FAILED=1; }

echo
echo "================ FINAL RESULTS ($TAG) ================"
$PY - benchmarks/results "$TAG" <<'EOF'
import glob
import json
import os
import sys

resdir, tag = sys.argv[1], sys.argv[2]


def latest(*pats):
    best = None
    for pat in pats:
        for f in glob.glob(os.path.join(resdir, pat)):
            if best is None or os.path.getmtime(f) > os.path.getmtime(best):
                best = f
    return best


def show(label, path, keys):
    if not path:
        print(f"{label:10s} NOT RUN")
        return False
    try:
        s = json.load(open(path)).get("summary", {})
    except Exception as exc:
        print(f"{label:10s} UNREADABLE ({exc})")
        return False
    print(f"{label:10s} {os.path.basename(path)}")
    for k in keys:
        print(f"  {k:22s} {s.get(k)}")
    return True


ok = True
ok &= show("toolcall", latest(f"toolcall_eval_{tag}_*.json"),
           ["n", "tool_f1", "arg_validity_rate", "retries", "median_seconds"])
ok &= show("bfcl", latest(f"bfcl_{tag}_*rescored.json", f"bfcl_{tag}_*.json"),
           ["n", "pass_at_k", "tool_f1", "irrelevance_acc", "arg_validity_rate",
            "retries", "mean_steps", "median_seconds"])
ok &= show("gaia", latest(f"agent_tasks_{tag}-gaia_*.json"),
           ["n", "pass_rate", "mean_steps_to_success", "median_seconds"])
ok &= show("tb", latest(f"agent_tasks_{tag}-tb_*.json"),
           ["n", "pass_rate", "mean_steps_to_success", "median_seconds"])
print("=====================================================")
sys.exit(0 if ok else 1)
EOF
[ "$FAILED" -eq 0 ] || { echo "SWEEP INCOMPLETE (see FAILED lines above)"; exit 1; }
echo "SWEEP COMPLETE"
