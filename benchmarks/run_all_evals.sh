#!/usr/bin/env bash
# Eval sweep in one command: own tools -> harness GAIA sample -> harness
# TB 2.1 sample, then a consolidated results table at the end.
#
# Usage (from voice-pipeline/, GPU idle, ONE model process at a time):
#   PYTHONPATH=. bash benchmarks/run_all_evals.sh [TAG]        # sampled sweep
#   PY=/path/to/python bash benchmarks/run_all_evals.sh mytag   # custom interp
#
# Harness steps sample their suites (--limit 5): full runs are Kaggle
# jobs (benchmarks/kaggle.md), not part of this sweep.
#
# Each step runs even if an earlier one fails; exit code is nonzero if
# anything failed. Every runner aborts loudly on CPU fallback (no silent
# hours-long CPU runs). Results land in benchmarks/results/.
set -u
TAG="${1:-kaggle}"
# PY override wins; else a *working* .venv (probe import — a broken venv
# errors in sitecustomize before user code runs), else system python3.
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ] && .venv/bin/python -c "import sys" 2>/dev/null; then PY=.venv/bin/python; else PY=python3; fi
fi
export PYTHONPATH=.

FAILED=0
step() { echo; echo "===== $1 ====="; }

step "1/3 own toolset (18 cases)"
$PY -u benchmarks/toolcall_eval.py --tag "$TAG" || { echo "toolcall FAILED"; FAILED=1; }

step "2/3 harness GAIA sample (5 tasks)"
$PY -u benchmarks/harness/eval.py --suite gaia --limit 5 --tag "$TAG" || { echo "gaia FAILED"; FAILED=1; }

step "3/3 harness TB 2.1 sample (5 tasks, easy-first)"
$PY -u benchmarks/harness/eval.py --suite tb21 --limit 5 --tag "$TAG" || { echo "tb21 FAILED"; FAILED=1; }

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
ok &= show("gaia", latest(f"harness_gaia_{tag}_*.json"),
           ["n", "n_graded", "pass_rate", "mean_steps_to_success", "median_seconds"])
ok &= show("tb21", latest(f"harness_tb21_{tag}_*.json"),
           ["n", "verifier_pending", "median_seconds"])
print("=====================================================")
sys.exit(0 if ok else 1)
EOF
[ "$FAILED" -eq 0 ] || { echo "SWEEP INCOMPLETE (see FAILED lines above)"; exit 1; }
echo "SWEEP COMPLETE"
