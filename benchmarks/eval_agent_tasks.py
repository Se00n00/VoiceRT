#!/usr/bin/env python3
"""Agent task eval: GAIA-L1 and TerminalBench-style tasks through VoiceAgent.

Each task runs as one autonomous turn (auto-approving confirm; policy
denies still apply) in a fresh temp workdir. Grading is oracle-based on
resulting file state (+ reply text), pytest-style: every check is an
independent boolean, a task passes iff all checks pass.

Metrics: pass rate, mean steps-to-success (tool calls on passing tasks),
median seconds, confirms seen.

Usage:
  PYTHONPATH=. .venv/bin/python -u benchmarks/eval_agent_tasks.py \
      --tasks benchmarks/gaia_l1.json --tag gaia [--timeout 600] [--limit N]
Results: benchmarks/results/agent_tasks_<tag>_<ts>.json
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.agent_eval_lib import make_workdir, run_task, warm_agent


def grade_task(workdir, reply, oracle):
    """Pure oracle grading: list of (check, ok, detail). Never raises."""
    results = []
    for check in oracle or []:
        ctype = check.get("type")
        try:
            if ctype == "file_contains":
                with open(os.path.join(workdir, check["path"]), encoding="utf-8",
                          errors="replace") as f:
                    body = f.read()
                ok = check["text"] in body
                detail = f"{check['path']} contains {check['text']!r}"
            elif ctype == "file_not_contains":
                with open(os.path.join(workdir, check["path"]), encoding="utf-8",
                          errors="replace") as f:
                    body = f.read()
                ok = check["text"] not in body
                detail = f"{check['path']} lacks {check['text']!r}"
            elif ctype == "file_equals":
                with open(os.path.join(workdir, check["path"]), encoding="utf-8",
                          errors="replace") as f:
                    body = f.read().strip()
                ok = body == check["text"]
                detail = f"{check['path']} == {check['text']!r} (got {body[:80]!r})"
            elif ctype == "file_exists":
                ok = os.path.exists(os.path.join(workdir, check["path"]))
                detail = f"{check['path']} exists"
            elif ctype == "reply_contains":
                ok = check["text"] in (reply or "")
                detail = f"reply contains {check['text']!r}"
            else:
                ok, detail = False, f"unknown check {ctype!r}"
        except FileNotFoundError:
            ok, detail = False, f"{check.get('path')}: file missing"
        except Exception as exc:
            ok, detail = False, f"grader error: {exc}"[:200]
        results.append({"check": ctype, "ok": bool(ok), "detail": detail})
    return results


async def main_async(args, tasks):
    agent = await warm_agent(model=args.model, backend=args.backend)
    rows = []
    for task in tasks:
        workdir = make_workdir(task.get("setup"))
        res = await run_task(agent, task["prompt"], workdir,
                             timeout_s=args.timeout,
                             session_id=f"eval-{task['id']}")
        checks = grade_task(workdir, res["reply"], task.get("oracle"))
        passed = bool(checks) and all(c["ok"] for c in checks)
        row = {"id": task["id"], "prompt": task["prompt"][:160],
               "passed": passed, "checks": checks, **res}
        rows.append(row)
        mark = "OK " if passed else "FAIL"
        print(f"[{mark}] {task['id']:6s} steps={res['steps']:2d} "
              f"confirms={res['confirms']} {res['seconds']:6.1f}s "
              f"reply={res['reply'][:70]!r}", flush=True)
    return rows, {"model": args.model, "backend": args.backend,
                  "timeout_s": args.timeout}


def summarize(rows):
    passed = [r for r in rows if r["passed"]]
    steps = [r["steps"] for r in passed]
    dts = sorted(r["seconds"] for r in rows)
    return {
        "n": len(rows),
        "pass_rate": round(len(passed) / len(rows), 3) if rows else 0.0,
        "mean_steps_to_success": (round(sum(steps) / len(steps), 2)
                                  if steps else None),
        "median_seconds": dts[len(dts) // 2] if dts else 0.0,
        "total_confirms": sum(r["confirms"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description="agent task eval (GAIA/TB)")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--backend", default="gemma")
    ap.add_argument("--tag", default="tasks")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    with open(args.tasks, encoding="utf-8") as f:
        tasks = json.load(f)
    if args.limit:
        tasks = tasks[:args.limit]
    rows, meta = asyncio.run(main_async(args, tasks))
    summary = summarize(rows)
    print("---")
    for k, v in summary.items():
        print(f"{k:22s} {v}")
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = f"benchmarks/results/agent_tasks_{args.tag}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
