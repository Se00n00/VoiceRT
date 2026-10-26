#!/usr/bin/env python3
"""Agent task eval: GAIA, TerminalBench-style and TB 2.1 tasks via VoiceAgent.

Each task runs as one autonomous turn (auto-approving confirm; policy
denies still apply) in a fresh temp workdir. Grading is oracle-based on
resulting file state (+ reply text), pytest-style: every check is an
independent boolean, a task passes iff all checks pass.

TB 2.1 tasks (benchmarks/harness/tb21.jsonl, via
benchmarks/harness/eval.py --adapt tb21)
carry a verifier-only oracle: the harness records the trajectory
(steps, seconds, reply) and leaves scoring to the task verifier
under harbor/docker (benchmarks/kaggle.md section 3). Those rows
report passed=null and count as verifier_pending in the summary.

The GAIA + TB 2.1 suites live in benchmarks/harness/ — prefer
benchmarks/harness/eval.py for those. This module keeps the
shared grade_task/summarize and still runs ad-hoc manifests (e.g.
benchmarks/terminal_bench.json, the local TB-style smoke).

Metrics: pass rate (over graded tasks), mean steps-to-success
(tool calls on passing tasks), median seconds, confirms seen.

Usage:
  PYTHONPATH=. .venv/bin/python -u benchmarks/eval_agent_tasks.py \
      --tasks benchmarks/terminal_bench.json --tag tb-smoke [--timeout 600] [--limit N]
Results: benchmarks/results/agent_tasks_<tag>_<ts>.json
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.agent_eval_lib import areanchor_workdir, make_workdir, run_task, warm_agent


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
            elif ctype == "reply_match":
                # GAIA rule: exact match after light normalization (outer
                # whitespace, trailing periods, wrapping quotes — model
                # verbosity artifacts, never answer content).
                def _norm(s):
                    s = str(s or "").strip()
                    s = s.strip("'\"").strip()
                    return s.rstrip(".").strip()
                want, got = _norm(check.get("text", "")), _norm(reply)
                ok = bool(want) and got == want
                detail = f"reply == {want!r} (got {got[:80]!r})"
            elif ctype == "tb21":
                # Verifier-graded under harbor/docker: the harness only
                # records the trajectory, never scores it locally.
                ok, detail = None, (f"verifier-only {check.get('task')}: "
                                    "grade via harbor (kaggle.md S3)")
            else:
                ok, detail = False, f"unknown check {ctype!r}"
        except FileNotFoundError:
            ok, detail = False, f"{check.get('path')}: file missing"
        except Exception as exc:
            ok, detail = False, f"grader error: {exc}"[:200]
        results.append({"check": ctype, "ok": ok, "detail": detail})
    return results


async def main_async(args, tasks):
    agent = await warm_agent(model=args.model, backend=args.backend,
                             max_tokens=args.max_tokens or None)
    rows = []
    try:
        for task in tasks:
            if args.level and int(task.get("level", 1)) not in args.level:
                continue
            workdir = make_workdir(task.get("setup"), task.get("attachments"))
            await areanchor_workdir(agent, workdir)
            res = await run_task(agent, task["prompt"], workdir,
                                 timeout_s=args.timeout,
                                 session_id=f"eval-{task['id']}")
            checks = grade_task(workdir, res["reply"], task.get("oracle"))
            if checks and all(c["ok"] is None for c in checks):
                passed = None
            else:
                passed = bool(checks) and all(c["ok"] for c in checks)
            row = {"id": task["id"], "level": int(task.get("level", 1)),
                   "prompt": task["prompt"][:160],
                   "passed": passed, "checks": checks, **res}
            rows.append(row)
            mark = "PEND" if passed is None else ("OK " if passed else "FAIL")
            print(f"[{mark}] {task['id']:6s} steps={res['steps']:2d} "
                  f"confirms={res['confirms']} {res['seconds']:6.1f}s "
                  f"reply={res['reply'][:70]!r}", flush=True)
        return rows, {"model": args.model, "backend": args.backend,
                      "timeout_s": args.timeout}
    finally:
        await agent.aclose()


def summarize(rows):
    graded = [r for r in rows if r.get("passed") is not None]
    passed = [r for r in graded if r["passed"]]
    steps = [r["steps"] for r in passed]
    dts = sorted(r["seconds"] for r in rows)
    by_level = {}
    for lv in (1, 2, 3):
        sub = [r for r in graded if r.get("level", 1) == lv]
        if sub:
            by_level[f"L{lv}"] = (f"{sum(1 for r in sub if r['passed'])}/"
                                  f"{len(sub)}")
    return {
        "n": len(rows),
        "n_graded": len(graded),
        "verifier_pending": len(rows) - len(graded),
        "pass_rate": round(len(passed) / len(graded), 3) if graded else 0.0,
        "by_level": by_level,
        "mean_steps_to_success": (round(sum(steps) / len(steps), 2)
                                  if steps else None),
        "median_seconds": dts[len(dts) // 2] if dts else 0.0,
        "total_confirms": sum(r["confirms"] for r in rows),
    }


def main():
    ap = argparse.ArgumentParser(description="agent task eval (GAIA/TB)")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HF id for local legs; model id for cloud legs")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--tag", default="tasks")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="per-step token budget (0 = backend floor; "
                         "reasoning legs need more, e.g. 2048)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--level", type=int, nargs="*", default=[],
                    help="GAIA levels to run, e.g. --level 1 2 (empty = all)")
    args = ap.parse_args()
    with open(args.tasks, encoding="utf-8") as f:
        raw = f.read()
    try:
        tasks = json.loads(raw)
    except Exception:
        tasks = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[args.offset:args.offset + args.limit]
    elif args.offset:
        tasks = tasks[args.offset:]
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
