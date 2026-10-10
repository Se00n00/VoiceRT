"""DRACO eval: deep-research reports, rubric-graded by an LLM judge.

Each task runs as one autonomous turn (auto-approving confirm; policy
denies still apply) with a long step budget (reports are thousands of
words), then a judge call grades the report against the task rubric:
binary MET/UNMET per criterion, weighted aggregate to 0-100 (positive
criteria add weight when met; negative-weight pitfalls subtract when
committed). Mirrors the DRACO paper's normalized score.

Dataset: benchmarks/draco/test.jsonl (100 tasks, perplexity-ai/draco).

Usage:
  PYTHONPATH=. .venv/bin/python -u benchmarks/draco_eval.py \
      --backend groq [--limit 3] [--offset 0] [--tag draco-groq]
      [--timeout 1800] [--max-tokens 8192]
Results: benchmarks/results/draco_<tag>_<ts>.json
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.agent_eval_lib import anchor_workdir, make_workdir, run_task, warm_agent

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "draco.jsonl")

REPORT_INSTRUCTION = (
    "\nResearch this thoroughly using web search (multiple queries, "
    "primary sources, compare claims). Save the full report with "
    "inline citations (source URLs) to report.md in the working "
    "directory, then reply with the COMPLETE report text as your "
    "final message."
)


def load_tasks(path: str, offset: int, limit: int) -> list:
    """DRACO jsonl slice -> [{id, problem, domain, criteria}]. Pure.

    ..
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        rows = json.loads(raw)
    except Exception:
        rows = [json.loads(l) for l in raw.splitlines() if l.strip()]
    tasks = []
    for r in rows[offset:(offset + limit) if limit else None]:
        try:
            rubric = json.loads(r["answer"])
        except Exception:
            continue
        criteria = []
        for s in rubric.get("sections", []):
            for c in s.get("criteria", []):
                criteria.append({
                    "id": str(c.get("id", "")),
                    "weight": c.get("weight", 0),
                    "section": str(s.get("id", "")),
                    "requirement": str(c.get("requirement", "")),
                })
        if not criteria:
            continue
        tasks.append({"id": str(r.get("id", ""))[:8],
                      "domain": str(r.get("domain", "")),
                      "problem": str(r.get("problem", "")),
                      "criteria": criteria})
    return tasks


def judge_prompt(problem: str, report: str, criteria: list) -> str:
    """Rubric grading prompt: JSON id->0/1 only. Pure.

    ..
    """
    lines = []
    for c in criteria:
        pol = "+" if c["weight"] >= 0 else "-pitfall"
        lines.append(f"- {c['id']} [{pol}] {c['requirement']}")
    return (
        "You grade a research report against a rubric. For each criterion "
        "reply 1 when satisfied (requirement met, or pitfall avoided) "
        "else 0.\n\nTASK:\n" + problem[:3000] +
        "\n\nREPORT:\n" + report[:24000] +
        "\n\nCRITERIA:\n" + "\n".join(lines) +
        "\n\nReply ONLY as JSON: {\"<id>\": 0/1, ...} — every id, no prose."
    )


def parse_verdicts(text: str, ids: list) -> dict:
    """First balanced {...} object -> {id: 0/1} (missing -> 0). Pure.

    ..
    """
    try:
        start = text.index("{")
    except ValueError:
        return {i: 0 for i in ids}
    depth, instr, esc, end = 0, False, False, -1
    for i in range(start, len(text)):
        ch = text[i]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    try:
        obj = json.loads(text[start:end]) if end > start else {}
    except Exception:
        return {i: 0 for i in ids}
    out = {}
    for i in ids:
        v = obj.get(i, 0)
        out[i] = 1 if v is True or v == 1 or str(v) == "1" else 0
    return out


def score_task(criteria: list, verdicts: dict) -> dict:
    """Weighted aggregate to 0-100 (paper's normalized score). Pure.

    ..
    """
    pos = sum(c["weight"] for c in criteria if c["weight"] > 0)
    got = 0.0
    for c in criteria:
        v = verdicts.get(c["id"], 0)
        if c["weight"] >= 0:
            got += c["weight"] * v
        else:
            got += c["weight"] * (1 - v)
    norm = max(0.0, min(100.0, 100.0 * got / pos)) if pos else 0.0
    met = sum(1 for c in criteria if verdicts.get(c["id"], 0))
    return {"score": round(norm, 2), "met": met, "n": len(criteria)}


def pick_report(workdir: str, reply: str) -> str:
    """report.md when substantial, else the final reply. Pure-ish.

    Models sometimes save the deliverable and reply briefly — grading
    the stub would score the envelope, not the work.
    ..
    """
    reply = str(reply or "")
    try:
        with open(os.path.join(workdir, "report.md"), encoding="utf-8",
                  errors="replace") as f:
            file_report = f.read()
    except Exception:
        return reply
    return file_report if len(file_report) > len(reply) else reply


async def judge_one(judge, problem: str, report: str,
                    criteria: list) -> dict:
    """Grade one report. Returns {score, met, n, verdicts}.

    ..
    """
    from src.models.llm import split_thinking

    res = await judge.generate(
        [{"role": "user", "content": judge_prompt(problem, report,
                                                  criteria)}],
        max_tokens=2000)
    _, answer = split_thinking(res.text)
    verdicts = parse_verdicts(answer or res.text,
                              [c["id"] for c in criteria])
    return {**score_task(criteria, verdicts), "verdicts": verdicts}


ELICIT = (
    "Now write out the COMPLETE research report as your final message: "
    "findings, numbers, and source URLs. Long and complete beats short."
)


async def main_async(args):
    import uuid

    tasks = load_tasks(DATASET, args.offset, args.limit)
    print(f"{len(tasks)} draco tasks (offset {args.offset})", flush=True)
    agent = await warm_agent(model=args.model, backend=args.backend,
                             max_tokens=args.max_tokens)
    from src.models.llm import LlmConfig, LlmModel

    # Grading judge: same bonsai leg by default (--judge-backend), so the
    # score measures one setup, not agent-vs-judge mixing.
    jbackend = args.judge_backend or args.backend
    if jbackend in ("groq", "gemini") and args.model != "Qwen/Qwen3-0.6B":
        key = "groq_model" if jbackend == "groq" else "gemini_model"
        judge = LlmModel(LlmConfig(backend=jbackend, **{key: args.model}))
    else:
        judge = LlmModel(LlmConfig(model=args.model, backend=jbackend))
    rows = []
    for i, t in enumerate(tasks):
        if i and args.pause:
            await asyncio.sleep(args.pause)
        workdir = make_workdir()
        anchor_workdir(agent, workdir)
        try:
            sid = f"draco-{t['id']}"
            res = await run_task(agent, t["problem"] + REPORT_INSTRUCTION,
                                 workdir, timeout_s=args.timeout,
                                 session_id=sid)
            report = pick_report(workdir, res["reply"])
            if len(report) < 500 and res["steps"] > 0 \
                    and not res["timed_out"]:
                # Researched but punted on delivery: one elicitation turn
                # in the same session (history continues the work).
                res2 = await run_task(agent, ELICIT, workdir,
                                      timeout_s=args.timeout,
                                      session_id=sid)
                cand = pick_report(workdir, res2["reply"])
                if len(cand) > len(report):
                    report, res = cand, res2
            grade = await judge_one(judge, t["problem"], report,
                                    t["criteria"])
            row = {"id": t["id"], "domain": t["domain"],
                   "problem": t["problem"][:200],
                   "report_chars": len(report),
                   "report_source": ("report.md" if report != res["reply"]
                                     else "reply"),
                   "steps": res["steps"], "seconds": res["seconds"],
                   "timed_out": res["timed_out"], **grade}
        except Exception as exc:
            row = {"id": t["id"], "domain": t["domain"],
                   "problem": t["problem"][:200], "error": str(exc)[:300],
                   "score": 0.0, "met": 0,
                   "n": len(t["criteria"])}
            report = ""
        if "verdicts" in row:
            del row["verdicts"]
        row["report"] = (report or "")[:12000]
        rows.append(row)
        print(f"[{t['id']}] {t['domain']:22s} "
              f"score={row.get('score', 0):6.2f} "
              f"met={row.get('met', 0):3d}/{row.get('n', 0):3d} "
              f"steps={row.get('steps', 0):3d} "
              f"{row.get('seconds', 0):7.1f}s "
              f"rep={len(report or ''):6d}ch "
              f"{row.get('error', '')}", flush=True)
    return rows, {"backend": args.backend, "model": args.model,
                  "judge_backend": args.judge_backend or args.backend,
                  "timeout_s": args.timeout,
                  "max_tokens": args.max_tokens}


def summarize(rows):
    """Mean score overall + per domain. Pure.

    ..
    """
    scores = [r["score"] for r in rows]
    by_domain: dict = {}
    for r in rows:
        by_domain.setdefault(r["domain"], []).append(r["score"])
    return {
        "n": len(rows),
        "mean_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "by_domain": {d: round(sum(v) / len(v), 2)
                      for d, v in sorted(by_domain.items())},
    }


def main():
    ap = argparse.ArgumentParser(description="draco deep-research eval")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--judge-backend", default="",
                    help="grading leg (default: same as --backend)")
    ap.add_argument("--tag", default="draco")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pause", type=int, default=45,
                    help="seconds between tasks (rate-limit pacing)")
    args = ap.parse_args()
    rows, meta = asyncio.run(main_async(args))
    summary = summarize(rows)
    print("---")
    for k, v in summary.items():
        print(f"{k:12s} {v}")
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = f"benchmarks/results/draco_{args.tag}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
