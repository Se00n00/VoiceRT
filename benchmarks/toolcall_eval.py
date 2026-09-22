#!/usr/bin/env python3
"""BFCL-style offline tool-call eval over our toolset (no weights mocked).

Methodology (mirrors Berkeley BFCL's offline AST check, adapted to our
single-model setup): each case is ONE greedy model call through the exact
production path (chat template + routed tools + stop strings). The raw
output is parsed with the production parsers; the predicted call is
compared AST-style against the expected tool + required args.

Metrics per run:
- tool F1 (micro over exact (tool, required-args) decisions)
- arg validity rate (predicted calls whose required args are present)
- retries (harness echo/garble second attempts)
- wall time per case (median reported)

Usage:
  PYTHONPATH=. .venv/bin/python -u benchmarks/toolcall_eval.py [--limit N]
      [--offset N] [--subset benchmarks/toolcall_subset.json]
      [--model openbmb/MiniCPM5-1B] [--backend minicpm] [--tag minicpm]
Results: benchmarks/results/toolcall_eval_<tag>_<ts>.json
"""
import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _required_args():
    from src.tools.terminal import TERMINAL_TOOLS

    return {t["name"]: set(t.get("parameters", {}).get("required", []))
            for t in TERMINAL_TOOLS}


def _args_dict(action):
    if action is None:
        return {}
    return {k: v for k, v in action.as_dict().items() if k != "action"}


async def _attempt(cm, llm, msgs, max_tokens, tools, counter):
    """One greedy attempt; returns raw text. Counts llm.stream calls."""
    acc: dict = {}
    orig_stream = llm.stream

    async def spy(*a, **k):
        counter["streams"] += 1
        async for tok in orig_stream(*a, **k):
            yield tok

    llm.stream = spy  # type: ignore
    try:
        async for _ in cm._stream_raw(msgs, max_tokens, tools, acc):
            pass
    finally:
        llm.stream = orig_stream  # type: ignore
    return acc.get("raw", "")


async def run_case(cm, llm, case, max_tokens):
    """Returns (predicted_action_or_None, retries, seconds, raws)."""
    from src.tools.terminal import tools_for_request

    text = case["prompt"]
    msgs = llm.messages_for_terminal(text, [], cwd=".", observation="")
    tools = tools_for_request(text, "")
    counter = {"streams": 0}
    raws = []
    t0 = time.perf_counter()
    raw = await _attempt(cm, llm, msgs, max_tokens, tools, counter)
    raws.append(raw)
    thinking, answer, action = cm._parse_action(raw)
    retries = 0
    if cm._needs_retry(raw, thinking, answer) is not None:
        retries = 1
        nudge = cm._needs_retry(raw, thinking, answer)
        msgs2 = msgs + [{"role": "user", "content": nudge}]
        raw2 = await _attempt(cm, llm, msgs2, max_tokens, tools, counter)
        raws.append(raw2)
        thinking, answer, action = cm._parse_action(raw2)
    dt = time.perf_counter() - t0
    return action, retries, dt, raws, counter["streams"]


def score_case(case, action, required):
    """(tp, fp, fn, valid_or_none). Exact (tool, required-args) match."""
    exp = case.get("expected")
    if exp is None:
        # chat case: any real tool call is a false positive
        if action is None or getattr(action, "op", "") == "done":
            return 0, 0, 0, None
        return 0, 1, 0, _valid(action, required)
    if action is None or getattr(action, "op", "") == "done":
        return 0, 0, 1, None
    if action.op != exp["tool"]:
        return 0, 1, 1, _valid(action, required)
    want = exp.get("args", {}) or {}
    got = _args_dict(action)
    if all(str(got.get(k, "")).strip() == str(v).strip() for k, v in want.items()):
        return 1, 0, 0, _valid(action, required)
    return 0, 1, 1, _valid(action, required)


def _valid(action, required):
    if action is None or getattr(action, "op", "") == "done":
        return None
    need = required.get(action.op, set())
    got = _args_dict(action)
    return all(str(got.get(k, "") or "").strip() for k in need)


async def main_async(args, cases):
    from src.agent.chat_model import LocalChatModel
    from src.models.llm import LlmConfig, LlmModel

    llm = LlmModel(LlmConfig(model=args.model, backend=args.backend))
    print(f"warming {args.model} [{args.backend}] ...", flush=True)
    await llm.warm()
    from src.models.llm import assert_cuda_leg

    print(f"ready on {assert_cuda_leg(llm)}.", flush=True)
    cm = LocalChatModel(llm=llm)
    base = int(getattr(getattr(llm, "config", None), "max_tokens", 48) or 48)
    is_minicpm = str(getattr(getattr(llm, "config", None), "backend", "")).startswith("minicpm")
    max_tokens = max(base, 320) if is_minicpm else max(base, 256)
    required = _required_args()

    rows = []
    for case in cases:
        action, retries, dt, raws, streams = await run_case(cm, llm, case, max_tokens)
        tp, fp, fn, valid = score_case(case, action, required)
        pred = None if action is None else {"tool": action.op, **_args_dict(action)}
        rows.append({
            "id": case["id"], "category": case.get("category", "?"),
            "prompt": case["prompt"],
            "expected": case.get("expected"),
            "predicted": pred, "tp": tp, "fp": fp, "fn": fn,
            "arg_valid": valid, "retries": retries,
            "stream_calls": streams, "seconds": round(dt, 2),
            "raw": (raws[-1] if raws else "")[:800],
        })
        got = "none" if pred is None else pred["tool"]
        want = "none" if case.get("expected") is None else case["expected"]["tool"]
        flag = "OK " if tp else ("fp " if fp else "FN ")
        print(f"[{flag}] {case['id']:4s} want={want:12s} got={got:12s} "
              f"retry={retries} {dt:5.1f}s", flush=True)
    return rows, {"model": args.model, "backend": args.backend,
                  "max_tokens": max_tokens}


def summarize(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    valids = [r["arg_valid"] for r in rows if r["arg_valid"] is not None]
    retries = sum(r["retries"] for r in rows)
    dts = sorted(r["seconds"] for r in rows)
    return {
        "n": len(rows),
        "tool_precision": round(prec, 3),
        "tool_recall": round(rec, 3),
        "tool_f1": round(f1, 3),
        "arg_validity_rate": (round(sum(valids) / len(valids), 3)
                              if valids else None),
        "arg_valid_n": len(valids),
        "retries": retries,
        "retry_rate": round(retries / len(rows), 3) if rows else 0.0,
        "median_seconds": dts[len(dts) // 2] if dts else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="BFCL-style offline tool-call eval")
    ap.add_argument("--subset", default="benchmarks/toolcall_subset.json")
    ap.add_argument("--model", default="openbmb/MiniCPM5-1B")
    ap.add_argument("--backend", default="minicpm")
    ap.add_argument("--tag", default="minicpm")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    with open(args.subset, encoding="utf-8") as f:
        cases = json.load(f)
    if args.offset:
        cases = cases[args.offset:]
    if args.limit:
        cases = cases[:args.limit]
    rows, meta = asyncio.run(main_async(args, cases))
    summary = summarize(rows)
    print("---")
    for k, v in summary.items():
        print(f"{k:18s} {v}")
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = f"benchmarks/results/toolcall_eval_{args.tag}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "rows": rows},
                  f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
