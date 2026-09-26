#!/usr/bin/env python3
"""Real BFCL eval (V1/V2/V3) for the local model — offline after sampling.

Data: genuine Berkeley Function Calling Leaderboard files (see provenance
in each manifest), V1 hand-graded (no machine answers ship with V1),
V2/V3 byte-exact upstream ground truth. Manifests: benchmarks/bfcl_*.json.

Method: ONE greedy model call per turn through the exact production path
(chat template + foreign tool specs + stop strings). Predicted calls are
extracted generically (native ``<function>`` XML or JSON) and scored
AST-style: function name must match and every ground-truth parameter must
match one of its acceptable values (any-of lists, type-lenient).

Metrics: tool F1 (micro, exact call match), arg validity rate (required
params present on predicted calls), retries (harness echo/garble nudges),
steps to success (multi-turn), pass@k (independent attempts, k>1).

Usage:
  PYTHONPATH=. .venv/bin/python -u benchmarks/bfcl_eval.py --version v3
  PYTHONPATH=. .venv/bin/python -u benchmarks/bfcl_eval.py --version all --k 3
  Flags: --limit/--offset (chunk a version), --model/--backend/--tag.
Results: benchmarks/results/bfcl_<tag>_<ts>.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FUNC_RE = re.compile(
    r'<function\s+name="([^"]+)"\s*>(.*?)</function\s*>',
    re.DOTALL | re.IGNORECASE)
_PARAM_RE = re.compile(
    r'<param\s+name="([^"]+)"\s*>(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</param\s*>',
    re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
_GEMMA_RE = re.compile(
    r"<\|tool_call>call:([A-Za-z0-9_.\-]+)(.*?)<tool_call\|>",
    re.DOTALL)
_CALL_RE = re.compile(r"^([\w.]+)\((.*)\)\s*$", re.DOTALL)

SYSTEM = ("You are a helpful assistant with access to functions. "
          "Call exactly the functions needed to answer, nothing else.")


# -- generic call extraction (any function names, not just ours) --------

_BARE_PAIR_RE = re.compile(
    r'name="([^"]+)"\s*>\s*([^<>\n]*?)(?=\s*name="[^"]+"\s*>|\s*$)',
    re.DOTALL | re.IGNORECASE | re.MULTILINE)


def _extract_bare(text, known):
    """Bare-tail form (1B quirk): ``name="f"> name="p">v`` runs.

    First pair opens the call; a later pair whose name is a known
    function opens a new call (parallel emissions), else it's a param.
    """
    pairs = _BARE_PAIR_RE.findall(text or "")
    if not pairs:
        return []
    known = set(known or [])
    calls, cur_name, cur_args = [], None, {}
    for name, val in pairs:
        name, val = name.strip(), val.strip()
        if cur_name is None:
            cur_name, cur_args = name, {}
        elif name in known:
            calls.append((cur_name, cur_args))
            cur_name, cur_args = name, {}
        else:
            cur_args[name] = val
    if cur_name is not None:
        calls.append((cur_name, cur_args))
    return calls


def extract_calls(text, known=None):
    """All (name, args) calls in raw output: Gemma native, XML, bare, JSON."""
    out = []
    for gm in _GEMMA_RE.finditer(text or ""):
        raw_args = (gm.group(2) or "").strip()
        try:
            args = json.loads(raw_args) if raw_args else {}
        except Exception:
            continue
        if isinstance(args, dict):
            out.append((gm.group(1).strip(),
                        {str(k): v for k, v in args.items()}))
    if out:
        return out
    for fm in _FUNC_RE.finditer(text or ""):
        params = {}
        for pm in _PARAM_RE.finditer(fm.group(2)):
            key = pm.group(1).strip()
            val = pm.group(2) if pm.group(2) is not None else (pm.group(3) or "")
            params[key] = val.strip()
        out.append((fm.group(1).strip(), params))
    if out:
        return out
    bare = _extract_bare(text, known)
    if bare:
        return bare
    for m in _JSON_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("function")
        args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name and isinstance(args, dict):
            out.append((str(name), {str(k): v for k, v in args.items()}))
    return out


def parse_gt_string(s):
    """'cd(folder='document')' -> ('cd', {'folder': 'document'})."""
    m = _CALL_RE.match((s or "").strip())
    if not m:
        return None, {}
    args = {}
    for part in m.group(2).split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        args[k.strip()] = v
    return m.group(1).strip(), args


# -- AST comparison (BFCL any-of-lists semantics, type-lenient) -----------

def _val_match(pred, accepted):
    """Predicted value matches if it equals ANY acceptable value.

    Accepts nested single-element lists (e.g. SQL GT like [["name"]]):
    one level is unwrapped before comparing. Type-lenient throughout.
    """
    for acc in accepted:
        if pred == acc:
            return True
        if isinstance(acc, list) and len(acc) == 1:
            if _val_match(pred, [acc[0]]):
                return True
            continue
        ps, vs = str(pred).strip(), str(acc).strip()
        if ps == vs or ps.lower() == vs.lower():
            return True
        try:
            if float(ps) == float(vs):
                return True
        except Exception:
            pass
    return False


def _call_match(pred, gt_options):
    """(name, args) matches if some ground-truth option matches fully.

    gt_options: [{func: {param: [acceptable...]}}]. Every GT-specified
    param must match; extra predicted params are ignored (official rule).
    """
    pname, pargs = pred
    for opt in gt_options or []:
        if not isinstance(opt, dict):
            continue
        for fname, fparams in opt.items():
            if fname != pname:
                continue
            ok = True
            for param, accepted in (fparams or {}).items():
                acc = accepted if isinstance(accepted, list) else [accepted]
                if param not in pargs:
                    # missing is fine iff empty is acceptable (optional param)
                    if not any(str(a or "") == "" for a in acc):
                        ok = False
                        break
                    continue
                if not _val_match(pargs[param], acc):
                    ok = False
                    break
            if ok:
                return True
    return False


def _required_of(spec):
    try:
        return list((spec or {}).get("parameters", {}).get("required", []) or [])
    except Exception:
        return []


# -- full-dataset machinery: filtering, resume, isolation ---------------

PROMPT_TOKEN_BUDGET = 7000
"""Skip cases whose estimated prompt exceeds this (max_len 8192 minus gen)."""


def load_manifest(path):
    """Manifest loader: .json (array or {cases:[...]}) or .jsonl.gz.

    Full-dataset manifests ship gzipped (1.6MB vs 28MB). Pure.
    """
    if str(path).endswith(".gz"):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    return data


def case_key(case) -> str:
    """Composite key: ids collide across versions (simple_0 x3)."""
    return f"{case.get('version', '?')}/{case.get('id', '?')}"


def filter_cases(cases, categories=None, done=None):
    """Pure: keep cases matching categories and not already done."""
    cats = set(categories or [])
    done = set(done or [])
    return [c for c in cases
            if (not cats or c.get("category") in cats)
            and case_key(c) not in done]


def load_done_ids(path):
    """Row ids (composite keys) from a prior results file. Never raises."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", data) if isinstance(data, dict) else data
        return {f"{r.get('version', '?')}/{r.get('id', '?')}"
                for r in rows if isinstance(r, dict)}
    except Exception:
        return set()


def _turn_texts(case):
    """All user texts in a case (single- or multi-turn) for budgeting."""
    q = case.get("question")
    if isinstance(q, str):
        return [q]
    texts = []
    if isinstance(q, list):
        for turn in q:
            msgs = turn if isinstance(turn, list) else [turn]
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    texts.append(str(m.get("content", "")))
    return texts or [""]


def estimate_case_tokens(case, max_tokens: int) -> int:
    """Rough prompt size (chars/4) incl. system + tools + worst turn."""
    tools = case.get("functions", [])
    if isinstance(tools, dict):
        tools = [tools]
    try:
        tools_s = json.dumps(tools)
    except Exception:
        tools_s = ""
    worst = max((len(t) for t in _turn_texts(case)), default=0)
    return (len(SYSTEM) + worst + len(tools_s)) // 4 + int(max_tokens)


def skipped_row(case, reason: str) -> dict:
    return {
        "id": case.get("id"), "version": case.get("version"),
        "category": case.get("category"), "prompt": _user_text(case.get("question"))[:200],
        "expected_tool": None, "expect_none": False,
        "tp": 0, "fp": 0, "fn": 0, "first_tp": 0,
        "arg_valid": None, "retries": 0, "steps": 0, "attempts": 0,
        "passed": False, "status": "skipped", "status_reason": reason,
        "first_calls": [], "first_raw": "", "seconds": 0.0,
    }


def error_row(case, message: str, seconds: float) -> dict:
    row = skipped_row(case, "")
    row.update(status="error", status_reason=str(message)[:300],
               seconds=round(seconds, 2))
    return row


def _arg_valid(name, args, spec_by_name):
    req = _required_of(spec_by_name.get(name))
    if not req:
        return True
    for r in req:
        v = (args or {}).get(r)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False
    return True


# -- model driving (production path, attempt counting for pass@k) ---------

async def _one_attempt(cm, llm, msgs, max_tokens, tools, counter):
    acc: dict = {}
    orig = llm.stream

    async def spy(*a, **k):
        counter["streams"] += 1
        async for tok in orig(*a, **k):
            yield tok

    llm.stream = spy  # type: ignore
    try:
        async for _ in cm._stream_raw(msgs, max_tokens, tools, acc):
            pass
    finally:
        llm.stream = orig  # type: ignore
    raw = acc.get("raw", "")
    thinking, answer, _ = cm._parse_action(raw)
    retries = 0
    if cm._needs_retry(raw, thinking, answer) is not None:
        retries = 1
        nudge = cm._needs_retry(raw, thinking, answer)
        acc2: dict = {}
        orig2 = llm.stream

        async def spy2(*a, **k):
            counter["streams"] += 1
            async for tok in orig2(*a, **k):
                yield tok

        llm.stream = spy2  # type: ignore
        try:
            async for _ in cm._stream_raw(
                    msgs + [{"role": "user", "content": nudge}],
                    max_tokens, tools, acc2):
                pass
        finally:
            llm.stream = orig2  # type: ignore
        raw = acc2.get("raw", "") or raw
    return raw, retries, counter["streams"]


def _user_text(question):
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        for turn in question:
            msgs = turn if isinstance(turn, list) else [turn]
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    return str(m.get("content", ""))
    return ""


async def run_single(cm, llm, case, max_tokens, k):
    """Single-turn case, k independent attempts. Returns row dict."""
    tools = case.get("functions", [])
    if isinstance(tools, dict):
        tools = [tools]
    text = _user_text(case.get("question"))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": text}]
    spec_by_name = {t.get("name"): t for t in tools if isinstance(t, dict)}
    known = set(spec_by_name)
    exp = case.get("expected")
    expect_none = not exp  # None or [] -> irrelevance: no call allowed
    attempts = []
    for _ in range(max(1, k)):
        counter = {"streams": 0}
        raw, retries, streams = await _one_attempt(cm, llm, msgs, max_tokens, tools, counter)
        calls = extract_calls(raw, known)
        if expect_none:
            ok = not calls
        elif not calls:
            ok = False
        elif len(exp) > 1:
            # parallel GT: whole predicted set must match, order-insensitive
            ok = len(calls) == len(exp) and all(
                any(_call_match(c, [g]) for g in exp) for c in calls)
        else:
            ok = _call_match(calls[0], exp)
        valid = None
        if calls:
            valid = all(_arg_valid(n, a, spec_by_name) for n, a in calls)
        attempts.append({"ok": ok, "calls": [(n, a) for n, a in calls],
                         "retries": retries, "streams": streams,
                         "valid": valid, "raw": raw[:600]})
    passed = any(a["ok"] for a in attempts)
    first = attempts[0]
    if passed:
        tp, fp, fn = 1, 0, 0
    elif expect_none and not first["calls"]:
        tp, fp, fn = 0, 0, 0  # correct rejection is neutral, not a failure
    elif expect_none:
        tp, fp, fn = 0, 1, 0
    else:
        tp, fp, fn = 0, 1, 1
    return {
        "id": case["id"], "version": case.get("version"),
        "category": case.get("category"), "prompt": text[:200],
        "expected_tool": (list(exp[0].keys())[0] if exp else None),
        "expect_none": bool(expect_none),
        "tp": tp, "fp": fp, "fn": fn,
        "first_tp": 1 if first["ok"] else 0,
        "arg_valid": first["valid"],
        "retries": sum(a["retries"] for a in attempts),
        "steps": 1, "attempts": len(attempts),
        "passed": passed, "first_calls": first["calls"],
        "first_raw": first["raw"],
    }


def _call_match_single(call, gt_options):
    return _call_match(call, gt_options)


async def run_multiturn(cm, llm, case, max_tokens):
    """Multi-turn case: feed user turns in order, score each turn's call."""
    tools = case.get("functions", [])
    spec_by_name = {t.get("name"): t for t in tools if isinstance(t, dict)}
    known = set(spec_by_name)
    history = [{"role": "system", "content": SYSTEM}]
    turns_ok, steps, retries, valids = 0, 0, 0, []
    detail = []
    gt_turns = case.get("expected") or []
    questions = case.get("question") or []
    for ti, turn in enumerate(questions):
        text = _user_text([turn])
        msgs = history + [{"role": "user", "content": text}]
        counter = {"streams": 0}
        raw, ret, _ = await _one_attempt(cm, llm, msgs, max_tokens, tools, counter)
        retries += ret
        steps += 1
        calls = extract_calls(raw, known)
        # GT turn: list of acceptable call STRINGS -> parse to options
        gt_opts = []
        if ti < len(gt_turns):
            for s in gt_turns[ti]:
                if isinstance(s, dict):
                    gt_opts.append({k: ({p: ([v] if not isinstance(v, list) else v)
                                            for p, v in (a or {}).items()})
                                    for k, a in s.items()})
                    continue
                n, a = parse_gt_string(s)
                if n:
                    gt_opts.append({n: {k: [v] for k, v in a.items()}})
        ok = bool(calls) and _call_match(calls[0], gt_opts)
        detail.append({"turn": ti, "ok": ok,
                       "calls": [(n, a) for n, a in calls]})
        if calls:
            valids.append(all(_arg_valid(n, a, spec_by_name) for n, a in calls))
        if ok:
            turns_ok += 1
        history = history + [{"role": "user", "content": text},
                             {"role": "assistant", "content": raw[:2000]}]
    passed = turns_ok == len(gt_turns) and len(gt_turns) > 0
    return {
        "id": case["id"], "version": case.get("version"),
        "category": case.get("category"),
        "prompt": _user_text([questions[0]])[:200] if questions else "",
        "expected_tool": "multi-turn",
        "tp": 1 if passed else 0, "fp": 0 if passed else 1,
        "fn": 0 if passed else 1,
        "first_tp": 1 if passed else 0,
        "arg_valid": (all(valids) if valids else None),
        "retries": retries, "steps": steps, "attempts": 1,
        "passed": passed, "turns_ok": turns_ok,
        "turns_total": len(gt_turns), "detail": detail,
    }


async def main_async(args, cases):
    from src.agent.chat_model import LocalChatModel
    from src.models.llm import LlmConfig, LlmModel

    llm = LlmModel(LlmConfig(model=args.model, backend=args.backend))
    print(f"warming {args.model} [{args.backend}] ...", flush=True)
    await llm.warm()
    from src.models.llm import assert_ready_leg

    print(f"ready on {assert_ready_leg(llm)}.", flush=True)
    cm = LocalChatModel(llm=llm)
    base = int(getattr(getattr(llm, "config", None), "max_tokens", 48) or 48)
    is_minicpm = str(getattr(getattr(llm, "config", None), "backend", "")).startswith("minicpm")
    max_tokens = max(base, 320) if is_minicpm else max(base, 256)

    rows = []
    for case in cases:
        est = estimate_case_tokens(case, max_tokens)
        if est > PROMPT_TOKEN_BUDGET:
            row = skipped_row(case, f"prompt ~{est} tok > {PROMPT_TOKEN_BUDGET} budget")
            rows.append(row)
            print(f"[SKIP] {row['id']:24s} {row['status_reason']}", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            if case.get("category") == "multi_turn":
                row = await run_multiturn(cm, llm, case, max_tokens)
            else:
                row = await run_single(cm, llm, case, max_tokens, args.k)
            row["seconds"] = round(time.perf_counter() - t0, 2)
            row["status"] = "ok"
        except Exception as exc:
            row = error_row(case, f"{type(exc).__name__}: {exc}",
                            time.perf_counter() - t0)
            print(f"[ERROR] {row['id']:24s} {row['status_reason']}", flush=True)
            rows.append(row)
            continue
        rows.append(row)
        first_calls = row.get("first_calls") or []
        got = first_calls[0][0] if first_calls else "none"
        want = row.get("expected_tool") or "none"
        mark = "OK " if row["passed"] else ("fp " if row["fp"] else "FAIL")
        print(f"[{mark}] {row['id']:24s} want={str(want):22s} got={str(got):22s} "
              f"retry={row['retries']} {row['seconds']:6.1f}s", flush=True)
    return rows, {"model": args.model, "backend": args.backend,
                  "max_tokens": max_tokens, "k": args.k}


def summarize(rows):
    tp = sum(r.get("first_tp", 0) for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    valids = [r["arg_valid"] for r in rows if r["arg_valid"] is not None]
    retries = sum(r["retries"] for r in rows)
    steps = [r.get("steps", 1) for r in rows]
    dts = sorted(r["seconds"] for r in rows)
    irrel = [r for r in rows if r.get("expect_none")]
    skipped = sum(1 for r in rows if r.get("status") == "skipped")
    errored = sum(1 for r in rows if r.get("status") == "error")
    return {
        "n": len(rows),
        "n_skipped": skipped,
        "n_errors": errored,
        "pass_at_k": round(sum(1 for r in rows if r["passed"]) / len(rows), 3) if rows else 0.0,
        "tool_precision": round(prec, 3),
        "tool_recall": round(rec, 3),
        "tool_f1": round(f1, 3),
        "irrelevance_acc": (round(sum(1 for r in irrel if r["passed"]) / len(irrel), 3)
                            if irrel else None),
        "irrelevance_n": len(irrel),
        "arg_validity_rate": (round(sum(valids) / len(valids), 3) if valids else None),
        "arg_valid_n": len(valids),
        "retries": retries,
        "retry_rate": round(retries / len(rows), 3) if rows else 0.0,
        "mean_steps": round(sum(steps) / len(steps), 2) if steps else 0.0,
        "median_seconds": dts[len(dts) // 2] if dts else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="real BFCL V1/V2/V3 offline eval")
    ap.add_argument("--manifests", nargs="+",
                    default=["benchmarks/bfcl_v1.json", "benchmarks/bfcl_v2.json",
                             "benchmarks/bfcl_v3.json"])
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--backend", default="gemma")
    ap.add_argument("--tag", default="gemma")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--categories", default="",
                    help="comma list to keep, e.g. simple,multiple "
                         "(default: all categories in the manifests)")
    ap.add_argument("--resume-from", default="",
                    help="prior results JSON: skip ids already present "
                         "(chunked multi-session full runs)")
    args = ap.parse_args()
    cases = []
    for path in args.manifests:
        cases.extend(load_manifest(path))
    # V1 hand grades join
    try:
        with open("benchmarks/bfcl_v1_grades.json", encoding="utf-8") as f:
            grades = json.load(f)
        for c in cases:
            if c.get("expected") is None and c.get("id") in grades:
                c["expected"] = grades[c["id"]]
                c["expected_kind"] = "hand"
    except Exception:
        pass
    if args.offset:
        cases = cases[args.offset:]
    if args.limit:
        cases = cases[:args.limit]
    cats = [c.strip() for c in (args.categories or "").split(",") if c.strip()]
    done = load_done_ids(args.resume_from) if args.resume_from else set()
    n0 = len(cases)
    cases = filter_cases(cases, categories=cats or None, done=done or None)
    print(f"cases: {len(cases)} (from {n0}"
          f"{', categories=' + ','.join(cats) if cats else ''}"
          f"{', resumed-skipped=' + str(n0 - len(cases)) if done else ''})",
          flush=True)
    rows, meta = asyncio.run(main_async(args, cases))
    summary = summarize(rows)
    print("---")
    for k, v in summary.items():
        print(f"{k:18s} {v}")
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = f"benchmarks/results/bfcl_{args.tag}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
