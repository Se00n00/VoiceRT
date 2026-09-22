#!/usr/bin/env python3
"""Sample BFCL V1/V2/V3 offline eval manifests (deterministic strided picks).

Reads raw BFCL JSONL (see README in benchmarks/results or re-download from
gorilla-llm/Berkeley-Function-Calling-Leaderboard at pinned revisions) and
writes:
  benchmarks/bfcl_v1.json  (V1 static files; expected=null -> hand-grade)
  benchmarks/bfcl_v2.json  (real ground_truth joined)
  benchmarks/bfcl_v3.json  (real ground_truth joined, incl. multi-turn)

V1 files carry no machine-readable answers, so V1 entries ship with
expected=null plus a printed grading sheet; grades are patched in by hand
and marked "hand". Everything else is byte-exact upstream ground truth.

Usage:
  PYTHONPATH=. .venv/bin/python benchmarks/sample_bfcl.py --raw-dir /tmp/bfcl_raw
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stride(n, k):
    """k deterministic indices spread across n entries."""
    if n <= k:
        return list(range(n))
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def load_answers(path):
    """possible_answer file -> {id: ground_truth}."""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            e = json.loads(line)
            out[e["id"]] = e.get("ground_truth")
    return out


def load_func_docs(d):
    """multi_turn_func_doc/*.jsonl -> {func_name: spec}.

    Returns (specs, file_of) where file_of maps func name -> source file.
    """
    specs, file_of = {}, {}
    if not os.path.isdir(d):
        return specs, file_of
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                payload = [json.loads(line) for line in f if line.strip()]
        except Exception:
            continue
        for spec in payload:
            if isinstance(spec, dict) and spec.get("name"):
                specs.setdefault(spec["name"], spec)
                file_of.setdefault(spec["name"], fn)
    return specs, file_of


# involved_classes in the sampled multi-turn cases -> func-doc files that
# define them (class names don't match filenames; map is explicit so a
# missing class fails loudly instead of silently starving the prompt).
CLASS_TO_FILES = {
    "GorillaFileSystem": ["gorilla_file_system.json"],
    "TwitterAPI": ["posting_api.json"],
    "VehicleControlAPI": ["vehicle_control.json"],
    "TradingBot": ["trading_bot.json"],
    "MessageAPI": ["message_api.json"],
    "TravelAPI": ["travel_booking.json"],
    "MathAPI": ["math_api.json"],
    "TicketAPI": ["ticket_api.json"],
}

# Full-dataset plan per version: (local file, answers file|None, category,
# upstream filename for provenance).
# Static files: V3 == main as of 2026-09-23 (verified identical to dataset
# commit 023218c, 2024-08-07). V1/V2 static were captured 2026-09-23 then
# vanished from upstream (404 at pinned rev AND main); local copies are
# checksummed in benchmarks/bfcl_raw.sha256. V3 live_* are user-contributed
# rolling data (no fixed revision) -> main, date recorded with results.
# Excluded deliberately: java/javascript (non-Python), exec_* (needs code
# execution), rest (no open answers file), multi_turn miss_* (no per-turn
# ground truth in open files), live_relevance/live_irrelevance (no machine
# answers). V1 has no machine answers at all -> stays at 12 hand-graded.
FULL_PLAN = {
    "v2": [
        ("simple", "simple", "simple", "BFCL_v2_simple.json"),
        ("multiple", "multiple", "multiple", "BFCL_v2_multiple.json"),
        ("parallel", "parallel", "parallel", "BFCL_v2_parallel.json"),
        ("parallel_multiple", "parallel_multiple", "parallel_multiple",
         "BFCL_v2_parallel_multiple.json"),
        ("irrelevance", None, "irrelevance", "BFCL_v2_irrelevance.json"),
        ("chatable", None, "chatable", "BFCL_v2_chatable.json"),
        ("sql", "sql", "sql", "BFCL_v2_sql.json"),
    ],
    "v3": [
        ("simple", "simple", "simple", "BFCL_v3_simple.json"),
        ("multiple", "multiple", "multiple", "BFCL_v3_multiple.json"),
        ("parallel", "parallel", "parallel", "BFCL_v3_parallel.json"),
        ("parallel_multiple", "parallel_multiple", "parallel_multiple",
         "BFCL_v3_parallel_multiple.json"),
        ("irrelevance", None, "irrelevance", "BFCL_v3_irrelevance.json"),
        ("chatable", None, "chatable", "BFCL_v3_chatable.json"),
        ("sql", "sql", "sql", "BFCL_v3_sql.json"),
        ("live_simple", "live_simple", "simple", "BFCL_v3_live_simple.json"),
        ("live_multiple", "live_multiple", "multiple",
         "BFCL_v3_live_multiple.json"),
        ("live_parallel", "live_parallel", "parallel",
         "BFCL_v3_live_parallel.json"),
        ("live_parallel_multiple", "live_parallel_multiple",
         "parallel_multiple", "BFCL_v3_live_parallel_multiple.json"),
    ],
}
FULL_MT = ["multi_turn_base", "multi_turn_composite", "multi_turn_long_context"]


def _mt_specs(R, specs, by_file, involved):
    """Resolve involved classes to embedded specs (asserts coverage)."""
    missing = [c for c in involved if c not in CLASS_TO_FILES]
    assert not missing, f"unmapped involved_classes: {missing}"
    names = []
    for c in involved:
        for fn in CLASS_TO_FILES[c]:
            names.extend(by_file.get(fn, []))
    return [specs[n] for n in names if n in specs]


def build_full(R):
    """Full V2+V3 manifests: every entry of every gradeable category."""
    out = {"v2": [], "v3": []}
    specs, file_of = load_func_docs(os.path.join(R, "funcdoc"))
    by_file: dict = {}
    for name, fn in file_of.items():
        by_file.setdefault(fn, []).append(name)
    for ver in ("v2", "v3"):
        answers = {}
        for local, ans, _cat, _up in FULL_PLAN[ver]:
            if ans is None:
                continue
            answers.update(load_answers(
                os.path.join(R, ver, "answers", f"{ans}.json")))
        for local, _ans, cat, upstream in FULL_PLAN[ver]:
            entries = load_jsonl(os.path.join(R, ver, "data", f"{local}.json"))
            for i, e in enumerate(entries):
                gt = answers.get(e.get("id")) if e.get("id") else None
                out[ver].append({
                    "id": e.get("id") or f"{cat}_{i}",
                    "version": ver, "category": cat,
                    "file": upstream, "index": i,
                    "question": e.get("question"),
                    "functions": e.get("function", []) or [],
                    "expected": gt,
                    "expected_kind": "none" if gt is None else "ast",
                })
    # multi-turn (v3 only): embed resolved specs per case
    for name in FULL_MT:
        entries = load_jsonl(os.path.join(R, "v3", "data", f"{name}.json"))
        gt_all = load_answers(os.path.join(R, "v3", "answers", f"{name}.json"))
        for i, e in enumerate(entries):
            involved = e.get("involved_classes") or []
            out["v3"].append({
                "id": e.get("id"), "version": "v3", "category": "multi_turn",
                "file": f"BFCL_v3_{name}.json", "index": i,
                "question": e.get("question"), "involved_classes": involved,
                "functions": _mt_specs(R, specs, by_file, involved),
                "expected": gt_all.get(e.get("id")),
                "expected_kind": "multiturn",
            })
    # orphans: entries that need GT but have none (would mis-score as
    # irrelevance) — drop them, loudly
    kept = {"v2": [], "v3": []}
    orphans = []
    for ver in out:
        for e in out[ver]:
            if (e["category"] not in ("irrelevance", "chatable")
                    and e["expected"] is None):
                orphans.append((ver, e["id"]))
            else:
                kept[ver].append(e)
    if orphans:
        print(f"WARNING: dropped {len(orphans)} GT-less entries (first: {orphans[:5]})")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/tmp/bfcl_raw")
    ap.add_argument("--out-dir", default="benchmarks")
    ap.add_argument("--full", action="store_true",
                    help="emit FULL V2+V3 manifests (all Python non-live "
                         "categories with open ground truth). V1 stays at "
                         "the 12 hand-graded cases (no machine answers).")
    args = ap.parse_args()
    R = args.raw_dir
    os.makedirs(args.out_dir, exist_ok=True)

    if args.full:
        full = build_full(R)
        for ver, entries in full.items():
            path = os.path.join(args.out_dir, f"bfcl_{ver}_full.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=1)
            print(f"wrote {path} ({len(entries)} cases)")
        from collections import Counter
        for ver, entries in full.items():
            print(f"  {ver}: {dict(Counter(e['category'] for e in entries))}")
        return

    manifests = {"v1": [], "v2": [], "v3": []}

    # ---- V1: static files, no machine answers -> hand-grade later ----
    # (upstream, k, category); "file" records the upstream filename.
    v1_plan = [
        ("v1/data/gorilla_openfunctions_v1_test_simple.json",
         "gorilla_openfunctions_v1_test_simple.json", 8, "simple"),
        ("v1/data/gorilla_openfunctions_v1_test_multiple_function.json",
         "gorilla_openfunctions_v1_test_multiple_function.json", 4, "multiple"),
    ]
    sheet = []
    for rel, upstream, k, cat in v1_plan:
        entries = load_jsonl(os.path.join(R, rel))
        for i in stride(len(entries), k):
            e = entries[i]
            fn = e.get("function")
            manifests["v1"].append({
                "id": e.get("id", f"v1-{i}"), "version": "v1",
                "category": cat, "file": upstream, "index": i,
                "question": e.get("question"),
                "functions": [fn] if isinstance(fn, dict) else (fn or []),
                "expected": None, "expected_kind": "hand",
            })
            sheet.append((e.get("id", f"v1-{i}"), e.get("question"), fn))

    # ---- V2: real ground truth ----
    v2_answers = {}
    for name in ("simple.json", "multiple.json"):
        v2_answers.update(load_answers(os.path.join(R, "v2/answers", name)))
    v2_plan = [
        ("v2/data/simple.json", "BFCL_v2_simple.json", 10, "simple"),
        ("v2/data/multiple.json", "BFCL_v2_multiple.json", 6, "multiple"),
        ("v2/data/irrelevance.json", "BFCL_v2_irrelevance.json", 5, "irrelevance"),
    ]
    for rel, upstream, k, cat in v2_plan:
        entries = load_jsonl(os.path.join(R, rel))
        for i in stride(len(entries), k):
            e = entries[i]
            gt = v2_answers.get(e.get("id"))
            manifests["v2"].append({
                "id": e.get("id"), "version": "v2", "category": cat,
                "file": upstream, "index": i,
                "question": e.get("question"),
                "functions": e.get("function", []),
                "expected": gt, "expected_kind": "none" if gt is None else "ast",
            })

    # ---- V3: real ground truth, incl. multi-turn ----
    v3_answers = {}
    for name in ("simple.json", "multiple.json"):
        v3_answers.update(load_answers(os.path.join(R, "v3/answers", name)))
    v3_plan = [
        ("v3/data/simple.json", "BFCL_v3_simple.json", 8, "simple"),
        ("v3/data/multiple.json", "BFCL_v3_multiple.json", 6, "multiple"),
        ("v3/data/irrelevance.json", "BFCL_v3_irrelevance.json", 5, "irrelevance"),
    ]
    for rel, upstream, k, cat in v3_plan:
        entries = load_jsonl(os.path.join(R, rel))
        for i in stride(len(entries), k):
            e = entries[i]
            gt = v3_answers.get(e.get("id"))
            manifests["v3"].append({
                "id": e.get("id"), "version": "v3", "category": cat,
                "file": upstream, "index": i,
                "question": e.get("question"),
                "functions": e.get("function", []),
                "expected": gt, "expected_kind": "none" if gt is None else "ast",
            })
    # multi-turn: resolve involved classes against func docs, embed specs
    specs, file_of = load_func_docs(os.path.join(R, "funcdoc"))
    by_file: dict = {}
    for name, fn in file_of.items():
        by_file.setdefault(fn, []).append(name)
    mt = load_jsonl(os.path.join(R, "v3", "data", "multi_turn_base.json"))
    mt_answers = load_answers(os.path.join(R, "v3", "answers", "multi_turn_base.json"))
    for i in stride(len(mt), 5):
        e = mt[i]
        involved = e.get("involved_classes") or []
        missing = [c for c in involved if c not in CLASS_TO_FILES]
        assert not missing, f"unmapped involved_classes: {missing}"
        fn_specs = []
        for c in involved:
            for fn in CLASS_TO_FILES[c]:
                fn_specs.extend(by_file.get(fn, []))
        fn_specs = [specs[n] for n in fn_specs if n in specs]
        manifests["v3"].append({
            "id": e.get("id"), "version": "v3", "category": "multi_turn",
            "file": "BFCL_v3_multi_turn_base.json", "index": i,
            "question": e.get("question"), "involved_classes": involved,
            "functions": fn_specs,
            "expected": mt_answers.get(e.get("id")),
            "expected_kind": "multiturn",
        })

    for ver, entries in manifests.items():
        path = os.path.join(args.out_dir, f"bfcl_{ver}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=1)
        print(f"wrote {path} ({len(entries)} cases)")

    print(f"\nfunc-doc specs resolved: {len(specs)}")
    print("\n=== V1 HAND-GRADING SHEET (fill expected ASTs) ===")
    for cid, q, fn in sheet:
        names = fn.get("name") if isinstance(fn, dict) else [f.get("name") for f in (fn or [])]
        print(f"\n--- {cid} ---\nQ: {q}\nfuncs: {names}")


if __name__ == "__main__":
    main()
