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
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/tmp/bfcl_raw")
    ap.add_argument("--out-dir", default="benchmarks")
    args = ap.parse_args()
    R = args.raw_dir

    manifests = {"v1": [], "v2": [], "v3": []}

    # ---- V1: static files, no machine answers -> hand-grade later ----
    v1_plan = [
        ("v1/gorilla_openfunctions_v1_test_simple.json", 8, "simple"),
        ("v1/gorilla_openfunctions_v1_test_multiple_function.json", 4, "multiple"),
    ]
    sheet = []
    for rel, k, cat in v1_plan:
        entries = load_jsonl(os.path.join(R, rel))
        for i in stride(len(entries), k):
            e = entries[i]
            fn = e.get("function")
            manifests["v1"].append({
                "id": e.get("id", f"v1-{i}"), "version": "v1",
                "category": cat, "file": os.path.basename(rel), "index": i,
                "question": e.get("question"),
                "functions": [fn] if isinstance(fn, dict) else (fn or []),
                "expected": None, "expected_kind": "hand",
            })
            sheet.append((e.get("id", f"v1-{i}"), e.get("question"), fn))

    # ---- V2: real ground truth ----
    v2_answers = {}
    for name in ("BFCL_v2_simple.json", "BFCL_v2_multiple.json"):
        v2_answers.update(load_answers(os.path.join(R, "v2/possible_answer", name)))
    v2_plan = [
        ("v2/BFCL_v2_simple.json", 10, "simple"),
        ("v2/BFCL_v2_multiple.json", 6, "multiple"),
        ("v2/BFCL_v2_irrelevance.json", 5, "irrelevance"),
    ]
    for rel, k, cat in v2_plan:
        entries = load_jsonl(os.path.join(R, rel))
        for i in stride(len(entries), k):
            e = entries[i]
            gt = v2_answers.get(e.get("id"))
            manifests["v2"].append({
                "id": e.get("id"), "version": "v2", "category": cat,
                "file": os.path.basename(rel), "index": i,
                "question": e.get("question"),
                "functions": e.get("function", []),
                "expected": gt, "expected_kind": "none" if gt is None else "ast",
            })

    # ---- V3: real ground truth, incl. multi-turn ----
    v3_answers = {}
    for name in ("v3_BFCL_v3_simple.json", "v3_BFCL_v3_multiple.json"):
        v3_answers.update(load_answers(os.path.join(R, name)))
    v3_plan = [
        ("BFCL_v3_simple.json", 8, "simple"),
        ("BFCL_v3_multiple.json", 6, "multiple"),
        ("v3_irrelevance.json", 5, "irrelevance"),
    ]
    for rel, k, cat in v3_plan:
        entries = load_jsonl(os.path.join(R, rel))
        for i in stride(len(entries), k):
            e = entries[i]
            gt = v3_answers.get(e.get("id"))
            manifests["v3"].append({
                "id": e.get("id"), "version": "v3", "category": cat,
                "file": os.path.basename(rel), "index": i,
                "question": e.get("question"),
                "functions": e.get("function", []),
                "expected": gt, "expected_kind": "none" if gt is None else "ast",
            })
    # multi-turn: resolve involved classes against func docs, embed specs
    specs, file_of = load_func_docs(os.path.join(R, "funcdoc"))
    by_file: dict = {}
    for name, fn in file_of.items():
        by_file.setdefault(fn, []).append(name)
    mt = load_jsonl(os.path.join(R, "v3_multi_turn_base.json"))
    mt_answers = load_answers(os.path.join(R, "v3_BFCL_v3_multi_turn_base.json"))
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
