"""Scoring wrappers (single source of truth lives in benchmarks/bfcl_eval)."""
from collections import Counter


def summarize(rows):
    from benchmarks.bfcl_eval import summarize as _summarize

    return _summarize(rows)


def version_breakdown(rows):
    out = {}
    for ver in sorted({r.get("version", "?") for r in rows}):
        sub = [r for r in rows if r.get("version") == ver]
        passed = sum(1 for r in sub if r.get("passed"))
        cats = dict(Counter(r.get("category", "?") for r in sub))
        out[ver] = {"n": len(sub), "passed": passed,
                    "pass_rate": round(passed / len(sub), 3) if sub else 0.0,
                    "categories": cats}
    return out
