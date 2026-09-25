#!/usr/bin/env python3
"""Unified full-dataset eval CLI (HuggingFace-only, full runs by default).

  PYTHONPATH=. python -m eval.evaluate bfcl --version all --list-only
  PYTHONPATH=. python -m eval.evaluate bfcl --version all --tag kaggle
  PYTHONPATH=. python -m eval.evaluate gaia --level 1 --list-only
  PYTHONPATH=. python -m eval.evaluate gaia --level 1 --tag kaggle-gaia

--limit exists ONLY as a smoke flag (harness check). Any --limit > 0 prints
a SMOKE banner; real numbers always use the full set (--limit 0).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.common import ensure_dir, load_done_keys, timestamp, write_results


def _add_common(ap):
    ap.add_argument("--model", default="openbmb/MiniCPM5-1B")
    ap.add_argument("--backend", default="minicpm")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--cache-dir", default="/tmp/hf_eval")
    ap.add_argument("--results-dir", default="eval/results")
    ap.add_argument("--limit", type=int, default=0,
                    help="SMOKE ONLY: truncate to N cases (default 0 = full)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--list-only", action="store_true",
                    help="fetch + normalize + print counts, no GPU/model")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="CPU smoke mode: skip the CUDA abort (SLOW, smoke only)")


def cmd_bfcl(args):
    from eval.bfcl import loader, scorer

    versions = {"all": ["v1", "v2", "v3"], "v1": ["v1"],
                "v2": ["v2"], "v3": ["v3"]}[args.version]
    cats = [c.strip() for c in (args.categories or "").split(",") if c.strip()] or None
    cases, metas = loader.load_cases(
        versions, args.cache_dir, rev=args.hf_rev, categories=cats,
        v1_questions=args.v1_questions, v1_grades=args.v1_grades)
    print(f"cases: {len(cases)} metas={ {k: (m.get('n', len([c for c in cases if c.get('version') == k]))) for k, m in metas.items()} }",
          flush=True)
    if args.limit:
        print(f"SMOKE: --limit {args.limit} truncates the FULL set "
              f"({len(cases)}); do not report these numbers.", flush=True)
        cases = cases[args.offset:args.offset + args.limit]
    elif args.offset:
        cases = cases[args.offset:]
    if args.list_only:
        from collections import Counter

        print(Counter((c.get("version"), c.get("category")) for c in cases))
        for ver, meta in metas.items():
            print(f"{ver}: {meta}")
        return 0

    from eval.bfcl import runner

    resume_keys = load_done_keys(args.resume_from) if args.resume_from else set()
    if resume_keys:
        print(f"resume: skipping {len(resume_keys)} done keys", flush=True)
    rows, run_meta = runner.run(cases, args.model, args.backend,
                                args.max_seq, args.k, resume_keys,
                                args.allow_cpu)
    summary = scorer.summarize(rows)
    breakdown = scorer.version_breakdown(rows)
    print("---")
    for k, v in summary.items():
        print(f"{k:18s} {v}")
    print(breakdown)
    out = os.path.join(args.results_dir, f"bfcl_{args.tag}_{timestamp()}.json")
    write_results(out, {"meta": {**run_meta, "loader": metas,
                                 "versions": versions, "categories": cats,
                                 "smoke": bool(args.limit),
                                 "cpu": bool(args.allow_cpu)},
                        "summary": summary, "breakdown": breakdown,
                        "rows": rows})
    print(f"wrote {out}")
    return 0


def cmd_gaia(args):
    from eval.gaia import loader

    levels = {"all": [1, 2, 3], "1": [1], "2": [2], "3": [3]}[args.level]
    tasks, meta = loader.load_tasks(levels, args.cache_dir, split=args.split)
    print(f"tasks: {len(tasks)} {meta}", flush=True)
    if args.limit:
        print(f"SMOKE: --limit {args.limit} truncates the FULL set "
              f"({len(tasks)}); do not report these numbers.", flush=True)
        tasks = tasks[args.offset:args.offset + args.limit]
    elif args.offset:
        tasks = tasks[args.offset:]
    if args.list_only:
        for t in tasks[:5]:
            print(t["id"], f"L{t['level']}", t["prompt"][:100])
        return 0

    from eval.gaia import runner

    rows, run_meta = runner.run(tasks, args.model, args.backend, args.timeout,
                                args.allow_cpu)
    summary = runner.summarize(rows)
    print("---")
    for k, v in summary.items():
        print(f"{k:18s} {v}")
    out = os.path.join(args.results_dir,
                       f"agent_tasks_{args.tag}_{timestamp()}.json")
    write_results(out, {"meta": {**run_meta, "loader": meta,
                                 "smoke": bool(args.limit),
                                 "cpu": bool(args.allow_cpu)},
                        "summary": summary, "rows": rows})
    print(f"wrote {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="full-dataset eval CLI (HF-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bfcl", help="BFCL V1/V2/V3 incl. V3 multi-turn")
    _add_common(b)
    b.add_argument("--version", default="all", choices=["all", "v1", "v2", "v3"])
    b.add_argument("--categories", default="")
    b.add_argument("--k", type=int, default=1)
    b.add_argument("--max-seq", type=int, default=16384,
                   help="min-relevant context (8192 floor, 16384 multi-turn/GAIA, 32768 long-context)")
    b.add_argument("--hf-rev", default="main")
    b.add_argument("--v1-questions", default="")
    b.add_argument("--v1-grades", default="")
    b.set_defaults(fn=cmd_bfcl)

    g = sub.add_parser("gaia", help="GAIA agent tasks")
    _add_common(g)
    g.add_argument("--level", default="1", choices=["all", "1", "2", "3"])
    g.add_argument("--split", default="validation", choices=["validation", "test"])
    g.add_argument("--timeout", type=int, default=600)
    g.set_defaults(fn=cmd_gaia)

    args = ap.parse_args()
    ensure_dir(args.results_dir)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
