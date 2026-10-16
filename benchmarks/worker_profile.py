#!/usr/bin/env python3
"""Worker profile: Bonsai 27B TTFT/decode sweep over prompt x gen sizes.

Same shape as leg_profile.py but for the worker leg (backend="bonsai"):
warms once, then times LlmModel.generate per (prompt toks, max_tokens)
cell through the production path (sidecar chat, no agent loop). The
first rep of each cell is warmup (server first-touch) and excluded.

Prompt sizes are counted with the Qwen tokenizer as a ruler (sidecar
legs ship no local tokenizer); buckets are approximate by design.

Usage:
  PYTHONPATH=. .venv/bin/python benchmarks/worker_profile.py [--tag bonsai]
      [--max-tokens 32 128 512] [--reps 3] [--smoke]
Results: benchmarks/results/worker_profile_<tag>_<ts>.json
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT_TARGETS = (40, 500, 2000)

_FILLER = ("The river runs past the old mill every morning before sunrise. "
           "Birds gather on the wire and argue about the weather. ")


def _filler_for(n_tokens: int) -> str:
    """Filler text of roughly n_tokens (~4 chars per token).

    Ruler-grade padding for prompt buckets, not a benchmark text.
    ..
    """
    reps = max(1, int(n_tokens * 4 / len(_FILLER)) + 1)
    return (_FILLER * reps)[: n_tokens * 4]


def _median(vals: list) -> float:
    """Median of a non-empty list, else 0.0.

    Sorting copy; never mutates the input.
    ..
    """
    s = sorted(vals)
    return float(s[len(s) // 2]) if s else 0.0


async def main_async(args) -> int:
    """Warm once, sweep cells, print medians, save envelope.

    Returns process exit code (0 on full sweep).
    ..
    """
    from src.models.llm import LlmConfig, LlmModel

    llm = LlmModel(LlmConfig(model=args.model, backend=args.backend,
                             max_tokens=max(args.max_tokens)))
    t0 = time.perf_counter()
    await llm.warm()
    print(f"warm done in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"ready: {LlmModel.assert_ready_leg(llm)}", flush=True)

    rows: list = []
    for target in PROMPT_TARGETS:
        body = _filler_for(target)
        msgs = [{"role": "system", "content": "Reply briefly."},
                {"role": "user", "content": f"Question follows.\n{body}"}]
        prompt_ids = len(await llm.encode(msgs))
        print(f"prompt target~{target} -> {prompt_ids} ids", flush=True)
        for limit in args.max_tokens:
            cell: list = []
            for rep in range(args.reps + 1):
                t = time.perf_counter()
                r = await llm.generate(msgs, max_tokens=limit)
                dt = time.perf_counter() - t
                n = len(r.output_ids)
                row = {"prompt_target": target, "prompt_ids": prompt_ids,
                       "max_tokens": limit, "rep": rep,
                       "ttft_s": round(r.ttft_s, 3),
                       "tps": round(r.tps, 1), "out_ids": n,
                       "wall_s": round(dt, 3)}
                rows.append(row)
                tag = "WARMUP" if rep == 0 else f"rep{rep}"
                print(f"[{tag}] prompt~{target} gen{limit}: "
                      f"ttft={r.ttft_s:.2f}s tps={r.tps:.1f} "
                      f"wall={dt:.1f}s ids={n}", flush=True)
                if rep > 0:
                    cell.append(row)
            kept = [c for c in cell]
            print(f"cell prompt~{target} gen{limit}: "
                  f"med ttft={_median([c['ttft_s'] for c in kept]):.2f}s "
                  f"med tps={_median([c['tps'] for c in kept]):.1f} "
                  f"med wall={_median([c['wall_s'] for c in kept]):.1f}s",
                  flush=True)

    meas = [r for r in rows if r["rep"] > 0]
    summary = {"n": len(meas),
               "med_ttft_s": round(_median([r["ttft_s"] for r in meas]), 3),
               "med_tps": round(_median([r["tps"] for r in meas]), 1)}
    try:
        from src.models.runtime import max_allocated_mb
        summary["vram_peak_mb"] = round(float(max_allocated_mb()), 0)
        print(f"vram peak: {summary['vram_peak_mb']:.0f}MB", flush=True)
    except Exception:
        pass
    meta = {"model": args.model, "backend": args.backend,
            "prompts": list(PROMPT_TARGETS),
            "max_tokens": list(args.max_tokens), "reps": args.reps}
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join("benchmarks", "results",
                        f"worker_profile_{args.tag}_{ts}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, fh,
                  indent=2)
    print(f"saved {out}", flush=True)
    return 0


def main() -> int:
    """Parse args and run the sweep.

    Smoke mode runs one small cell for harness checks.
    ..
    """
    ap = argparse.ArgumentParser(description="Bonsai worker TTFT/decode sweep")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--backend", default="bonsai")
    ap.add_argument("--tag", default="bonsai")
    ap.add_argument("--max-tokens", type=int, nargs="+",
                    default=[32, 128, 512])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--smoke", action="store_true",
                    help="one small cell (prompt~40 x gen32)")
    args = ap.parse_args()
    if args.smoke:
        global PROMPT_TARGETS
        PROMPT_TARGETS = (40,)
        args.max_tokens = [32]
        args.reps = 1
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
