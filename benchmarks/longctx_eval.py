#!/usr/bin/env python3
"""Long-context eval: prefill scaling + needle retrieval per backend.

Part A (scale): synthetic prompts at --lengths -> TTFT, wall, VRAM peak.
Part B (needle): a planted fact at --depths of each length, asked back
with one short generate; scored by exact-substring recall. Greedy,
single-generate probes through LlmModel (no agent loop).

Prompt sizes are counted with the Qwen tokenizer as a ruler (sidecar
legs ship no local tokenizer); buckets are approximate by design.
Length cells above the backend context (qwen max_seq, bonsai_ctx) are
skipped with a reason row instead of failing.

Usage:
  PYTHONPATH=. .venv/bin/python benchmarks/longctx_eval.py [--backend qwen]
      [--lengths 1024 2048 4096 8192] [--depths 0.0 0.5 1.0]
Results: benchmarks/results/longctx_<backend>_<ts>.json
"""

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEEDLE = "The secret fruit is blueberry pancakes."
NEEDLE_KEY = "blueberry pancakes"

_HAY = ("The harbor master logs every arrival before noon. "
        "Gulls circle the lighthouse when the tide turns. "
        "Abrigde over the marsh carries the morning bus. "
        "The bakery on fifth street opens before dawn. ")


def _haystack_for(n_tokens: int) -> str:
    """Neutral filler of roughly n_tokens (~4 chars per token).

    Ruler-grade padding for length buckets, not a benchmark text.
    ..
    """
    reps = max(1, int(n_tokens * 4 / len(_HAY)) + 1)
    return (_HAY * reps)[: n_tokens * 4]


def _median(vals: list) -> float:
    """Median of a non-empty list, else 0.0.

    Sorting copy; never mutates the input.
    ..
    """
    s = sorted(vals)
    return float(s[len(s) // 2]) if s else 0.0


def _vram_peak() -> float:
    """Local peak VRAM in MB, or -1 when unmeasurable.

    Sidecar VRAM lives off-process; the local number is then partial.
    ..
    """
    try:
        from src.models.runtime import max_allocated_mb, reset_peak_stats
        reset_peak_stats()
        return float(max_allocated_mb())
    except Exception:
        return -1.0


async def _fit(llm, want_ids: int) -> tuple:
    """Filler grown to roughly want_ids, returning (text, actual_ids).

    One measure-and-top-up pass; buckets stay approximate on purpose.
    ..
    """
    text = _haystack_for(want_ids)
    ids = len(await llm.encode([{"role": "user", "content": text}]))
    if ids < int(want_ids * 0.8):
        text = _haystack_for(want_ids + (want_ids - ids))
        ids = len(await llm.encode([{"role": "user", "content": text}]))
    return text, ids


async def main_async(args) -> int:
    """Warm once, sweep lengths, print medians, save envelope.

    Returns process exit code (0 on full sweep).
    ..
    """
    from src.models.llm import LlmConfig, LlmModel

    llm = LlmModel(LlmConfig(model=args.model, backend=args.backend,
                             max_tokens=32))
    t0 = time.perf_counter()
    await llm.warm()
    print(f"warm done in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"ready: {LlmModel.assert_ready_leg(llm)}", flush=True)
    ctx = int(getattr(llm.config, "bonsai_ctx", 0) or 0) \
        if args.backend == "bonsai" else int(llm.config.max_seq)
    print(f"backend ctx: {ctx}", flush=True)

    rows: list = []
    for length in args.lengths:
        if length > ctx:
            rows.append({"part": "skip", "length_target": length,
                         "reason": f"over backend ctx {ctx}"})
            print(f"length {length}: SKIP (ctx {ctx})", flush=True)
            continue
        hay, prompt_ids = await _fit(llm, length)
        # Part A: prefill cost of the bare haystack.
        msgs = [{"role": "system", "content": "Reply with one word."},
                {"role": "user", "content": f"{hay}\nReply: ok."}]
        _vram_peak()
        t = time.perf_counter()
        r = await llm.generate(msgs, max_tokens=16)
        dt = time.perf_counter() - t
        peak = _vram_peak()
        rows.append({"part": "scale", "length_target": length,
                     "prompt_ids": prompt_ids, "ttft_s": round(r.ttft_s, 3),
                     "wall_s": round(dt, 3), "out_ids": len(r.output_ids),
                     "vram_peak_mb": round(peak, 0)})
        print(f"[scale] len~{length} ({prompt_ids} ids): "
              f"ttft={r.ttft_s:.2f}s wall={dt:.2f}s vram={peak:.0f}MB",
              flush=True)
        # Part B: needle at each depth of the same haystack.
        for depth in args.depths:
            at = int(len(hay) * float(depth))
            body = hay[:at] + " " + NEEDLE + " " + hay[at:]
            q = [{"role": "system",
                  "content": "Answer with the exact phrase."},
                 {"role": "user",
                  "content": f"{body}\nQuestion: What is the secret "
                             f"fruit? Reply with the exact phrase."}]
            t = time.perf_counter()
            out = await llm.generate(q, max_tokens=32)
            dt = time.perf_counter() - t
            hit = NEEDLE_KEY in str(out.text or "").lower()
            rows.append({"part": "needle", "length_target": length,
                         "depth": float(depth), "recall": bool(hit),
                         "ttft_s": round(out.ttft_s, 3),
                         "wall_s": round(dt, 3)})
            print(f"[needle] len~{length} d={float(depth):.2f}: "
                  f"{'HIT' if hit else 'MISS'} "
                  f"reply={str(out.text or '')[:60]!r}", flush=True)

    scale = [r for r in rows if r["part"] == "scale"]
    needles = [r for r in rows if r["part"] == "needle"]
    by_len = {}
    for r in scale:
        by_len.setdefault(r["length_target"], []).append(r["ttft_s"])
    summary = {
        "n_scale": len(scale), "n_needle": len(needles),
        "ttft_by_len": {str(k): round(_median(v), 3)
                        for k, v in sorted(by_len.items())},
        "recall_rate": round(sum(1 for r in needles if r["recall"])
                             / max(len(needles), 1), 3),
    }
    print(f"summary: {summary}", flush=True)
    meta = {"model": args.model, "backend": args.backend, "ctx": ctx,
            "lengths": list(args.lengths),
            "depths": [float(d) for d in args.depths]}
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join("benchmarks", "results",
                        f"longctx_{args.backend}_{ts}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": summary, "rows": rows}, fh,
                  indent=2)
    print(f"saved {out}", flush=True)
    return 0


def main() -> int:
    """Parse args and run the length sweep.

    Smoke mode runs one short length, one depth.
    ..
    """
    ap = argparse.ArgumentParser(description="long-context scale + needle")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--backend", default="qwen")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[1024, 2048, 4096, 8192])
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0])
    ap.add_argument("--smoke", action="store_true",
                    help="one length (256), one depth (0.5)")
    args = ap.parse_args()
    if args.smoke:
        args.lengths = [256]
        args.depths = [0.5]
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
