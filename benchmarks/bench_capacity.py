"""Capacity + efficiency benchmark: TTFT/TPO/throughput/GPU/cost.

Drives the served path (VoiceEngine, same kernels as /v1/*) across a
generation-length x concurrency matrix while polling nvidia-smi for GPU
utilization, power, and memory. Reports throughput, efficiency
(tok/s per watt, tok/s per GB), and cost per million tokens at a local
amortized GPU rate (default $0.05/hr, --cost-per-hour).

Outputs (benchmarks/results/):
  capacity_<ts>.json  machine-readable run (append-friendly)
  capacity.md         summary tables (regenerated each run)
  cap_*.png           graphs (needs matplotlib, else skipped gracefully)

Run: PYTHONPATH=. python benchmarks/bench_capacity.py [--quick] [--full]
     PYTHONPATH=. python scripts/benchmark.py capacity [--quick]
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PROMPTS = [
    "Reply in one short spoken sentence: hello, my name is Ada.",
    "Reply in one short spoken sentence: what is the future of AI inference?",
]

E2E_TEXT = "Hello, this is a voice pipeline capacity test."


# -- GPU telemetry (nvidia-smi subprocess, CPU-safe fallback) ------------
class NvTelemetry:
    """Poll nvidia-smi in a thread; summary() -> util/power/mem stats."""

    def __init__(self, interval_s=0.5):
        self.interval = float(interval_s)
        self._stop = threading.Event()
        self._th = None
        self.utils, self.powers, self.mems = [], [], []

    def _poll_once(self):
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,power.draw,memory.used",
                 "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                u, p, m = [x.strip() for x in
                           r.stdout.strip().splitlines()[0].split(",")]
                return float(u), float(p), float(m)
        except Exception:
            pass
        return None

    def _loop(self):
        while not self._stop.is_set():
            v = self._poll_once()
            if v is not None:
                u, p, m = v
                self.utils.append(u)
                self.powers.append(p)
                self.mems.append(m)
            self._stop.wait(self.interval)

    def __enter__(self):
        if self._poll_once() is not None:
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=3)

    @staticmethod
    def _p95(xs):
        if not xs:
            return 0.0
        o = sorted(xs)
        return o[min(len(o) - 1, int(0.95 * len(o)))]

    def summary(self):
        def mean(xs):
            return statistics.fmean(xs) if xs else 0.0
        return {"samples": len(self.utils),
                "gpu_util_mean": mean(self.utils),
                "gpu_util_p95": self._p95(self.utils),
                "power_w_mean": mean(self.powers),
                "mem_used_mb_max": max(self.mems) if self.mems else 0.0}


def _vram_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except Exception:
        pass
    return 0.0


def _p50(xs):
    return statistics.median(xs) if xs else 0.0


# -- LLM sweep ------------------------------------------------------------
def bench_llm(eng, genlens, concs, tele):
    """(genlen x conc) chat matrix. Returns row dicts."""
    import torch

    rows = []
    for gl in genlens:
        for cc in concs:
            ids_list = [eng.prompt_ids(p) for p in PROMPTS]
            out = []

            def one(ids):
                t0 = time.perf_counter()
                with torch.no_grad():
                    r = eng.llm.generate(ids, max_new_tokens=gl)
                wall = time.perf_counter() - t0
                n = len(r["ids"])
                tpo = ((wall - r["ttft"]) / max(n - 1, 1)) if n > 1 else 0.0
                return {"ttft": r["ttft"], "n": n, "wall": wall,
                        "tpo": tpo, "tps": n / max(wall, 1e-9)}

            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=cc) as ex:
                futs = [ex.submit(one, ids) for ids in
                        (ids_list * ((cc + len(ids_list) - 1) // len(ids_list)))[:cc]]
                out = [f.result() for f in futs]
            wall = time.perf_counter() - t0
            toks = sum(o["n"] for o in out)
            tele_sum = tele.summary()
            rows.append({
                "genlen": gl, "conc": cc, "reqs": len(out),
                "ttft_p50_ms": _p50([o["ttft"] for o in out]) * 1000,
                "tpo_ms": statistics.fmean([o["tpo"] for o in out]) * 1000,
                "tps_mean": statistics.fmean([o["tps"] for o in out]),
                "thr_req_s": len(out) / max(wall, 1e-9),
                "thr_tok_s": toks / max(wall, 1e-9),
                "vram_mb": _vram_mb(),
                "gpu_util_mean": tele_sum["gpu_util_mean"],
                "gpu_util_p95": tele_sum["gpu_util_p95"],
                "power_w_mean": tele_sum["power_w_mean"],
            })
            r = rows[-1]
            print(f"llm gl={gl:3d} cc={cc} ttft={r['ttft_p50_ms']:6.0f}ms "
                  f"tpo={r['tpo_ms']:5.1f}ms tps={r['tps_mean']:5.1f} "
                  f"thr={r['thr_tok_s']:5.1f}tok/s util={r['gpu_util_mean']:4.1f}% "
                  f"pwr={r['power_w_mean']:5.1f}W", flush=True)
    return rows


# -- E2E round-trip (TTS synth -> stream_turn) -----------------------------
def bench_e2e(eng):
    import numpy as np

    wav, sr = eng.speak(E2E_TEXT)["wav"], 24000
    x = np.asarray(wav, dtype=np.float32)
    # resample 24k -> 16k input (linear, no deps)
    idx = np.linspace(0, len(x) - 1, num=int(len(x) * 16000 / sr))
    audio = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    r = eng.stream_turn(audio, 16000)
    row = {"ttfa_ms": r["ttfa_s"] * 1000, "e2e_ms": r["total_s"] * 1000,
           "reply_chars": len(r["reply"]), "vram_mb": r.get("vram_mb", 0.0),
           "wav_s": len(r["wav"]) / 24000}
    print(f"e2e ttfa={row['ttfa_ms']:.0f}ms e2e={row['e2e_ms']:.0f}ms "
          f"reply={r['reply']!r} vram={row['vram_mb']:.0f}MB", flush=True)
    return row


# -- STT / TTS spot checks -------------------------------------------------
def bench_stt_tts(eng):
    import numpy as np

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16000 * 3) * 0.05).astype(np.float32)
    t0 = time.perf_counter()
    stt = eng.transcribe(audio, 16000)
    stt_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    s = eng.speak("Capacity spot check, one short sentence.")
    wav, sr = s["wav"], s["sr"]
    tts_ms = (time.perf_counter() - t0) * 1000
    row = {"stt_ms": stt_ms, "stt_rtf": stt.get("rtf", 0.0),
           "tts_ms": tts_ms,
           "tts_rtf": tts_ms / 1000 / max(len(wav) / float(sr), 1e-9)}
    print(f"stt {stt_ms:.0f}ms rtf={row['stt_rtf']:.3f} | "
          f"tts {tts_ms:.0f}ms rtf={row['tts_rtf']:.3f}", flush=True)
    return row


# -- cost + efficiency -----------------------------------------------------
def with_cost(rows, rate_per_hr):
    for r in rows:
        thr = max(r["thr_tok_s"], 1e-9)
        r["cost_per_1m_usd"] = 1e6 / thr / 3600 * rate_per_hr
        r["toks_per_watt"] = thr / max(r["power_w_mean"], 1e-9)
        r["toks_per_gb"] = thr / max(r["vram_mb"] / 1024, 1e-9)
    return rows


# -- outputs ---------------------------------------------------------------
def write_results(payload):
    os.makedirs(RESULTS, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(RESULTS, f"capacity_{ts}.json")
    with open(jp, "w") as f:
        json.dump(payload, f, indent=1)
    md = ["# Capacity benchmark", "",
          f"_GPU: {payload['meta']['gpu']} | "
          f"rate: ${payload['meta']['cost_per_hour_usd']}/hr | "
          f"{payload['meta']['ts']}_", "",
          "## LLM (TTFT / TPO / throughput / efficiency / cost)",
          "",
          "| genlen | conc | TTFT p50 | TPO | TPS | req/s | tok/s | util% | W | tok/s/W | $/1M |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in payload["llm"]:
        md.append(f"| {r['genlen']} | {r['conc']} | {r['ttft_p50_ms']:.0f}ms | "
                  f"{r['tpo_ms']:.1f}ms | {r['tps_mean']:.1f} | "
                  f"{r['thr_req_s']:.2f} | {r['thr_tok_s']:.1f} | "
                  f"{r['gpu_util_mean']:.1f} | {r['power_w_mean']:.1f} | "
                  f"{r['toks_per_watt']:.2f} | ${r['cost_per_1m_usd']:.4f} |")
    md += ["", "## E2E round-trip (TTS synth -> stream_turn)", "",
           f"TTFA {payload['e2e']['ttfa_ms']:.0f}ms, "
           f"E2E {payload['e2e']['e2e_ms']:.0f}ms, "
           f"VRAM {payload['e2e']['vram_mb']:.0f}MB", "",
           "## STT/TTS spot", "",
           f"STT {payload['spot']['stt_ms']:.0f}ms "
           f"(RTF {payload['spot']['stt_rtf']:.3f}), TTS "
           f"{payload['spot']['tts_ms']:.0f}ms "
           f"(RTF {payload['spot']['tts_rtf']:.3f})", ""]
    mp = os.path.join(RESULTS, "capacity.md")
    with open(mp, "w") as f:
        f.write("\n".join(md))
    return jp, mp


def write_graphs(payload):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); writing stdlib SVG instead")
        return write_svg_graphs(payload)
    outs = []
    llm = payload["llm"]
    gls = sorted({r["genlen"] for r in llm})
    ccs = sorted({r["conc"] for r in llm})

    def series(key):
        return {cc: [next(r[key] for r in llm
                           if r["genlen"] == gl and r["conc"] == cc)
                     for gl in gls] for cc in ccs}

    figs = [
        ("cap_ttft_vs_conc.png", "TTFT p50 vs concurrency", "conc",
         "TTFT p50 (ms)",
         lambda: [(f"genlen {gl}",
                   [next(r["ttft_p50_ms"] for r in llm
                         if r["genlen"] == gl and r["conc"] == cc)
                    for cc in ccs]) for gl in gls], ccs),
        ("cap_thr_vs_genlen.png", "Throughput vs generation length",
         "genlen", "tok/s",
         lambda: [(f"conc {cc}", series("thr_tok_s")[cc]) for cc in ccs], gls),
        ("cap_cost_vs_genlen.png", "Cost per 1M tokens vs generation length",
         "genlen", "$/1M tokens",
         lambda: [(f"conc {cc}", series("cost_per_1m_usd")[cc])
                  for cc in ccs], gls),
        ("cap_util_vs_thr.png", "GPU util vs throughput", "tok/s", "util %",
         lambda: [(f"genlen {gl}",
                   [(next(r["thr_tok_s"] for r in llm
                          if r["genlen"] == gl and r["conc"] == cc),
                     next(r["gpu_util_mean"] for r in llm
                          if r["genlen"] == gl and r["conc"] == cc))
                    for cc in ccs]) for gl in gls], None),
    ]
    for fname, title, xl, yl, fn, xs in figs:
        fig, ax = plt.subplots()
        for label, ys in fn():
            if xs is None:
                x, y = zip(*ys)
                ax.plot(x, y, "o-", label=label)
            else:
                ax.plot(xs, ys, "o-", label=label)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend()
        p = os.path.join(RESULTS, fname)
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        outs.append(p)
    print("graphs:", ", ".join(os.path.basename(p) for p in outs))
    return outs


def write_svg_graphs(payload):
    """Four stdlib-only SVG line charts (no third-party deps)."""
    llm = payload["llm"]
    gls = sorted({r["genlen"] for r in llm})
    ccs = sorted({r["conc"] for r in llm})
    COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    def svg(path, title, xl, yl, series, xs, fmt="{:.1f}"):
        W, H, P = 640, 400, 56
        allx = [x for _, pts in series for x, _ in pts]
        ally = [y for _, pts in series for _, y in pts]
        x0, x1 = min(allx), max(allx)
        y0, y1 = 0.0, max(ally) * 1.15 if ally else 1.0
        if x1 == x0:
            x1 = x0 + 1
        sx = lambda x: P + (x - x0) / (x1 - x0) * (W - 2 * P)
        sy = lambda y: H - P - (y - y0) / (y1 - y0) * (H - 2 * P)
        el = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
              f'height="{H}" font-family="sans-serif">',
              f'<text x="{W//2}" y="24" text-anchor="middle" '
              f'font-size="16" font-weight="bold">{title}</text>']
        el.append(f'<line x1="{P}" y1="{H-P}" x2="{W-P}" y2="{H-P}" '
                  f'stroke="#333"/>')
        el.append(f'<line x1="{P}" y1="{P}" x2="{P}" y2="{H-P}" stroke="#333"/>')
        el.append(f'<text x="{W//2}" y="{H-8}" text-anchor="middle" '
                  f'font-size="12">{xl}</text>')
        el.append(f'<text x="12" y="{H//2}" text-anchor="middle" '
                  f'font-size="12" transform="rotate(-90 12 {H//2})">{yl}</text>')
        for i, (label, pts) in enumerate(series):
            c = COLORS[i % len(COLORS)]
            p = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
            el.append(f'<polyline points="{p}" fill="none" stroke="{c}" '
                      f'stroke-width="2"/>')
            for x, y in pts:
                el.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" '
                          f'fill="{c}"><title>{label}: ({x}, {fmt.format(y)})'
                          f'</title></circle>')
            el.append(f'<text x="{W-P+8}" y="{P+18*i}" font-size="12" '
                      f'fill="{c}">{label}</text>')
        el.append('</svg>')
        with open(path, "w") as f:
            f.write("\n".join(el))
        return path

    outs = []
    outs.append(svg(os.path.join(RESULTS, "cap_ttft_vs_conc.svg"),
                    "TTFT p50 vs concurrency", "concurrency", "TTFT p50 (ms)",
                    [(f"genlen {gl}",
                      [(cc, next(r["ttft_p50_ms"] for r in llm
                                 if r["genlen"] == gl and r["conc"] == cc))
                       for cc in ccs]) for gl in gls], ccs, fmt="{:.0f}"))
    outs.append(svg(os.path.join(RESULTS, "cap_thr_vs_genlen.svg"),
                    "Throughput vs generation length", "genlen", "tok/s",
                    [(f"conc {cc}",
                      [(gl, next(r["thr_tok_s"] for r in llm
                                 if r["genlen"] == gl and r["conc"] == cc))
                       for gl in gls]) for cc in ccs], gls))
    outs.append(svg(os.path.join(RESULTS, "cap_cost_vs_genlen.svg"),
                    "Cost per 1M tokens vs generation length", "genlen",
                    "$/1M tokens",
                    [(f"conc {cc}",
                      [(gl, next(r["cost_per_1m_usd"] for r in llm
                                 if r["genlen"] == gl and r["conc"] == cc))
                       for gl in gls]) for cc in ccs], gls, fmt="{:.4f}"))
    outs.append(svg(os.path.join(RESULTS, "cap_util_vs_thr.svg"),
                    "GPU util vs throughput", "tok/s", "util %",
                    [(f"genlen {gl}",
                      [(next(r["thr_tok_s"] for r in llm
                             if r["genlen"] == gl and r["conc"] == cc),
                        next(r["gpu_util_mean"] for r in llm
                             if r["genlen"] == gl and r["conc"] == cc))
                       for cc in ccs]) for gl in gls], None))
    print("graphs:", ", ".join(os.path.basename(p) for p in outs))
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="genlens {16,48} x conc {1,2} (default matrix)")
    ap.add_argument("--full", action="store_true",
                    help="genlens {16,48,128} x conc {1,2,4}")
    ap.add_argument("--cost-per-hour", type=float, default=0.05)
    ap.add_argument("--no-graphs", action="store_true")
    args = ap.parse_args(argv)

    if args.full:
        genlens, concs = (16, 48, 128), (1, 2, 4)
    else:
        genlens, concs = (16, 48), (1, 2)

    try:
        from engine.engine import VoiceEngine
    except Exception as exc:
        print(f"OMITTED: VoiceEngine unavailable ({exc})")
        return 2
    try:
        eng = VoiceEngine()
    except Exception as exc:
        print(f"OMITTED: could not construct VoiceEngine ({exc})")
        return 2

    from runtime.capacity import probe_vram

    gpu = probe_vram()
    with NvTelemetry() as tele:
        llm_rows = bench_llm(eng, genlens, concs, tele)
        e2e_row = bench_e2e(eng)
        spot_row = bench_stt_tts(eng)
        tele_sum = tele.summary()
    with_cost(llm_rows, args.cost_per_hour)
    payload = {
        "meta": {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "gpu": gpu.get("name", "cpu"),
            "vram_total_mb": gpu.get("total_mb", 0.0),
            "cost_per_hour_usd": args.cost_per_hour,
            "telemetry": tele_sum,
            "capacity_plan": eng.sessions.stats(),
        },
        "llm": llm_rows,
        "e2e": e2e_row,
        "spot": spot_row,
    }
    jp, mp = write_results(payload)
    print(f"wrote {jp}\n      {mp}")
    if not args.no_graphs:
        write_graphs(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
