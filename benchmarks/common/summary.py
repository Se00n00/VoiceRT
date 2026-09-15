"""Generate results/summary.json/csv and summary table."""
import json
import csv
from pathlib import Path
from collections import defaultdict

def _load(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def _avg(rows, key, backend):
    vals = [float(r[key]) for r in rows if r.get("backend")==backend and r.get(key)]
    return sum(vals)/len(vals) if vals else 0

def generate_summary(results_dir: Path):
    results_dir = Path(results_dir)
    rows = []

    # LLM
    llm = _load(results_dir / "llm_latency.csv")
    if llm:
        # pick batch 1, seq 512, out 64 median
        for be in ("pytorch","triton"):
            # find representative
            cand = [r for r in llm if r.get("backend")==be]
            if not cand:
                continue
            # use median of medians
            try:
                median = sum(float(r["median_ms"]) for r in cand)/len(cand)
                ttft = sum(float(r.get("ttft_p50_ms",0)) for r in cand)/len(cand)
                tpot = sum(float(r.get("tpot_p50_ms",0)) for r in cand)/len(cand)
                tput = sum(float(r.get("throughput",0)) for r in cand)/len(cand)
            except Exception:
                continue
        # Build per-metric rows
        # Need pytorch vs triton speedup
        def speedup(p, t):
            return (p/t) if t else 0
        # Collect
        # LLM TTFT p50
        p_torch = _avg(llm, "ttft_p50_ms", "pytorch")
        p_triton = _avg(llm, "ttft_p50_ms", "triton")
        rows.append({"Model":"LLM","Metric":"TTFT p50 (ms)","PyTorch":f"{p_torch:.2f}","Triton":f"{p_triton:.2f}","Speedup":f"{speedup(p_torch,p_triton):.2f}x" if p_triton else "n/a"})
        p_torch = _avg(llm, "tpot_p50_ms", "pytorch")
        p_triton = _avg(llm, "tpot_p50_ms", "triton")
        rows.append({"Model":"LLM","Metric":"TPOT p50 (ms)","PyTorch":f"{p_torch:.3f}","Triton":f"{p_triton:.3f}","Speedup":f"{speedup(p_torch,p_triton):.2f}x"})
        p_torch = _avg(llm, "throughput", "pytorch")
        p_triton = _avg(llm, "throughput", "triton")
        rows.append({"Model":"LLM","Metric":"tok/s","PyTorch":f"{p_torch:.1f}","Triton":f"{p_triton:.1f}","Speedup":f"{(p_triton/p_torch if p_torch else 0):.2f}x"})
        # STT
    stt = _load(results_dir / "stt_latency.csv")
    if stt:
        p_torch = _avg(stt, "median_ms", "pytorch")
        p_triton = _avg(stt, "median_ms", "triton")
        rows.append({"Model":"STT","Metric":"latency median (ms)","PyTorch":f"{p_torch:.1f}","Triton":f"{p_triton:.1f}","Speedup":f"{p_torch/p_triton:.2f}x" if p_triton else "n/a"})
        p_torch = _avg(stt, "rtf", "pytorch")
        p_triton = _avg(stt, "rtf", "triton")
        rows.append({"Model":"STT","Metric":"RTF","PyTorch":f"{p_torch:.3f}","Triton":f"{p_triton:.3f}","Speedup":f"{p_torch/p_triton:.2f}x" if p_triton else "n/a"})
        p_torch = _avg(stt, "throughput", "pytorch")
        p_triton = _avg(stt, "throughput", "triton")
        rows.append({"Model":"STT","Metric":"audio s/s","PyTorch":f"{p_torch:.1f}","Triton":f"{p_triton:.1f}","Speedup":f"{p_triton/p_torch:.2f}x" if p_torch else "n/a"})
    tts = _load(results_dir / "tts_latency.csv")
    if tts:
        p_torch = _avg(tts, "median_ms", "pytorch")
        p_triton = _avg(tts, "median_ms", "triton")
        rows.append({"Model":"TTS","Metric":"latency median (ms)","PyTorch":f"{p_torch:.1f}","Triton":f"{p_triton:.1f}","Speedup":f"{p_torch/p_triton:.2f}x" if p_triton else "n/a"})
        p_torch = _avg(tts, "rtf", "pytorch")
        p_triton = _avg(tts, "rtf", "triton")
        rows.append({"Model":"TTS","Metric":"RTF","PyTorch":f"{p_torch:.3f}","Triton":f"{p_triton:.3f}","Speedup":f"{p_torch/p_triton:.2f}x" if p_triton else "n/a"})

    # Kernels speedup summary
    for prefix in ("llm_kernels","stt_kernels","tts_kernels"):
        path = results_dir / f"{prefix}_latency.csv"
        kro = _load(path)
        if not kro:
            continue
        # group by kernel
        grouped = defaultdict(dict)
        for r in kro:
            grouped[r["kernel"]][r["backend"]] = float(r.get("median_ms",0))
        for k, d in grouped.items():
            pt = d.get("pytorch",0)
            tr = d.get("triton",0)
            if pt and tr:
                rows.append({"Model": prefix, "Metric": f"{k} speedup", "PyTorch": f"{pt:.3f}ms", "Triton": f"{tr:.3f}ms", "Speedup": f"{pt/tr:.2f}x"})

    # Save
    if not rows:
        rows = [{"Model":"none","Metric":"no data","PyTorch":"-","Triton":"-","Speedup":"-"}]
    with open(results_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model","Metric","PyTorch","Triton","Speedup"])
        w.writeheader()
        w.writerows(rows)
    with open(results_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Summary saved to {results_dir}/summary.csv with {len(rows)} rows")
    for r in rows:
        print(r)
    return rows
