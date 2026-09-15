"""Generate plots from benchmark CSVs."""
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_csv(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def _to_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0

def plot_llm(results_dir: Path):
    llm = load_csv(results_dir / "llm_latency.csv")
    if not llm:
        print("No llm_latency.csv")
        return
    # Group by backend
    # TTFT vs input tokens (batch=1)
    # Filter batch 1
    for backend in ("pytorch","triton"):
        pass
    # Prepare data: for batch=1, plot median vs input_length for each backend
    data = defaultdict(list)  # (backend, input_len) -> median
    # Use dict input_len -> {pytorch, triton}
    grouped = defaultdict(dict)
    for row in llm:
        try:
            il = int(row.get("input_length", 0))
            bs = int(row.get("batch_size", 1))
            if bs != 1:
                continue
            be = row.get("backend","")
            grouped[il][be] = _to_float(row.get("median_ms",0))
        except Exception:
            continue
    if grouped:
        xs = sorted(grouped.keys())
        y_torch = [grouped[x].get("pytorch",0) for x in xs]
        y_triton = [grouped[x].get("triton",0) for x in xs]
        # TTFT vs input tokens: actually use ttft_p50_ms
        # For simplicity plot median (E2E) vs input tokens
        # But also plot TTFT
        ttft_torch = []
        ttft_triton = []
        for x in xs:
            # find rows for ttft
            for row in llm:
                if int(row.get("input_length",0))==x and int(row.get("batch_size",1))==1 and row.get("backend")=="pytorch":
                    ttft_torch.append(_to_float(row.get("ttft_p50_ms", row.get("median_ms",0))))
                if int(row.get("input_length",0))==x and int(row.get("batch_size",1))==1 and row.get("backend")=="triton":
                    ttft_triton.append(_to_float(row.get("ttft_p50_ms", row.get("median_ms",0))))
            # fallback if not found
        # Ensure lengths match — allow single backend
        if (len(ttft_torch)==len(xs) or len(ttft_triton)==len(xs)) and (ttft_torch or ttft_triton):
            plt.figure(figsize=(8,5))
            if len(ttft_torch)==len(xs):
                plt.plot(xs, ttft_torch, marker="o", label="PyTorch TTFT")
            if len(ttft_triton)==len(xs):
                plt.plot(xs, ttft_triton, marker="s", label="Triton TTFT")
            plt.xlabel("Input tokens")
            plt.ylabel("TTFT p50 (ms)")
            plt.title("LLM TTFT vs Input Tokens (batch=1)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "llm_ttft_vs_input.png", dpi=150)
            plt.close()
        # Latency vs input
        plt.figure(figsize=(8,5))
        plt.plot(xs, y_torch, marker="o", label="PyTorch")
        plt.plot(xs, y_triton, marker="s", label="Triton")
        plt.xlabel("Input tokens")
        plt.ylabel("E2E median (ms)")
        plt.title("LLM E2E vs Input Tokens (batch=1)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "plots" / "llm_latency_vs_input.png", dpi=150)
        plt.close()

    # TPOT vs batch size (seq len fixed)
    # Group by batch
    grouped_batch = defaultdict(dict)
    for row in llm:
        try:
            bs = int(row.get("batch_size",1))
            il = int(row.get("input_length",0))
            # pick most common il (e.g., 512)
            if il != 512 and len(grouped)!=0:
                continue
            grouped_batch[bs][row.get("backend","")] = _to_float(row.get("tpot_p50_ms", row.get("median_ms",0)))
        except Exception:
            continue
    if grouped_batch:
        xs = sorted(grouped_batch.keys())
        y_torch = [grouped_batch[x].get("pytorch",0) for x in xs]
        y_triton = [grouped_batch[x].get("triton",0) for x in xs]
        if any(y_torch) or any(y_triton):
            plt.figure(figsize=(8,5))
            plt.plot(xs, y_torch, marker="o", label="PyTorch TPOT")
            plt.plot(xs, y_triton, marker="s", label="Triton TPOT")
            plt.xlabel("Batch size")
            plt.ylabel("TPOT p50 (ms)")
            plt.title("LLM TPOT vs Batch Size")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "llm_tpot_vs_batch.png", dpi=150)
            plt.close()

    # Throughput vs concurrency
    conc = load_csv(results_dir / "llm_concurrency.csv")
    if conc:
        grouped = defaultdict(dict)
        for row in conc:
            try:
                c = int(row.get("concurrency",0))
                be = row.get("backend","triton")
                grouped[c][be] = _to_float(row.get("output_tokens_per_sec", row.get("throughput",0)))
            except Exception:
                continue
        if grouped:
            xs = sorted(grouped.keys())
            y = [grouped[x].get("triton", grouped[x].get("pytorch",0)) for x in xs]
            plt.figure(figsize=(8,5))
            plt.plot(xs, y, marker="o")
            plt.xlabel("Concurrency")
            plt.ylabel("Output tok/s")
            plt.title("LLM Throughput vs Concurrency")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "llm_throughput_vs_concurrency.png", dpi=150)
            plt.close()

    # Peak VRAM vs seq length
    if grouped:
        xs = sorted(set(int(r.get("input_length",0)) for r in llm if r.get("input_length")))
        # Use batch 1
        y_torch = []
        y_triton = []
        for x in xs:
            # average peak for that seq len
            vals_t = [_to_float(r.get("peak_mb",0)) for r in llm if int(r.get("input_length",0))==x and r.get("backend")=="pytorch"]
            vals_tr = [_to_float(r.get("peak_mb",0)) for r in llm if int(r.get("input_length",0))==x and r.get("backend")=="triton"]
            y_torch.append(sum(vals_t)/len(vals_t) if vals_t else 0)
            y_triton.append(sum(vals_tr)/len(vals_tr) if vals_tr else 0)
        if any(y_torch) or any(y_triton):
            plt.figure(figsize=(8,5))
            plt.plot(xs, y_torch, marker="o", label="PyTorch")
            plt.plot(xs, y_triton, marker="s", label="Triton")
            plt.xlabel("Sequence length")
            plt.ylabel("Peak VRAM (MB)")
            plt.title("LLM Peak VRAM vs Seq Len")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "llm_vram_vs_seq.png", dpi=150)
            plt.close()

def plot_stt(results_dir: Path):
    stt = load_csv(results_dir / "stt_latency.csv")
    if not stt:
        return
    # latency vs audio duration batch 1
    grouped = defaultdict(dict)
    for row in stt:
        try:
            dur = float(row.get("audio_duration_s",0))
            bs = int(row.get("batch_size",1))
            if bs != 1:
                continue
            be = row.get("backend","")
            grouped[dur][be] = _to_float(row.get("median_ms",0))
        except Exception:
            continue
    if grouped:
        xs = sorted(grouped.keys())
        y_t = [grouped[x].get("pytorch",0) for x in xs]
        y_tr = [grouped[x].get("triton",0) for x in xs]
        plt.figure(figsize=(8,5))
        plt.plot(xs, y_t, marker="o", label="PyTorch")
        plt.plot(xs, y_tr, marker="s", label="Triton")
        plt.xlabel("Audio duration (s)")
        plt.ylabel("Latency median (ms)")
        plt.title("STT Latency vs Audio Duration")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "plots" / "stt_latency_vs_duration.png", dpi=150)
        plt.close()
        # RTF vs duration
        rtfs_t = []
        rtfs_tr = []
        for x in xs:
            for row in stt:
                if float(row.get("audio_duration_s",0))==x and int(row.get("batch_size",1))==1 and row.get("backend")=="pytorch":
                    rtfs_t.append(_to_float(row.get("rtf",0)))
                if float(row.get("audio_duration_s",0))==x and int(row.get("batch_size",1))==1 and row.get("backend")=="triton":
                    rtfs_tr.append(_to_float(row.get("rtf",0)))
        if len(rtfs_t)==len(xs) or len(rtfs_tr)==len(xs):
            plt.figure(figsize=(8,5))
            if len(rtfs_t)==len(xs):
                plt.plot(xs, rtfs_t, marker="o", label="PyTorch RTF")
            if len(rtfs_tr)==len(xs):
                plt.plot(xs, rtfs_tr, marker="s", label="Triton RTF")
            plt.xlabel("Audio duration (s)")
            plt.ylabel("RTF (infer/dur) lower=better")
            plt.title("STT RTF vs Duration")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "stt_rtf_vs_duration.png", dpi=150)
            plt.close()
    # throughput vs batch
    # pick dur=5
    grouped_b = defaultdict(dict)
    for row in stt:
        try:
            if float(row.get("audio_duration_s",0)) not in (5.0, 5):
                continue
            bs = int(row.get("batch_size",0))
            grouped_b[bs][row.get("backend","")] = _to_float(row.get("audio_throughput_s_per_s", row.get("throughput",0)))
        except Exception:
            continue
    if grouped_b:
        xs = sorted(grouped_b.keys())
        y_t = [grouped_b[x].get("pytorch",0) for x in xs]
        y_tr = [grouped_b[x].get("triton",0) for x in xs]
        if any(y_t) or any(y_tr):
            plt.figure(figsize=(8,5))
            plt.plot(xs, y_t, marker="o", label="PyTorch")
            plt.plot(xs, y_tr, marker="s", label="Triton")
            plt.xlabel("Batch size")
            plt.ylabel("Audio s/s")
            plt.title("STT Throughput vs Batch")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "stt_throughput_vs_batch.png", dpi=150)
            plt.close()
    # peak VRAM vs audio duration (batch 1)
    grouped_vram = defaultdict(dict)
    for row in stt:
        try:
            dur = float(row.get("audio_duration_s",0))
            bs = int(row.get("batch_size",1))
            if bs != 1:
                continue
            grouped_vram[dur][row.get("backend","")] = _to_float(row.get("peak_mb",0))
        except Exception:
            continue
    if grouped_vram:
        xs = sorted(grouped_vram.keys())
        y_t = [grouped_vram[x].get("pytorch",0) for x in xs]
        y_tr = [grouped_vram[x].get("triton",0) for x in xs]
        if any(y_t) or any(y_tr):
            plt.figure(figsize=(8,5))
            plt.plot(xs, y_t, marker="o", label="PyTorch")
            plt.plot(xs, y_tr, marker="s", label="Triton")
            plt.xlabel("Audio duration (s)")
            plt.ylabel("Peak VRAM (MB)")
            plt.title("STT Peak VRAM vs Audio Duration")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "stt_vram_vs_duration.png", dpi=150)
            plt.close()

def plot_tts(results_dir: Path):
    tts = load_csv(results_dir / "tts_latency.csv")
    if not tts:
        return
    grouped = defaultdict(dict)
    for row in tts:
        try:
            toks = int(row.get("text_tokens", row.get("text_length",0)))
            bs = int(row.get("batch_size",1))
            if bs != 1:
                continue
            grouped[toks][row.get("backend","")] = _to_float(row.get("median_ms",0))
        except Exception:
            continue
    if grouped:
        xs = sorted(grouped.keys())
        y_t = [grouped[x].get("pytorch",0) for x in xs]
        y_tr = [grouped[x].get("triton",0) for x in xs]
        plt.figure(figsize=(8,5))
        plt.plot(xs, y_t, marker="o", label="PyTorch")
        plt.plot(xs, y_tr, marker="s", label="Triton")
        plt.xlabel("Text tokens")
        plt.ylabel("Latency median (ms)")
        plt.title("TTS Latency vs Text Length")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "plots" / "tts_latency_vs_tokens.png", dpi=150)
        plt.close()
        # RTF
        rtfs_t = []
        rtfs_tr = []
        for x in xs:
            for row in tts:
                if int(row.get("text_tokens",0))==x and int(row.get("batch_size",1))==1 and row.get("backend")=="pytorch":
                    rtfs_t.append(_to_float(row.get("rtf",0)))
                if int(row.get("text_tokens",0))==x and int(row.get("batch_size",1))==1 and row.get("backend")=="triton":
                    rtfs_tr.append(_to_float(row.get("rtf",0)))
        if len(rtfs_t)==len(xs) or len(rtfs_tr)==len(xs):
            plt.figure(figsize=(8,5))
            if len(rtfs_t)==len(xs):
                plt.plot(xs, rtfs_t, marker="o", label="PyTorch RTF")
            if len(rtfs_tr)==len(xs):
                plt.plot(xs, rtfs_tr, marker="s", label="Triton RTF")
            plt.xlabel("Text tokens")
            plt.ylabel("RTF")
            plt.title("TTS RTF vs Text Length")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "tts_rtf_vs_tokens.png", dpi=150)
            plt.close()
    # throughput vs batch (pick 50 tokens)
    grouped_b = defaultdict(dict)
    for row in tts:
        try:
            toks = int(row.get("text_tokens", row.get("text_length",0)))
            if toks != 50:
                continue
            bs = int(row.get("batch_size",0))
            grouped_b[bs][row.get("backend","")] = _to_float(row.get("audio_throughput_s_per_s", row.get("throughput",0)))
        except Exception:
            continue
    if grouped_b:
        xs = sorted(grouped_b.keys())
        y_t = [grouped_b[x].get("pytorch",0) for x in xs]
        y_tr = [grouped_b[x].get("triton",0) for x in xs]
        if any(y_t) or any(y_tr):
            plt.figure(figsize=(8,5))
            plt.plot(xs, y_t, marker="o", label="PyTorch")
            plt.plot(xs, y_tr, marker="s", label="Triton")
            plt.xlabel("Batch size")
            plt.ylabel("Audio s/s")
            plt.title("TTS Throughput vs Batch (50 tokens)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "tts_throughput_vs_batch.png", dpi=150)
            plt.close()
    # peak VRAM vs text length (batch 1)
    grouped_vram = defaultdict(dict)
    for row in tts:
        try:
            toks = int(row.get("text_tokens", row.get("text_length",0)))
            bs = int(row.get("batch_size",1))
            if bs != 1:
                continue
            grouped_vram[toks][row.get("backend","")] = _to_float(row.get("peak_mb",0))
        except Exception:
            continue
    if grouped_vram:
        xs = sorted(grouped_vram.keys())
        y_t = [grouped_vram[x].get("pytorch",0) for x in xs]
        y_tr = [grouped_vram[x].get("triton",0) for x in xs]
        if any(y_t) or any(y_tr):
            plt.figure(figsize=(8,5))
            plt.plot(xs, y_t, marker="o", label="PyTorch")
            plt.plot(xs, y_tr, marker="s", label="Triton")
            plt.xlabel("Text tokens")
            plt.ylabel("Peak VRAM (MB)")
            plt.title("TTS Peak VRAM vs Text Length")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(results_dir / "plots" / "tts_vram_vs_tokens.png", dpi=150)
            plt.close()

def plot_kernels(results_dir: Path):
    for prefix in ("llm_kernels","stt_kernels","tts_kernels"):
        path = results_dir / f"{prefix}_latency.csv"
        rows = load_csv(path)
        if not rows:
            continue
        # group by kernel
        grouped = defaultdict(dict)
        for r in rows:
            k = r.get("kernel","")
            be = r.get("backend","")
            grouped[k][be] = _to_float(r.get("median_ms",0))
        if not grouped:
            continue
        kernels = list(grouped.keys())
        y_t = [grouped[k].get("pytorch",0) for k in kernels]
        y_tr = [grouped[k].get("triton",0) for k in kernels]
        # latency bar
        plt.figure(figsize=(10,5))
        x = range(len(kernels))
        w=0.35
        plt.bar([i-w/2 for i in x], y_t, width=w, label="PyTorch")
        plt.bar([i+w/2 for i in x], y_tr, width=w, label="Triton")
        plt.xticks(list(x), kernels, rotation=20)
        plt.ylabel("Median ms")
        plt.title(f"{prefix} PyTorch vs Triton Latency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "plots" / f"{prefix}_latency.png", dpi=150)
        plt.close()
        # speedup
        speedups = [(y_t[i]/y_tr[i] if y_tr[i] else 0) for i in range(len(kernels))]
        plt.figure(figsize=(10,5))
        plt.bar(kernels, speedups)
        plt.axhline(1.0, color="red", linestyle="--")
        plt.xticks(rotation=20)
        plt.ylabel("Speedup (PyTorch/Triton) >1 Triton faster")
        plt.title(f"{prefix} Triton Speedup")
        plt.tight_layout()
        plt.savefig(results_dir / "plots" / f"{prefix}_speedup.png", dpi=150)
        plt.close()


def generate_all(results_dir: Path):
    results_dir = Path(results_dir)
    (results_dir / "plots").mkdir(parents=True, exist_ok=True)
    plot_llm(results_dir)
    plot_stt(results_dir)
    plot_tts(results_dir)
    plot_kernels(results_dir)
    print(f"Plots saved to {results_dir/'plots'}")
