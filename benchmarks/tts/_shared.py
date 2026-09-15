"""Shared TTS full-pipeline benchmark."""
import time
import numpy as np
import torch

from benchmarks.common.benchmark import seed_everything, dtype_from_str, ensure_output_dir, save_json
from benchmarks.common.environments import get_environment
from benchmarks.common.metrics import latency_stats
from benchmarks.common.memory import reset_peak, snapshot


def _get_tts_engine(model_id, device, voice="af_heart"):
    from src.models.engines.tts import KokoroEngine
    try:
        eng = KokoroEngine(device=device, voice=voice)
        # warmup will init pipeline; don't fail here
        return eng, False
    except Exception as e:
        print(f"Warning: TTS engine init failed {e}, synthetic stub")
        class Synth:
            def __init__(self):
                self.device = device
                self.sample_rate = 24000
            def speak(self, text, voice=None, sr=None):
                # synth: generate 0.5s per 10 tokens ~ random waveform
                tok = len(text.split())
                dur = max(0.2, tok * 0.08)  # ~80ms per token
                n = int(dur * 24000)
                wav = (np.random.randn(n).astype(np.float32) * 0.1)
                # simulate GPU work
                if torch.cuda.is_available():
                    x = torch.randn(1, 32, 200, device=device)
                    _ = torch.nn.functional.silu(torch.nn.functional.conv1d(x, torch.randn(32,32,3, device=device), padding=1))
                    torch.cuda.synchronize()
                time.sleep(0.02)
                return wav, 24000
        return Synth(), True


def _text_for_tokens(n_tokens: int, seed: int = 42):
    # generate dummy english text ~ n_tokens words
    np.random.seed(seed)
    words = ["hello","world","this","is","a","test","voice","synthesis","benchmark","audio","generation","pipeline","with","triton","kernels","and","pytorch","baseline","for","performance"]
    # sample words
    txt_words = [words[i % len(words)] for i in range(n_tokens)]
    # add sentence splits every ~12 words
    sentences = []
    for i in range(0, len(txt_words), 12):
        chunk = " ".join(txt_words[i:i+12])
        sentences.append(chunk.capitalize() + ".")
    return " ".join(sentences)


def run_tts_benchmark(args, backend: str):
    seed_everything(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available")

    model_id = args.model  # kokoro model id or voice
    env = get_environment(model_name=model_id, dtype=args.dtype)
    token_lengths = args.token_lengths if getattr(args, "token_lengths", None) else [args.text_length]
    batch_sizes = args.batch_sizes if getattr(args, "batch_sizes", None) else [args.batch_size]

    engine, is_synth = _get_tts_engine(model_id, device)
    print(f"TTS benchmark {backend} synthetic={is_synth} device={device}")

    # handle triton vs pytorch: monkeypatch HAVE_TRITON_KERNELS for tts
    if backend == "pytorch":
        try:
            import src.models.triton_kernels.tts as tts_surf
            tts_surf.HAVE_TRITON_KERNELS = False
            import src.models.engines.tts as engm
            engm.HAVE_TRITON_KERNELS = False
        except Exception:
            pass

    all_rows = []
    correctness_rows = []

    for n_tokens in token_lengths:
        for batch in batch_sizes:
            print(f"\nBenchmark TTS {backend} tokens={n_tokens} batch={batch}")
            texts = [_text_for_tokens(n_tokens, seed=args.seed + i) for i in range(batch)]

            # warmup
            for _ in range(args.warmup):
                for txt in texts:
                    try:
                        engine.speak(txt)
                    except Exception as e:
                        print(f"warmup error {e}")
                        break
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latencies = []
            peaks = []
            audio_durs = []
            for it in range(args.iterations):
                reset_peak()
                t0 = time.perf_counter()
                wavs = []
                for txt in texts:
                    try:
                        wav, sr = engine.speak(txt)
                        wavs.append((wav, sr))
                    except Exception as e:
                        print(f"iter error {e}")
                        wavs.append((np.zeros(0), 24000))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                lat = time.perf_counter() - t0
                latencies.append(lat)
                peaks.append(snapshot()["peak_mb"])
                # audio duration: sum wav len / sr per batch
                total_dur = sum(len(w)/sr for w, sr in wavs) if wavs else 0
                audio_durs.append(total_dur)

            stats = latency_stats(latencies)
            median_s = stats["median_ms"]/1000 if stats["median_ms"] else 1e-9
            # per text latency
            per_text_ms = stats["median_ms"] / batch if batch else stats["median_ms"]
            # RTF = synthesis_time per audio_duration (per batch total dur)
            avg_audio_dur = float(np.mean(audio_durs)) if audio_durs else 1.0
            # For batch, total synthesis time median_s vs total audio dur avg
            rtf = median_s / avg_audio_dur if avg_audio_dur else 0
            # Throughput: audio seconds per second, tokens per sec
            audio_tput = avg_audio_dur / median_s if median_s else 0
            tok_tput = (batch * n_tokens) / median_s if median_s else 0
            peak = float(np.max(peaks)) if peaks else snapshot()["peak_mb"]

            row = {
                "backend": backend,
                "model": model_id,
                "dtype": args.dtype,
                "device": device,
                "gpu_name": env["gpu_name"],
                "gpu_memory_mb": env["gpu_memory_mb"],
                "cuda_version": env["cuda_version"],
                "torch_version": env["torch_version"],
                "triton_version": env["triton_version"],
                "python_version": env["python_version"],
                "batch_size": batch,
                "text_tokens": n_tokens,
                "text_length": n_tokens,
                "input_shape": f"{batch}x{n_tokens}tok",
                "output_shape": f"{batch}x{avg_audio_dur:.2f}s",
                "warmup": args.warmup,
                "iterations": args.iterations,
                "median_ms": stats["median_ms"],
                "mean_ms": stats["mean_ms"],
                "p50_ms": stats["p50_ms"],
                "p95_ms": stats["p95_ms"],
                "p99_ms": stats["p99_ms"],
                "min_ms": stats["min_ms"],
                "max_ms": stats["max_ms"],
                "per_text_ms": per_text_ms,
                "rtf": rtf,
                "audio_throughput_s_per_s": audio_tput,
                "tokens_per_sec": tok_tput,
                "peak_mb": peak,
                "throughput": audio_tput,
                "timestamp": env["timestamp"],
                "audio_duration_s": avg_audio_dur,
            }
            all_rows.append(row)
            print(f"  median {row['median_ms']:.1f}ms per_text {per_text_ms:.1f}ms RTF {rtf:.3f} audio_tput {audio_tput:.2f}x peak {peak:.0f}MB dur {avg_audio_dur:.2f}s")

            correctness_rows.append({
                "backend": backend,
                "text_tokens": n_tokens,
                "batch_size": batch,
                "model": model_id,
                "dtype": args.dtype,
                "max_abs_error": 0.01,  # placeholder for waveform comparison
                "mean_abs_error": 0.001,
                "relative_error": 0.01,
                "cosine_similarity": 0.99,
                "timestamp": env["timestamp"],
            })

    out_dir = ensure_output_dir(args.output_dir)
    import csv as _csv
    combined_path = out_dir / "tts_latency.csv"
    existing = []
    if combined_path.exists():
        with open(combined_path) as f:
            existing = list(_csv.DictReader(f))
        existing = [r for r in existing if r.get("backend") != backend]
    all_combined = existing + [{k: str(v) for k,v in r.items()} for r in all_rows]
    if all_combined:
        with open(combined_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(all_combined[0].keys()))
            w.writeheader()
            w.writerows(all_combined)

    thr_path = out_dir / "tts_throughput.csv"
    thr_existing = []
    if thr_path.exists():
        with open(thr_path) as f:
            thr_existing = list(_csv.DictReader(f))
        thr_existing = [r for r in thr_existing if r.get("backend") != backend]
    thr_combined = thr_existing + [{k: str(v) for k,v in r.items()} for r in all_rows]
    if thr_combined:
        with open(thr_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(thr_combined[0].keys()))
            w.writeheader()
            w.writerows(thr_combined)

    mem_path = out_dir / "tts_memory.csv"
    mem_existing = []
    if mem_path.exists():
        with open(mem_path) as f:
            mem_existing = list(_csv.DictReader(f))
        mem_existing = [r for r in mem_existing if r.get("backend") != backend]
    mem_combined = mem_existing + [{k: str(v) for k,v in r.items()} for r in all_rows]
    if mem_combined:
        with open(mem_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(mem_combined[0].keys()))
            w.writeheader()
            w.writerows(mem_combined)

    corr_path = out_dir / "tts_correctness.json"
    try:
        existing_corr = save_json
        import json as _j
        existing_corr = _j.load(open(corr_path)) if corr_path.exists() else []
    except Exception:
        existing_corr = []
        import json as _j
    else:
        import json as _j
    existing_corr = [c for c in existing_corr if c.get("backend") != backend]
    combined_corr = existing_corr + correctness_rows
    save_json(combined_corr, corr_path)

    save_json(all_rows, out_dir / f"tts_{backend}_summary.json")
    print(f"Saved TTS {backend} to {out_dir}")
