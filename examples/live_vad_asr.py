#!/usr/bin/env python3
"""Live VAD + ASR demo: realtime voice-activity display, STT per segment.

VAD + STT legs only (no LLM, no TTS — warm takes seconds, not minutes).

Sources:
  --wav speech.wav   stream a file paced in realtime (default test path)
  --mic              capture from the default ALSA device (arecord subprocess)

An incremental state machine over VadModel.prob() prints live speech bars
per chunk, announces segment open/close, and transcribes each committed
segment with SttModel. Ctrl-C stops and prints a summary.

Usage:
  PYTHONPATH=. .venv/bin/python examples/live_vad_asr.py --wav in.wav
  PYTHONPATH=. .venv/bin/python examples/live_vad_asr.py --mic --seconds 20
"""

import argparse
import asyncio
import subprocess
import sys
import time

import numpy as np

SR = 16000
LINE_CHUNKS = 16  # one display line per ~0.5 s at 512-sample chunks


def wav_source(path, chunk):
    import soundfile as sf

    from engine.audio import resample

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32).ravel()
    if int(sr) != SR:
        wav = np.asarray(resample(wav, int(sr), SR), dtype=np.float32)
    for i in range(0, len(wav), chunk):
        yield wav[i:i + chunk]
        time.sleep(chunk / SR)  # realtime pacing


def mic_source(chunk):
    cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1",
           "-t", "raw"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise SystemExit("arecord not found; install alsa-utils or use --wav")
    want = int(chunk) * 2
    try:
        while True:
            raw = proc.stdout.read(want)
            if not raw:
                break
            yield (np.frombuffer(raw, dtype="<i2").astype(np.float32)
                   / 32768.0)
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


def bar(prob, thresh):
    return "#" if prob >= thresh else "."


async def main_async(args) -> int:
    from src.models.stt import SttModel
    from src.models.vad import VadConfig, VadModel

    vad = VadModel(VadConfig(threshold=args.threshold,
                             min_speech_s=args.min_speech,
                             min_sil_s=args.min_sil))
    stt = SttModel()
    t0 = time.perf_counter()
    await vad.warm()
    await stt.warm()
    # STT warmup transcribe: first call pays Triton autotune (~2 s); do it
    # here so segment #1 isn't slow. Content-independent (kernel configs).
    try:
        await stt.transcribe(np.zeros(SR, dtype=np.float32), SR)
    except Exception:
        pass
    print(f"VAD+STT warmed in {time.perf_counter() - t0:.1f}s "
          f"(thresh={args.threshold} min_speech={args.min_speech}s "
          f"min_sil={args.min_sil}s)", flush=True)
    vad.reset()

    chunk = int(args.chunk)
    need_speech = max(1, int(round(args.min_speech / (chunk / SR))))
    need_sil = max(1, int(round(args.min_sil / (chunk / SR))))
    if args.mic:
        print("listening on mic (Ctrl-C to stop)...", flush=True)
        stream = mic_source(chunk)
    else:
        print(f"streaming {args.wav} paced realtime...", flush=True)
        stream = wav_source(args.wav, chunk)

    t_start = time.perf_counter()
    deadline = t_start + float(args.seconds) if args.seconds > 0 else 0.0
    n_chunk = 0
    n_samples = 0  # audio clock: stamps stay true when STT blocks the loop
    line = []
    last_prob = 0.0
    in_seg = False
    run_speech = run_sil = 0
    seg_buf: list = []
    preroll: list = []  # rolling ~1 s so late opens don't clip first words
    PREROLL_N = max(1, int(round(1.0 / (chunk / SR))))
    seg_start = 0.0
    segments: list = []
    stt_ms = 0.0

    async def close_segment(t_now):
        nonlocal in_seg, run_speech, run_sil, seg_buf, stt_ms
        dur = (sum(len(c) for c in seg_buf) / SR) if seg_buf else 0.0
        audio = np.concatenate(seg_buf) if seg_buf else np.zeros(0)
        in_seg, run_speech, run_sil, seg_buf = False, 0, 0, []
        if dur < args.min_speech:
            print(f"  x dropped blip ({dur:.2f}s < min_speech)", flush=True)
            return
        t = time.perf_counter()
        try:
            res = await stt.transcribe(audio, SR)
            text = str(res.text or "").strip()
        except Exception as exc:
            text = f"[STT error: {exc}]"
        dt = (time.perf_counter() - t) * 1000.0
        stt_ms += dt
        segments.append((seg_start, t_now, text))
        print(f"  ■ END {t_now:6.2f}s  ({dur:.2f}s speech)  "
              f"ASR {dt:.0f}ms: {text!r}", flush=True)

    try:
        for piece in stream:
            if deadline and time.perf_counter() >= deadline:
                break
            frame = np.asarray(piece, dtype=np.float32).ravel()
            if frame.size == 0:
                continue
            # ONE stateful ONNX probe per chunk: the displayed number IS
            # the number the open/close machine decides on.
            try:
                leg = vad._backend()
                last_prob = float(await asyncio.to_thread(leg.prob, frame))
            except Exception:
                last_prob = 0.0
            p = last_prob >= args.threshold
            n_samples += len(frame)
            t_now = n_samples / SR  # audio clock, not wall clock
            preroll.append(frame)
            if len(preroll) > PREROLL_N:
                del preroll[0]
            line.append(bar(last_prob, args.threshold))
            n_chunk += 1

            if p:
                run_speech += 1
                run_sil = 0
            else:
                run_speech = 0
                run_sil += 1

            if not in_seg and run_speech >= need_speech:
                in_seg = True
                seg_buf = list(preroll[:-1])  # include pre-roll, not this frame twice
                seg_start = max(0.0, t_now - sum(len(c) for c in seg_buf) / SR)
                print(f"  ▶ SPEECH @ {t_now:6.2f}s", flush=True)
            if in_seg:
                seg_buf.append(frame)
                if run_sil >= need_sil:
                    await close_segment(t_now)

            if n_chunk % LINE_CHUNKS == 0:
                print(f"t={t_now:6.2f}s [{''.join(line)}] p={last_prob:.2f}",
                      flush=True)
                line = []
    except KeyboardInterrupt:
        print("\nstopped by user", flush=True)
    finally:
        if in_seg:
            await close_segment(n_samples / SR)

    total = n_samples / SR
    speech_s = sum(b - a for a, b, _ in segments)
    print(f"\n=== {len(segments)} segment(s), {speech_s:.2f}s speech in "
          f"{total:.1f}s audio, STT total {stt_ms:.0f}ms ===", flush=True)
    for a, b, text in segments:
        print(f"  [{a:6.2f} - {b:6.2f}] {text!r}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="live VAD + ASR demo")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", default="", help="wav file to stream realtime")
    src.add_argument("--mic", action="store_true", help="capture via arecord")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after SECS (0 = until EOF/Ctrl-C)")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-speech", type=float, default=0.25)
    ap.add_argument("--min-sil", type=float, default=0.30)
    args = ap.parse_args()
    try:
        return asyncio.run(main_async(args))
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
