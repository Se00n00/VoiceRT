"""Endpoint-only pipeline sweep (stdlib HTTP only).

Ported from VOICE/bench_api_e2e.py onto urllib + concurrent.futures
(no httpx/jiwer/numpy): each request POSTs one wav to POST /v1/voice
(server-side VAD->STT->LLM->TTS) and records server ttfa_s/total_s plus
client-side E2E. Concurrency sweep over c=1,2,4.

Wav payloads: synthetic 2s 16kHz PCM16 sine written with stdlib `wave`
(no deps), so plumbing is measurable without sample files.

Run: PYTHONPATH=voice-pipeline python benchmarks/benchmark_pipeline.py
       [--base-url URL] [--requests N]
"""
import argparse
import concurrent.futures
import io
import json
import math
import os
import statistics
import struct
import sys
import time
import urllib.request
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DEFAULT = "http://127.0.0.1:8003"


def synth_wav_bytes(dur_s=2.0, sr=16000, freq=440.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        n = int(dur_s * sr)
        for i in range(n):
            v = int(12000 * math.sin(2 * math.pi * freq * i / sr))
            w.writeframes(struct.pack("<h", v))
    return buf.getvalue()


def _encode_multipart(field, filename, data, ctype="audio/wav"):
    boundary = "---- voicepipe12345"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + data + tail
    return body, f"multipart/form-data; boundary={boundary}"


def one(base, data):
    t0 = time.perf_counter()
    try:
        body, ctype = _encode_multipart("f", "a.wav", data)
        req = urllib.request.Request(
            base + "/v1/voice", data=body,
            headers={"Content-Type": ctype}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            j = json.loads(resp.read().decode())
        e2e = time.perf_counter() - t0
        return {
            "ok": True,
            "ttfa": float(j.get("ttfa_s", e2e)),
            "total": float(j.get("total_s", e2e)),
            "e2e": e2e,
        }
    except Exception as exc:
        return {"ok": False, "ttfa": 0.0, "total": 0.0, "e2e": 0.0,
                "error": str(exc)}


def check_health(base):
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as resp:
            json.loads(resp.read().decode())
        return True
    except Exception as exc:
        print(f"server {base} not reachable ({exc})")
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=BASE_DEFAULT)
    ap.add_argument("--requests", type=int, default=8)
    args = ap.parse_args(argv)
    base = args.base_url.rstrip("/")

    if not check_health(base):
        print("OMITTED: server not up; start it with `make server` first")
        return 2
    payloads = [synth_wav_bytes(freq=f) for f in (440.0, 554.0, 659.0)]
    print(f"base={base} payloads={len(payloads)} (synthetic 2s sine)", flush=True)
    for c in (1, 2, 4):
        reqs = [payloads[i % len(payloads)] for i in range(args.requests)]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
            res = list(ex.map(lambda d: one(base, d), reqs))
        wall = time.perf_counter() - t0
        ok = [x for x in res if x["ok"]]
        if not ok:
            err = res[0].get("error", "?") if res else "?"
            print(f"c={c} ALL FAILED ({err})")
            continue
        ttfa = sorted(x["ttfa"] * 1000 for x in ok)
        tot = sorted(x["e2e"] * 1000 for x in ok)
        q = lambda a, p: a[min(int(p * len(a)), len(a) - 1)]  # noqa: E731
        print(
            f"c={c} ok={len(ok)}/{len(res)} "
            f"TTFA p50={q(ttfa, .5):.0f}ms p99={q(ttfa, .99):.0f}ms "
            f"E2E p50={q(tot, .5):.0f}ms p99={q(tot, .99):.0f}ms "
            f"mean_E2E={statistics.fmean(tot):.0f}ms "
            f"req/s={len(ok) / wall:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
