"""VRAM-aware capacity planning: probe the GPU, derive servable sessions.

On server startup the serving story is decided here, not hardcoded:

1. :func:`probe_vram` asks the system how much VRAM exists (nvidia-smi,
   torch.cuda fallback, CPU-safe zeros).
2. :func:`estimate_session_mb` prices one session from the generation
   length: the preallocated Qwen KV-cache + per-turn history working set
   + transient TTS scratch.
3. :func:`plan_capacity` divides usable VRAM (total − warm baseline −
   headroom) by the per-session price → ``max_sessions``, plus a measured
   ``max_inflight`` for concurrent turns.

All functions use only the stdlib except an optional lazy torch import,
so the math is unit-testable on CPU (see tests/runtime/test_capacity.py).
"""
import subprocess

HEADROOM_FRAC = 0.10
HEADROOM_MIN_MB = 400.0
PER_TURN_MB = 150.0
MAX_INFLIGHT_DEFAULT = 4
QUEUE_TIMEOUT_S = 10.0
SAFETY_FACTOR = 1.25
SESSIONS_CAP = 1000

# Qwen2.5-0.5B dims (must match models/qwen.py FALLBACK_DIMS).
QWEN_LAYERS = 24
QWEN_KV_HEADS = 2
QWEN_HEAD_DIM = 64
QWEN_MAX_LEN = 512
QWEN_HIDDEN = 896


def probe_vram():
    """Return {total_mb, free_mb, cuda, source, name}. Never raises."""
    out = {"total_mb": 0.0, "free_mb": 0.0, "cuda": False,
           "source": "none", "name": "cpu"}
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.free",
             "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            line = r.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                out.update(name=parts[0], total_mb=float(parts[1]),
                           free_mb=float(parts[2]), cuda=True,
                           source="nvidia-smi")
                return out
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(idx).total_memory
            out.update(total_mb=total / 1024 ** 2, cuda=True,
                       source="torch.cuda")
            try:
                out["name"] = torch.cuda.get_device_name(idx)
            except Exception:
                out["name"] = "cuda"
            return out
    except Exception:
        pass
    return out


def kv_cache_mb(layers=QWEN_LAYERS, kv_heads=QWEN_KV_HEADS,
                head_dim=QWEN_HEAD_DIM, max_len=QWEN_MAX_LEN,
                dtype_bytes=2):
    """Static Qwen KV-cache: 2 (K+V) x L x Hk x max_len x dh x bytes."""
    return (2.0 * layers * kv_heads * max_len * head_dim
            * dtype_bytes / 1024 ** 2)


def estimate_session_mb(max_new_tokens=48, max_turns=20, prompt_tokens=200,
                        hidden=QWEN_HIDDEN, dtype_bytes=2,
                        per_turn_mb=PER_TURN_MB, safety=SAFETY_FACTOR,
                        **_ignored):
    """Price one session in MB. Returns {total_mb, breakdown}.

    - kv_cache: preallocated per-engine static cache share (amortized 1x:
      the cache is per-engine, but one busy session pins a full decode
      worth of working set, so we price it 1:1 for safety).
    - history: max_turns x tokens/turn x hidden x bytes (re-prefill set).
    - transient: per-turn TTS scratch + mel buffers.
    Priced 1:1 then scaled by ``safety``.
    """
    kv = kv_cache_mb(dtype_bytes=dtype_bytes)
    tokens_per_turn = float(prompt_tokens) + float(max_new_tokens)
    history = (float(max_turns) * tokens_per_turn * hidden
               * dtype_bytes / 1024 ** 2)
    transient = float(per_turn_mb)
    raw = kv + history + transient
    total = raw * float(safety)
    return {"total_mb": total,
            "breakdown": {"kv_cache_mb": kv, "history_mb": history,
                          "transient_mb": transient, "safety": float(safety)}}


def plan_capacity(vram_total_mb, baseline_mb, max_new_tokens=48,
                  headroom_frac=HEADROOM_FRAC,
                  headroom_min_mb=HEADROOM_MIN_MB,
                  sessions_cap=SESSIONS_CAP,
                  max_inflight=MAX_INFLIGHT_DEFAULT,
                  queue_timeout_s=QUEUE_TIMEOUT_S, **est_kwargs):
    """Divide usable VRAM by the per-session price -> serving plan dict.

    usable = total − baseline − headroom; headroom = max(frac x total,
    min_mb). max_sessions is clamped to [1, sessions_cap]; max_inflight
    stays at the measured serialization point (c=4 halves throughput).
    Never raises: unknown VRAM (total<=0) yields a conservative plan.
    """
    total = float(vram_total_mb or 0.0)
    baseline = float(baseline_mb or 0.0)
    est = estimate_session_mb(max_new_tokens=max_new_tokens, **est_kwargs)
    per_session = max(est["total_mb"], 1e-6)
    if total <= 0:
        return {"vram_total_mb": 0.0, "baseline_mb": baseline,
                "headroom_mb": float(headroom_min_mb),
                "usable_mb": 0.0, "per_session_mb": per_session,
                "breakdown": est["breakdown"],
                "max_sessions": 1, "max_inflight": int(max_inflight),
                "generation_length": int(max_new_tokens),
                "queue_timeout_s": float(queue_timeout_s),
                "conservative": True}
    headroom = max(total * float(headroom_frac), float(headroom_min_mb))
    usable = max(total - baseline - headroom, 0.0)
    n = int(usable // per_session)
    n = max(1, min(n, int(sessions_cap)))
    return {"vram_total_mb": total, "baseline_mb": baseline,
            "headroom_mb": headroom, "usable_mb": usable,
            "per_session_mb": per_session, "breakdown": est["breakdown"],
            "max_sessions": n, "max_inflight": int(max_inflight),
            "generation_length": int(max_new_tokens),
            "queue_timeout_s": float(queue_timeout_s),
            "conservative": False}
