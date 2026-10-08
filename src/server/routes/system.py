"""Liveness + model-leg routes."""

import asyncio
import time

from fastapi import APIRouter

from .. import state
from ..state import (
    MODEL_PROFILES,
    _current_model,
    _t_boot,
    _turns,
    get_agent,
    switch_model,
)

router = APIRouter()


@router.get("/health")
def health():
    try:
        loaded = bool(state._agent is not None and getattr(state._agent, "warmed", False))
    except Exception:
        loaded = False
    try:
        missing = list(getattr(state._agent, "missing", []) or [])
    except Exception:
        missing = []
    return {"ok": True, "uptime_s": time.time() - _t_boot,
            "agent_loaded": loaded, "missing": missing}


@router.get("/model")
def model_info():
    cur = _current_model.get("name")
    avail = []
    for key, p in MODEL_PROFILES.items():
        avail.append({"name": key, "label": p["label"], "backend": p["backend"],
                      "desc": p["desc"], "ready": bool(p.get("ready")),
                      "current": key == cur})
    return {"current": cur, "available": avail}


@router.post("/model/switch")
async def model_switch(payload: dict):
    return await switch_model((payload or {}).get("name", ""))


def _vram() -> dict:
    out = {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0,
           "total_mb": 0, "name": "", "cuda": False}
    try:
        import torch as _t
    except Exception:
        return out
    try:
        if not _t.cuda.is_available():
            return out
        dev = _t.cuda.current_device()
        free, total = _t.cuda.mem_get_info(dev)
        out.update({
            "allocated_mb": round(_t.cuda.memory_allocated(dev) / 1048576),
            "reserved_mb": round(_t.cuda.memory_reserved(dev) / 1048576),
            "peak_mb": round(_t.cuda.max_memory_allocated(dev) / 1048576),
            "total_mb": round(total / 1048576),
            "name": str(_t.cuda.get_device_name(dev)),
            "cuda": True,
        })
    except Exception:
        pass
    return out


def _ram_cpu() -> dict:
    total_mb, avail_mb, percent, load, count = 0, 0, 0, [], 0
    try:
        import os as _os
        count = int(_os.cpu_count() or 0)
        try:
            load = [round(float(v), 2) for v in _os.getloadavg()]
        except Exception:
            load = []
        meminfo: dict = {}
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].endswith(":"):
                        try:
                            meminfo[parts[0][:-1]] = int(parts[1])
                        except ValueError:
                            pass
        except Exception:
            pass
        if meminfo:
            total_mb = round(meminfo.get("MemTotal", 0) / 1024)
            avail_mb = round(meminfo.get("MemAvailable", meminfo.get("MemFree", 0)) / 1024)
            used_mb = max(0, total_mb - avail_mb)
        else:
            used_mb = 0
        def _cpu_times():
            with open("/proc/stat", encoding="utf-8") as f:
                p = f.readline().split()
            vals = [int(v) for v in p[1:8]]
            return vals
        try:
            a = _cpu_times()
            time.sleep(0.1)
            b = _cpu_times()
            idle = (b[3] + b[4]) - (a[3] + a[4])
            total = sum(b) - sum(a)
            if total > 0:
                percent = round((1 - idle / total) * 100)
        except Exception:
            percent = 0
    except Exception:
        used_mb = 0
    return {"total_mb": total_mb, "used_mb": used_mb,
            "available_mb": avail_mb, "percent": percent,
            "load": load, "count": count}


def _gpu_util() -> int:
    try:
        import subprocess as _sp
        out = _sp.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                       "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=3)
        return max(0, int(str(out.stdout or "0").split()[0]))
    except Exception:
        return 0


def _leg_labels(agent) -> dict:
    def _short(model: str, fallback: str) -> str:
        try:
            s = str(model or "").strip()
            return s.split("/")[-1] if s else fallback
        except Exception:
            return fallback
    try:
        stt_model = getattr(getattr(agent, "stt", None), "config", None)
        stt_name = _short(getattr(stt_model, "model", ""), "whisper")
    except Exception:
        stt_name = "whisper"
    try:
        tts_model = getattr(getattr(agent, "tts", None), "config", None)
        tts_name = _short(getattr(tts_model, "model", ""), "kokoro")
    except Exception:
        tts_name = "kokoro"
    try:
        cur = MODEL_PROFILES.get(_current_model.get("name"), {})
        llm_name = str(cur.get("label") or cur.get("name") or "llm")
    except Exception:
        llm_name = "llm"
    return {"vad": "silero", "stt": stt_name, "llm": llm_name, "tts": tts_name}


@router.get("/metrics")
async def metrics():
    agent = get_agent()
    vram, ram = await asyncio.gather(
        asyncio.to_thread(_vram),
        asyncio.to_thread(_ram_cpu),
    )
    gpu = await asyncio.to_thread(_gpu_util)
    return {
        "uptime_s": time.time() - _t_boot,
        "turns": {"done": _turns["done"],
                  "total_s": round(_turns["total_s"], 1),
                  "last_s": round(_turns["last_s"], 2)},
        "queue": {"pending": _turns["active"], "locks": 0},
        "vram": vram,
        "ram": {"total_mb": ram["total_mb"], "used_mb": ram["used_mb"],
                "available_mb": ram["available_mb"]},
        "cpu": {"percent": ram["percent"], "load": ram["load"], "count": ram["count"]},
        "gpu_util": gpu,
    }


@router.get("/metrics/context")
async def metrics_context(sid: str = ""):
    sid = str(sid or "").strip()
    tokens, ctx = 0, 4096
    try:
        agent = get_agent()
        sessions = getattr(agent, "sessions", None)
        if sessions is not None and sid:
            hist = sessions.history(sid) or []
            chars = sum(len(str(m.get("content", "") or "")) for m in hist
                        if isinstance(m, dict))
            tokens = max(0, chars // 4)
        llm_cfg = getattr(getattr(agent, "llm", None), "config", None)
        ctx = int(getattr(llm_cfg, "max_len", 0) or 4096)
    except Exception:
        pass
    pct = min(1.0, tokens / max(1, ctx))
    return {"tokens": tokens, "ctx": ctx, "pct": pct}


@router.get("/legs")
async def legs():
    agent = get_agent()
    try:
        from src.tools.terminal import ALLOWED_OPS
        ops = sorted(set(ALLOWED_OPS))
    except Exception:
        ops = []
    try:
        cur = MODEL_PROFILES.get(_current_model.get("name"), {})
        model = {"name": str(cur.get("name", "")),
                 "label": str(cur.get("label", "")),
                 "backend": str(cur.get("backend", ""))}
    except Exception:
        model = {"name": "", "label": "", "backend": ""}
    return {"legs": _leg_labels(agent), "model": model,
            "policy": {"ops": ops, "deny": [], "confirm": ops}}
