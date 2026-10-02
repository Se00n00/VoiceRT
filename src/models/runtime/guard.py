"""Pre-execution GPU/memory guard: refuse BEFORE the box can wedge.

Background: on 2026-10-02 a ``kill -9`` landed mid-CUDA-compile and wedged
the RTX 3050 into ``ERR!`` state (torch CUDA ``False``, cold boot required).
A guard that fires post-mortem is a log line, not a guard — so every check
here runs BEFORE weights load, kernels compile, or sidecars spawn.

Modes via ``VOICE_GUARD``: ``0``/``false``/``no`` disables all checks;
``warn`` logs problems without raising; anything else (default) enforces.
When no CUDA device is visible the GPU checks downgrade to warn — a
missing/wedged driver must degrade to CPU fallback, never to a hang.

Full design: ``docs/plans/gpu-memory-guard.md``.
"""
import fcntl
import os
import subprocess

LOCK_ENV = "VOICE_GPU_LOCK"
LOCK_PATH_DEFAULT = "/tmp/voice-gpu.lock"
MODE_ENV = "VOICE_GUARD"
SMI_TIMEOUT_S = 5
COMPILE_RESERVE_MB = 400.0

# Conservative per-backend warm estimates (weights + KV + working set).
# The fused legs re-check precisely at build (check_budget); these only
# need to be sane enough to stop a doomed warm before it starts.
LEG_VRAM_MB = {
    "qwen": 2000.0,
    "minicpm": 2600.0,
    "minicpm_q4k": 1200.0,
}
LEG_RAM_MB = {
    "qwen": 1500.0,
    "minicpm": 2400.0,
    "minicpm_q4k": 1000.0,
}
STT_VRAM_MB = 800.0
STT_RAM_MB = 800.0
TTS_VRAM_MB = 800.0
TTS_RAM_MB = 800.0
VAD_RAM_MB = 100.0


class GuardRefusal(MemoryError):
    """Raised when preflight refuses work. Carries a machine-readable reason."""

    def __init__(self, what: str, reason: str, detail: str = ""):
        self.what = what
        self.reason = reason
        self.detail = detail
        msg = f"GUARD REFUSED {what}: {reason}."
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


def guard_mode() -> str:
    """Enforcement mode from the environment."""
    raw = os.environ.get(MODE_ENV, "").strip().lower()
    if raw in ("0", "false", "no", "off", "disable"):
        return "off"
    if raw in ("warn", "warn-only", "log"):
        return "warn"
    return "enforce"


def lock_path() -> str:
    """Tenant lockfile path (overridable for tests / multi-user boxes)."""
    return os.environ.get(LOCK_ENV, "").strip() or LOCK_PATH_DEFAULT


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _smi_text() -> tuple[bool, str]:
    """Run nvidia-smi; return (ran_ok, output). Never raises."""
    try:
        r = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True,
            timeout=SMI_TIMEOUT_S)
        return True, r.stdout + r.stderr
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    except Exception as exc:  # noqa: BLE001 - timeout or spawn failure
        return False, f"nvidia-smi failed: {exc}"


def driver_health() -> tuple[bool, str]:
    """True when a CUDA device is usable right now.

    A wedged driver (``ERR!`` rows, Xid storms) must fail HERE with a cold
    boot message — never as a mysterious hang inside a generate call.
    """
    if not _cuda_available():
        return False, "torch CUDA reports no usable device"
    ran, out = _smi_text()
    if ran and "ERR!" in out:
        return False, ("nvidia-smi reports ERR! — driver wedged, cold boot "
                        "required (shutdown, unplug 60s; a reboot is not "
                        "enough)")
    return True, "cuda ok"


_LOCK_FD = None


def _holder_cmd(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(
                "utf-8", "replace").strip()[:120]
    except Exception:
        return ""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def tenant_holder() -> tuple[int | None, str]:
    """Pid + cmd currently holding the GPU tenant lock, if any."""
    try:
        with open(lock_path(), encoding="utf-8") as f:
            pid = int(f.read().strip().split()[0])
    except Exception:
        return None, ""
    if pid > 0 and _pid_alive(pid):
        return pid, _holder_cmd(pid)
    return None, ""


def acquire_tenant_lock() -> tuple[bool, int | None, str]:
    """Claim the single-GPU-tenant lock for this process (held for life).

    Returns (acquired, holder_pid, holder_cmd): ``acquired`` True means
    this process now holds the tenancy (same-process re-entry is a no-op
    success). Otherwise the pid/cmd name the live holder — two warmed
    processes on a 4GB card die together, so the second one never starts.
    A lockfile naming a dead pid is stale: flock succeeds and we reclaim.
    """
    global _LOCK_FD
    if _LOCK_FD is not None:
        return True, None, ""
    path = lock_path()
    try:
        fd = open(path, "a+")
    except Exception as exc:  # noqa: BLE001 - unwritable lock dir
        return False, -1, f"could not open lockfile {path}: {exc}"
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        fd.close()
        pid, cmd = tenant_holder()
        if pid is None:
            # Locked but holder unreadable (garbage pid, or holder between
            # write and lock): still locked, still not ours. Refuse.
            return False, -1, f"lockfile {path} is locked by an unknown holder"
        return False, pid, cmd
    fd.seek(0)
    fd.truncate()
    fd.write(f"{os.getpid()}\n")
    fd.flush()
    _LOCK_FD = fd  # held until process exit: this IS the tenancy
    return True, None, ""


def release_tenant_lock() -> None:
    """Release the tenant lock. Test isolation only — production holders
    keep it for process life (releasing invites the OOM the lock stops)."""
    global _LOCK_FD
    fd, _LOCK_FD = _LOCK_FD, None
    if fd is not None:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            fd.close()
        except Exception:
            pass


def sidecar_gpu_estimate(backend: str, config=None) -> tuple[bool, float]:
    """(needs_gpu, vram_mb) for llama-server sidecar backends.

    CPU-only sidecars (gemma/qwen06/qwen17) need no GPU: (False, 0).
    Bonsai parks ``-ngl`` transformer blocks in VRAM, so a nonzero layer
    count prices VRAM (weights share + context reserve); ``ngl=0`` is
    pure CPU and needs none. Estimates are conservative by design — the
    sidecar's own checks stay authoritative, this only stops a blind
    claim on a card that has no room.
    """
    if str(backend) != "bonsai":
        return False, 0.0
    try:
        from src.models.bonsai_llamacpp import (
            CUDA_CTX_RESERVE_MB,
            GGUF_FILES,
            N_LANG_LAYERS,
            resolve_ngl,
        )

        ngl = resolve_ngl(getattr(config, "bonsai_ngl", "auto"))
        if ngl <= 0:
            return False, 0.0
        total_mb = float(GGUF_FILES["ptq1_0"][1]) / 1024.0 / 1024.0
        per_layer = total_mb / max(1, int(N_LANG_LAYERS))
        return True, ngl * per_layer + float(CUDA_CTX_RESERVE_MB)
    except Exception:
        # Unknown layout: assume it may touch the card (tenant lock +
        # driver check) but price no VRAM beyond that.
        return True, 0.0


def _ram_avail_gb() -> float:
    try:
        from src.models.bonsai_llamacpp import mem_available_gb

        return float(mem_available_gb())
    except Exception:
        return 0.0


def _vram_total_mb() -> float:
    try:
        from src.models.runtime.capacity import probe_vram

        return float(probe_vram().get("total_mb", 0.0) or 0.0)
    except Exception:
        return 0.0


def preflight(what: str, vram_mb: float = 0.0, ram_mb: float = 0.0,
              needs_gpu: bool = True) -> dict:
    """Refuse doomed work before it starts. Returns a status dict.

    Enforcing (default) raises :class:`GuardRefusal` on the first problem.
    ``warn`` mode logs every problem and returns them collected.
    ``off`` mode (``VOICE_GUARD=0``) skips all checks.
    """
    from src.models.runtime.memory import fits

    mode = guard_mode()
    if mode == "off":
        return {"guard": "off", "what": what}
    problems: list[tuple[str, str]] = []

    def fail(reason: str, detail: str = "") -> None:
        if mode == "warn":
            problems.append((reason, detail))
            print(f"guard warn [{what}]: {reason}. {detail}".rstrip(),
                  flush=True)
        else:
            raise GuardRefusal(what, reason, detail)

    cuda = _cuda_available()
    if needs_gpu:
        if not cuda:
            # No CUDA in sight: refuse GPU-bound work, but as warn even in
            # enforce mode — CPU-fallback legs must keep working (and CPU
            # test boxes must keep passing) where there is no GPU to wedge.
            problems.append(("no CUDA device visible",
                             "CPU fallback may proceed; GPU work refused"))
            print(f"guard warn [{what}]: no CUDA device visible. "
                  f"CPU fallback may proceed; GPU work refused.", flush=True)
            if ram_mb > 0:
                avail = _ram_avail_gb()
                if 0.0 < avail < ram_mb / 1024.0:
                    fail(f"only {avail:.1f}GB RAM available",
                         f"need ~{ram_mb / 1024.0:.1f}GB")
            return {"guard": mode, "what": what, "cuda": False,
                    "problems": problems}
        ok, detail = driver_health()
        if not ok:
            fail("GPU driver wedged", detail)
        acquired, pid, cmd = acquire_tenant_lock()
        if not acquired:
            who = f"pid {pid} ({cmd})" if pid and pid > 0 else cmd
            fail(f"GPU held by {who}",
                 "one warmed process per card — refusing the second")
        if vram_mb > 0:
            total = _vram_total_mb()
            if total > 0 and not fits(vram_mb, total - COMPILE_RESERVE_MB):
                from src.models.runtime.memory import summary

                u = summary()
                fail(f"need ~{vram_mb:.0f}MB VRAM",
                     f"card {total:.0f}MB, allocated "
                     f"{u.get('allocated_mb', 0.0):.0f}MB")
    if ram_mb > 0:
        avail = _ram_avail_gb()
        if 0.0 < avail < ram_mb / 1024.0:
            fail(f"only {avail:.1f}GB RAM available",
                 f"need ~{ram_mb / 1024.0:.1f}GB")
    if mode == "warn" and problems:
        return {"guard": mode, "what": what, "problems": problems}
    return {"guard": mode, "what": what, "cuda": cuda,
            "problems": problems}
