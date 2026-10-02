# PLAN: pre-execution GPU memory guard

Status: steps 1–2 implemented 2026-10-02 (`src/models/runtime/guard.py` +
`tests/runtime/test_guard.py`, warm paths wired, enforcing by default).
Steps 3–4 (hot-path gates, SIGTERM handlers, `/health` field, TUI badge)
remain open. Trigger: RTX 3050 wedged into `ERR!` state (torch CUDA
`False`, cold boot required) after a `kill -9` landed mid-CUDA-compile,
plus the standing rule that a second GPU process means OOM on this 4GB
box.

Goal: refuse or serialize work **before** the GPU can wedge — never clean
up after. A guard that fires post-mortem is a log line, not a guard.

## 1. What already exists (do not reinvent)

| Guard | File | Covers |
|---|---|---|
| `fits` / `check_budget` → `MemoryBudgetExceeded` (dual-inherits `RuntimeError`, so server error-mapping yields HTTP 503) | `src/models/runtime/memory.py` | single VRAM allocation vs budget |
| `probe_vram` / `estimate_session_mb` / `plan_capacity` | `src/models/runtime/capacity.py` | session count from baseline+headroom |
| `_check_ram` (precise tier: real GGUF bytes + KV ctx + offload; refuses swap-death) | `src/models/bonsai_llamacpp.py:422` | worker sidecar CPU-RAM fit |
| `assert_cuda_leg` / `assert_ready_leg` (fail fast, no silent CPU fallback) | `src/models/llm.py:77` | sick driver aborts runners early |
| `KVCacheBatched` + `check_budget(est, 4000, headroom 400)` | fused legs (`src/models/qwen.py`, `whisper.py`) | static KV fit at build |

## 2. Gaps (why the box still died)

1. **No cross-process guard.** Two processes each fit their own budget and
   die together. The ONE-GPU-process rule lives in prose (`AGENTS.md`),
   not in code. The stale 18h `bridge.py` + a fresh warm is exactly this.
2. **No pre-execution check on the hot path.** `generate`/`transcribe`/
   `speak` assume the warm-time budget still holds. KV growth, batch > 1,
   and first-call Triton compile spikes are unbudgeted — compile storms
   are the most likely wedge vector (CPU idle, GPU idle, driver gone).
3. **No driver-health gate.** Nothing checks `torch.cuda.is_available()`
   or parses `nvidia-smi` `ERR!`/`Xid` before launching work; a wedged
   driver fails as a mysterious hang (cf. `/deep` debugging 2026-10-02,
   which chased a hang that was partly this).
4. **No graceful-shutdown path.** SIGTERM → uvicorn shutdown is slow;
   operators reach for `kill -9`, which mid-compile wedges the driver.
   No handler releases CUDA (`empty_cache`, context teardown) on the way
   out, and no doc says "never -9 a CUDA-busy process".
5. **Refusals are invisible.** `_check_ram` prints to a log nobody watches.
   The TUI badge (`●/○ bridge`) cannot distinguish "warming" from
   "refused: need 7.7GB, have 3.9GB".

## 3. Design: `src/models/runtime/guard.py`

One entry point, called before every expensive action:

```python
preflight(what: str, vram_mb: float = 0, ram_mb: float = 0) -> dict
# raises GuardRefusal(MemoryError) with machine-readable reason; else
# returns {"vram_free_mb": …, "ram_avail_mb": …, "driver": "ok"}
```

Checks, in order (cheapest first, each skippable via env for tests):

1. **Driver health** — `torch.cuda.is_available()` (when CUDA expected)
   AND `nvidia-smi` parses clean (no `ERR!`, no `Xid` in `dmesg`-lite
   tail). Failure → `GuardRefusal("driver wedged — cold boot required
   (shutdown, unplug 60s); a reboot is not enough")`. Never hang: wrap
   the smi call in a 5s timeout.
2. **Single GPU tenant** — pid lockfile (`/tmp/voice-gpu.lock`,
   `fcntl.flock` + staleness check via `/proc`). Second warmer gets
   `GuardRefusal("GPU held by pid <p> (<cmd>); one process per card")`.
3. **VRAM fit** — `memory.fits(need, total - headroom)` with `need`
   covering weights + KV + a compile-spike reserve (first call only;
   flag cleared after first successful generate per process).
4. **RAM fit** — generalize bonsai's `_check_ram` tiers: precise when
   weight bytes are known, flat `min_ram_gb` rail otherwise. Reuse, don't
   fork: move the helper into `runtime/` and call it from both.

Call sites (all cheap, cached per process where hot):

- `LlmModel/SttModel/TtsModel.warm()` — full check (this is where the
  18h hang and the bonsai refusal both happened).
- `generate` / `transcribe` / `speak` — driver-health + tenant check
  only (no per-token math); refuse in milliseconds, not after a hang.
- Sidecar spawn (`gemma_llamacpp` server start) — RAM fit + port clash.
- Bench/eval entry points — same `preflight`, so sweeps fail fast
  instead of wedging the shared box mid-run.

Shutdown path (kills the `kill -9` habit):

- `SIGTERM` handler in `server.py` / `bridge.py` entry: stop accepting,
  cancel turns, `torch.cuda.empty_cache()` + `synchronize()` with a
  timeout, then exit. Document next to it: **never `kill -9` a process
  with live CUDA work** — that is what wedged 2026-10-02.

Visibility:

- `GET /health` gains `guard: {"last_refusal": str | null,
  "refusals": int}`; TUI badge shows `⊘ refused: <short reason>` when
  set, distinct from `○ bridge down`. Today's failure ("need 7.7GB,
  have 3.9GB") would have been on the status bar, not in a log tail.

## 4. Acceptance

- Unit (`tests/`): refusal messages for simulated pressure (fake
  budgets, fake `ERR!` smi output, stale/fresh lockfile); `fits` math
  unchanged; guard disabled cleanly via `VOICE_GUARD=0`.
- Integration: second concurrent warm attempt refused naming the holder
  pid; `kill -TERM` during a turn exits ≤10s with CUDA released
  (`nvidia-smi` shows no residue); `/health` carries the refusal.
- Chaos (documents a driver limit, not a fix): `kill -9` mid-compile may
  still wedge — the guard's job is shrinking that window (fast refusal,
  graceful TERM path, tenant lock), stated openly in the docstring.

## 5. Non-goals

- Fixing the NVIDIA driver, using swap, multi-GPU scheduling, or touching
  `src/inference/engine.py` (do-not-touch).
- Changing budgets themselves (`capacity.py` numbers stay; the guard
  enforces them earlier and across processes).

## 6. Rollout

1. `runtime/guard.py` + unit tests (no behavior change: guard defaults
   to warn-only `VOICE_GUARD=warn` for one session).
2. Wire warm paths → flip default to enforcing.
3. Wire hot paths (health-gate only) + SIGTERM handlers + `/health`
   field + TUI badge state.
4. Record: refusal counts per reason in `benchmarks/results/` run logs;
   first real refusal is a test pass, not an incident.
