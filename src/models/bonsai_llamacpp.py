"""Ternary Bonsai 2 27B leg via a local Prism-fork llama.cpp sidecar.

Weights live off-process (``llama-server``, OpenAI-compatible API), so the
GPU/CPU layer split is a server flag, not torch code: ``-ngl N`` parks N
transformer blocks in VRAM, the rest stays mmapped on CPU/RAM. ``"auto"``
picks N from free VRAM at warm (see :func:`resolve_ngl`) — one equation
covers both the 4GB laptop (partial offload) and a T4 (full offload).

Model facts (PrismML, Apache 2.0, verified 2026-09-27):
- ``prism-ml/Ternary-Bonsai-2-27B-gguf``: PTQ1_0 5.93GB (1.76bpw, small)
  or PQ2_0 7.25GB (2.16bpw, faster prefill, demo default).
- ternary {-1,0,+1} g128 + FP16 scales + blockwise Hadamard rotation.
- 64 language blocks, hybrid attention (~75% linear / ~25% full),
  262K native context — 16K needs no RoPE hacks.
- vision tower ships separately (``mmproj`` HQQ 4-bit, 0.63GB).
- native tool calling (``tools[]`` -> ``tool_calls``) + thinking mode
  (``thinking_budget_tokens`` per request, ``0`` disables).
- 4-bit KV cache ~lossless (whitepaper) — hence ``--cache-type-k q4_0``.

HARD REQUIREMENT: the PrismML llama.cpp fork (``prism-b10658``+) — the
rotated weight basis needs a runtime Walsh-Hadamard transform that is not
upstream. Stock builds refuse PTQ1_0/PQ2_0, and a Q2_0 file loads silent
+ outputs gibberish. :func:`resolve_server_bin` enforces the fork tag.

Tool-call grammar: composed raw text (contract for
``parse_gemma_action`` in :mod:`src.tools.terminal`) — optional chat
``content``, then zero or more ``<|tool_call>call:NAME ARGS <tool_call|>``
blocks. One parser serves both sidecars.

Matches the sidecar selection contract so :class:`LlmModel` can select
it via ``LlmConfig(backend="bonsai")``:

- ``is_sidecar = True``: :class:`LlmModel` passes ``(messages, tools)``
  straight through instead of encoding to ids first.
- ``chat`` / ``chat_stream`` speak text; streamed pieces carry
  ``token_id=-1`` upstream (callers already fall back to joined pieces).
"""

import atexit
import json
import os
import subprocess
import time
import urllib.request

__all__ = [
    "BonsaiLlamaCpp",
    "compose_raw",
    "to_openai_tools",
    "resolve_gguf",
    "resolve_mmproj",
    "resolve_server_bin",
    "resolve_ngl",
    "ngl_attempts",
    "kv_mb_est",
    "ram_required_gb",
    "cached_gguf_bytes",
    "mem_available_gb",
    "vram_free_mb",
]

REPO_ID = "prism-ml/Ternary-Bonsai-2-27B-gguf"
GGUF_FILES = {
    "ptq1_0": ("Ternary-Bonsai-2-27B-PTQ1_0.gguf", 5930000000),
    "pq2_0": ("Ternary-Bonsai-2-27B-PQ2_0.gguf", 7250000000),
}
MMPROJ_FILE = "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
MMPROJ_BYTES = 700000000  # Q8_0 tower; tolerance-checked, not exact

N_LANG_LAYERS = 64
N_CTX_DEFAULT = 16384
KV_TYPE_DEFAULT = "q4_0"
# Full-attention layers (~25% of 64) dominate KV; linear layers are O(1)
# state. Placeholder until M4 reads config.json post-download — the 16K
# bench corrects it by measurement, never by theory.
N_FULL_ATTN_LAYERS = 16
KV_MB_16K_Q4 = 800.0
CUDA_CTX_RESERVE_MB = 450.0
# Fixed-ish VRAM beyond weights at 16K ctx (compute pp buffers + full KV).
# Measured 2026-09-28, -c 16384 q4_0: ngl=9+mmproj → 3092MB total
# (~836 weights + 629 mmproj + ~1630 overhead); ngl=6 no-mmproj →
# 1907-2287MB (~558 weights + ~1350-1730 overhead). Prefill holds
# ~100 tok/s linear to 13.5K true tokens; decode ~1.1 tok/s at ngl=6
# (CPU-bound: 58/64 layers on CPU + disabled fused FA/GDN).
COMPUTE_OVERHEAD_MB = 1600.0
# Non-weight CPU-side headroom for the compute buffers that sit outside the
# KV estimate. ~1GB measured slack over the weights+KV sum.
RAM_OVERHEAD_GB = 1.0

TOOL_OPEN = "<|tool_call>call:"
TOOL_CLOSE = "<tool_call|>"


def mem_available_gb() -> float:
    """Free RAM in GB from /proc/meminfo (0.0 when unreadable)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return 0.0


def vram_free_mb() -> float:
    """Free VRAM in MB: torch first, nvidia-smi fallback, 0.0 if none."""
    try:
        import torch

        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return float(free) / 1024.0 / 1024.0
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        return float(out.strip().split()[0])
    except Exception:
        return 0.0


def kv_mb_est(n_ctx, kv_type=KV_TYPE_DEFAULT,
              base_mb=KV_MB_16K_Q4, base_ctx=N_CTX_DEFAULT) -> float:
    """Scale the 16K/q4_0 KV placeholder linearly with context.

    q8_0 costs ~2x q4_0; f16 ~4x. Corrected by measurement in M4.
    """
    mult = {"q4_0": 1.0, "q8_0": 2.0, "f16": 4.0}.get(str(kv_type), 1.0)
    return float(base_mb) * float(n_ctx) / float(base_ctx) * mult


def resolve_ngl(explicit="auto", vram_free="auto", gguf_bytes=None,
                n_ctx=N_CTX_DEFAULT, kv_type=KV_TYPE_DEFAULT,
                reserve_mb=CUDA_CTX_RESERVE_MB,
                overhead_mb=COMPUTE_OVERHEAD_MB,
                n_layers=N_LANG_LAYERS) -> int:
    """GPU layer count: explicit int passes through, ``"auto"`` computes.

    ``ngl = clamp(floor((free - reserve - kv - overhead) / per_layer))``.
    The overhead term is load-bearing: weights-only math OOMs (measured
    2026-09-27: ngl=27 + 16K ctx blew 3.8GB on ~1GB compute buffers the
    per-layer estimate ignores). Zero VRAM (or shortfall) means
    CPU-only — slow but functional, never negative, never over the
    layer count.
    """
    if explicit is not None and str(explicit) != "auto":
        return max(0, min(int(explicit), int(n_layers)))
    if vram_free is None or str(vram_free) == "auto":
        free = float(vram_free_mb())
    else:
        free = float(vram_free)
    if free <= 0:
        return 0
    total = float(gguf_bytes if gguf_bytes else GGUF_FILES["ptq1_0"][1])
    per_layer = total / 1024.0 / 1024.0 / float(n_layers)
    avail = (free - float(reserve_mb) - kv_mb_est(n_ctx, kv_type)
             - float(overhead_mb))
    if avail <= 0:
        return 0
    return max(0, min(int(avail // per_layer), int(n_layers)))


def ngl_attempts(est, n_layers=N_LANG_LAYERS) -> list:
    """Spawn attempts, halving on OOM-abort: [est, est//2, ..., 0].

    The estimate is conservative but VRAM reality (other legs, driver
    slack) varies — halving converges to a bootable split instead of
    aborting the whole warm. Deduped, order-preserving, floors at 0
    (CPU-only always boots).
    """
    out = []
    n = int(est)
    while True:
        n = max(0, min(n, int(n_layers)))
        if n not in out:
            out.append(n)
        if n <= 0:
            return out
        n //= 2


def _snapshot(*files):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=REPO_ID, allow_patterns=list(files))


def cached_gguf_bytes(explicit=None, packing="ptq1_0"):
    """Byte size of the GGUF if it is already on disk, else ``None``.

    Never downloads. :func:`warm` uses this to check RAM precisely when the
    weights are already here, and to fall back to a flat pre-download
    rail when they are not — so a 6GB fetch is never spent on a box that
    cannot run the result.
    """
    try:
        if explicit and str(explicit) != "auto":
            p = os.path.abspath(os.path.expanduser(str(explicit)))
            return os.path.getsize(p) if os.path.isfile(p) else None
        from huggingface_hub import try_to_load_from_cache

        fname, _want = GGUF_FILES[str(packing)]
        p = try_to_load_from_cache(REPO_ID, fname)
        if not p or not os.path.isfile(p):
            return None
        return os.path.getsize(p)
    except Exception:
        return None


def ram_required_gb(gguf_bytes, n_ctx=N_CTX_DEFAULT, kv_type=KV_TYPE_DEFAULT,
                    n_gpu_layers="auto", n_layers=N_LANG_LAYERS,
                    overhead_gb=RAM_OVERHEAD_GB) -> float:
    """CPU-resident RAM this sidecar needs, in GB.

    Derived rather than flat, because a constant is wrong in both
    directions for these weights: the PTQ1_0 GGUF is ~6GB, not the ~15GB a
    dense 27B Q4 would be, so a flat 8GB rail refuses setups that fit with
    room to spare. What lands on the CPU depends on how many layers fit in
    VRAM, so this tracks :func:`resolve_ngl` — more layers on the card,
    less RAM here.
    """
    ngl = resolve_ngl(n_gpu_layers, gguf_bytes=gguf_bytes, n_ctx=n_ctx,
                      kv_type=kv_type, n_layers=n_layers)
    cpu_weights = float(gguf_bytes) * max(0.0, 1.0 - ngl / max(1, n_layers))
    kv = kv_mb_est(n_ctx, kv_type) * 1e6
    return (cpu_weights + kv + float(overhead_gb) * 1e9) / 1e9


def resolve_gguf(explicit=None, packing="ptq1_0"):
    """Find the GGUF weights: explicit path, else HF cache download."""
    if explicit and str(explicit) != "auto":
        p = os.path.abspath(os.path.expanduser(str(explicit)))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"bonsai gguf not found: {p}")
        return p
    fname, want = GGUF_FILES[str(packing)]
    root = _snapshot(fname)
    p = os.path.join(root, fname)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"bonsai gguf missing after download: {p}")
    size = os.path.getsize(p)
    if abs(size - want) / want > 0.05:
        raise ValueError(
            f"ABORT: {p} is {size} bytes, expected ~{want} "
            f"(incomplete download?). Re-fetch and retry.")
    return p


def resolve_mmproj(explicit=None):
    """Find the vision tower: explicit path, ``"skip"`` disables, else HF."""
    if explicit is not None and str(explicit) == "skip":
        return None
    if explicit and str(explicit) != "auto":
        p = os.path.abspath(os.path.expanduser(str(explicit)))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"bonsai mmproj not found: {p}")
        return p
    root = _snapshot(MMPROJ_FILE)
    p = os.path.join(root, MMPROJ_FILE)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"bonsai mmproj missing after download: {p}")
    return p


def _is_prism_checkout(bin_path, _run=None):
    """True when the binary was built from a PrismML-Eng checkout.

    The fork's ``--version`` carries no prism tag, so fall back to the
    source tree next to the binary (``<root>/build/bin/llama-server``
    -> ``<root>/.git``): branch ``prism`` or a PrismML-Eng remote
    proves provenance. Anything unverifiable returns False (fail
    closed — a stock build outputs silent gibberish on these weights).
    """
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(bin_path))))
        if not os.path.isdir(os.path.join(root, ".git")):
            return False
        run = _run or (lambda c: subprocess.run(
            c, capture_output=True, text=True, timeout=15))
        branch = run(["git", "-C", root, "branch",
                      "--show-current"]).stdout.strip()
        if branch == "prism":
            return True
        remotes = run(["git", "-C", root, "remote", "-v"]).stdout.lower()
        return "prismml" in remotes
    except Exception:
        return False


def resolve_server_bin(explicit=None, require_prism_fork=True,
                       _run=None):
    """Find llama-server + enforce the Prism fork (rotated-basis kernels).

    Stock upstream builds refuse PTQ1_0/PQ2_0 (or worse: silent gibberish
    on Q2_0), so a non-fork binary is a hard abort, not a warning.
    The fork's ``--version`` carries no prism tag, so provenance falls
    back to the source checkout next to the binary (see
    :func:`_is_prism_checkout`).
    """
    cands = []
    if explicit and str(explicit) != "auto":
        cands.append(os.path.abspath(os.path.expanduser(str(explicit))))
    cands.append(os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "third_party", "llama.cpp", "build", "bin", "llama-server")))
    import shutil

    which = shutil.which("llama-server")
    if which:
        cands.append(which)
    run = _run or (lambda c: subprocess.run(
        c, capture_output=True, text=True, timeout=15))
    for p in cands:
        if not (p and os.path.isfile(p) and os.access(p, os.X_OK)):
            continue
        if not require_prism_fork:
            return p
        try:
            ver = run([p, "--version"]).stdout + run([p, "--version"]).stderr
        except Exception:
            continue
        if "prism" in ver.lower():
            return p
        if _is_prism_checkout(p, _run=run):
            return p
        raise SystemExit(
            f"ABORT: {p} is not the PrismML llama.cpp fork "
            f"(need prism-b10658+ for the rotated ternary basis). "
            f"Build github.com/PrismML-Eng/llama.cpp and retry.")
    raise FileNotFoundError(
        "llama-server not found. Build the PrismML fork "
        "(github.com/PrismML-Eng/llama.cpp, target llama-server) or pass "
        "LlmConfig(bonsai_bin='/path/to/llama-server').")


def to_openai_tools(tools):
    """TERMINAL_TOOLS dicts -> OpenAI ``{type, function}`` specs."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or "name" not in t:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object"}),
            },
        })
    return out


def compose_raw(content, tool_calls, reasoning=""):
    """Compose parser-target raw text: think block + content + tool blocks.

    ``tool_calls`` are OpenAI-style ``{"name":..., "arguments": str|dict}``.
    The server returns reasoning in a separate ``reasoning_content`` field —
    wrapping it in ``<think>`` here is what lets :func:`split_thinking`
    downstream turn it into ``thinking`` events (otherwise Bonsai's
    thoughts are silently dropped and the TUI shows no think trace).
    Same raw-text grammar the sidecar parser consumes.
    """
    parts = []
    if reasoning and str(reasoning).strip():
        parts.append(f"<think>{reasoning}</think>")
    if content:
        parts.append(str(content))
    for tc in tool_calls or []:
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        name = str(fn.get("name", "") or "")
        args = fn.get("arguments", "")
        if isinstance(args, dict):
            args = json.dumps(args)
        args = str(args or "").strip() or "{}"
        if name:
            parts.append(f"{TOOL_OPEN}{name}{args}{TOOL_CLOSE}")
    return "".join(parts)


class BonsaiLlamaCpp:
    """27B-ternary sidecar leg. ``warm()`` attaches or spawns; ``close()`` frees."""

    is_sidecar = True

    def __init__(self, gguf_path="auto", packing="ptq1_0", mmproj="auto",
                 host="127.0.0.1", port=8081,
                 n_ctx=N_CTX_DEFAULT, n_gpu_layers="auto",
                 cache_type_k=KV_TYPE_DEFAULT, threads=0,
                 thinking_budget=None, server_bin="auto",
                 min_ram_gb=8.0, startup_timeout=600,
                 _post_fn=None, _stream_fn=None, _popen=None):
        self.gguf_path = gguf_path
        self.packing = str(packing)
        self.mmproj = mmproj
        self.host = host
        self.port = int(port)
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = n_gpu_layers
        self.cache_type_k = str(cache_type_k)
        self.threads = int(threads)
        self.thinking_budget = thinking_budget
        self.server_bin = server_bin
        self.min_ram_gb = float(min_ram_gb)
        self.startup_timeout = float(startup_timeout)
        self._post_fn = _post_fn
        self._stream_fn = _stream_fn
        self._popen = _popen or subprocess.Popen
        self._proc = None
        self.ngl_used = 0
        self.base_url = f"http://{host}:{int(port)}"

    # -- lifecycle ------------------------------------------------------
    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(
                    self.base_url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def _check_ram(self):
        """Refuse to boot when the CPU side genuinely does not fit.

        Two tiers, because the honest number depends on whether we know the
        weights yet. Cached: compute the real requirement from the actual
        GGUF size, the KV context and how many layers fit on the card, and
        compare that. Not cached: fall back to the flat ``min_ram_gb``
        pre-download rail rather than spend a 6GB fetch on a box that
        cannot run the result.

        Both tiers refuse; they differ only in how precisely they know. The
        point of the precise tier is that a flat constant is *wrong* for
        these ternary weights (~6GB, not a dense 27B's ~15GB) and was
        refusing configurations that fit with over a gigabyte to spare.
        """
        free = mem_available_gb()
        cached = cached_gguf_bytes(self.gguf_path, self.packing)
        if cached:
            need = ram_required_gb(cached, self.n_ctx, self.cache_type_k,
                                   self.n_gpu_layers)
            if 0.0 < free < need:
                raise MemoryError(
                    f"ABORT: {free:.1f}GB RAM available, need {need:.1f}GB "
                    f"for the CPU-resident layers + context at n_ctx="
                    f"{self.n_ctx}. Close apps, or lower the worker's "
                    f"n_ctx, and retry. Refusing swap-death on purpose.")
            return
        if 0.0 < free < self.min_ram_gb:
            raise MemoryError(
                f"ABORT: {free:.1f}GB RAM available, need >="
                f"{self.min_ram_gb:.0f}GB before downloading {self.packing} "
                f"weights. Close apps and retry. Refusing swap-death on "
                f"purpose.")

    def warm(self):
        """Attach to a healthy server or spawn one. Raises on failure."""
        if self._health():
            return self
        self._check_ram()
        gguf = resolve_gguf(self.gguf_path, self.packing)
        size = os.path.getsize(gguf)
        auto = self.n_gpu_layers is None or str(self.n_gpu_layers) == "auto"
        est = resolve_ngl("auto" if auto else self.n_gpu_layers,
                          gguf_bytes=size, n_ctx=self.n_ctx,
                          kv_type=self.cache_type_k)
        # Explicit stays single-attempt (operator's word is law); auto
        # walks the halving ladder so a tight box boots small instead of
        # aborting (CPU-only 0 always boots).
        tries = ngl_attempts(est) if auto else [est]
        self.ngl_used = est
        if self._health():
            return self
        bin_path = resolve_server_bin(self.server_bin)
        last_err = ""
        import tempfile

        for ngl in tries:
            cmd = [bin_path, "-m", gguf, "--port", str(self.port),
                   "-c", str(self.n_ctx),
                   "--n-gpu-layers", str(ngl),
                   "--cache-type-k", self.cache_type_k,
                   "--log-disable"]
            if self.mmproj is not None and str(self.mmproj) != "skip":
                cmd += ["--mmproj", resolve_mmproj(self.mmproj)]
            if self.threads > 0:
                cmd += ["-t", str(self.threads)]
            logf = tempfile.NamedTemporaryFile(prefix="bonsai-server-",
                                               suffix=".log", delete=False)
            logname = logf.name
            self._proc = self._popen(cmd, stdout=logf,
                                     stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     start_new_session=True)
            atexit.register(self.close)
            t0 = time.monotonic()
            dead_tail = ""
            while time.monotonic() - t0 < self.startup_timeout:
                if self._proc.poll() is not None:
                    try:
                        with open(logname, "rb") as f:
                            f.seek(max(0, os.path.getsize(logname) - 2000))
                            dead_tail = f.read().decode("utf-8",
                                                        "replace")[-1500:]
                    except Exception:
                        pass
                    break
                if self._health():
                    try:
                        os.unlink(logname)
                    except Exception:
                        pass
                    self.ngl_used = ngl
                    return self
                time.sleep(2.0)
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
            last_err = (f"ngl={ngl} died during load [{logname}]:\n"
                        f"{dead_tail}")
        raise RuntimeError(
            f"ABORT: llama-server failed all offload attempts "
            f"{tries}. Last error:\n{last_err[-1500:]}")

    def close(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    # -- HTTP (injectable for tests) ------------------------------------
    def _post(self, path, payload, timeout):
        if self._post_fn is not None:
            return self._post_fn(path, payload, timeout)
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout) as c:
            r = c.post(path, json=payload)
            r.raise_for_status()
            return r.json()

    def _stream(self, path, payload, timeout):
        if self._stream_fn is not None:
            yield from self._stream_fn(path, payload, timeout)
            return
        import httpx

        with httpx.Client(base_url=self.base_url, timeout=timeout) as c:
            with c.stream("POST", path, json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    yield line

    # -- chat contract ---------------------------------------------------
    def _payload(self, messages, tools, max_tokens, stop, stream):
        p = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,  # greedy, house rule (tool-call determinism)
            "stream": bool(stream),
            "timings_per_token": True,
            "cache_prompt": True,
        }
        oai = to_openai_tools(tools)
        if oai:
            p["tools"] = oai
        if self.thinking_budget is not None:
            # Prism fork: per-request reasoning effort (0 disables).
            p["thinking_budget_tokens"] = self.thinking_budget
        if stop:
            p["stop"] = list(stop)
        return p

    def chat(self, messages, tools=None, max_tokens=320, stop=None):
        """One full turn. Returns dict with composed ``text`` (never raises
        on model content; raises on transport/server errors)."""
        out = self._post("/v1/chat/completions",
                         self._payload(messages, tools, max_tokens, stop,
                                       False),
                         timeout=600)
        msg = (out.get("choices") or [{}])[0].get("message", {})
        tcs = [{"function": {"name": (tc.get("function") or {}).get("name", ""),
                               "arguments": (tc.get("function") or {}).get(
                                   "arguments", "") or ""}}
               for tc in (msg.get("tool_calls") or [])]
        t = out.get("timings", {})
        n = int(t.get("predicted_n", 0) or 0)
        ms = float(t.get("predicted_ms", 0.0) or 0.0)
        return {
            "text": compose_raw(msg.get("content"), tcs,
                                msg.get("reasoning_content", "")),
            "tool_calls": tcs,
            "ttft": float(t.get("prompt_ms", 0.0) or 0.0) / 1000.0,
            "decode_tps": (1000.0 * n / ms) if ms > 0 and n else 0.0,
        }

    def chat_stream(self, acc, messages, tools=None, max_tokens=320, stop=None):
        """Yield ("text"|"think", piece) live, then one composed tool block.

        Reasoning deltas stream tagged (callers render them into a think
        trace live); the tail carries tool calls only — the live think
        pieces already accumulated into the caller's raw text, so putting
        the full reasoning here too would duplicate it. Stats land in
        ``acc`` (``ttft``/``decode_tps``/``tool_calls``).
        """
        tcs: dict = {}
        for line in self._stream(
                "/v1/chat/completions",
                self._payload(messages, tools, max_tokens, stop, True),
                timeout=600):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except Exception:
                continue
            delta = (ev.get("choices") or [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield ("text", str(content))
            rc = delta.get("reasoning_content")
            if rc:
                yield ("think", str(rc))
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tcs.setdefault(idx, {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = str(fn["name"])
                if fn.get("arguments"):
                    slot["arguments"] += str(fn["arguments"])
            if ev.get("timings"):
                t = ev["timings"]
                n = int(t.get("predicted_n", 0) or 0)
                ms = float(t.get("predicted_ms", 0.0) or 0.0)
                acc["ttft"] = float(t.get("prompt_ms", 0.0) or 0.0) / 1000.0
                acc["decode_tps"] = (1000.0 * n / ms) if ms > 0 and n else 0.0
        ordered = [{"function": {"name": v["name"], "arguments": v["arguments"]}}
                   for _, v in sorted(tcs.items()) if v["name"]]
        acc["tool_calls"] = ordered
        tail = compose_raw("", ordered)
        if tail:
            yield ("text", tail)
