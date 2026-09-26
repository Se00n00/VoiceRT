"""Gemma-4-E4B-it Q4_K_M leg via a local llama.cpp sidecar (CPU-only).

Big brain on CPU/RAM, fast hands on GPU: this leg keeps zero VRAM (the
ONE-GPU-process rule is unaffected) and serves the 8B model through a
``llama-server`` subprocess (or an already-healthy one it attaches to)
over its OpenAI-compatible API. The server renders Google's canonical
Gemma-4 template itself, so native tool calls come back structured::

    <|tool_call>call:list{"path": "."}<tool_call|>

Grammar of the composed raw text (contract for the S3 parser
``parse_gemma_action`` in :mod:`src.tools.terminal`): optional chat
``content``, then zero or more ``<|tool_call>call:NAME ARGS <tool_call|>``
blocks where ``ARGS`` is the raw JSON arguments object string. Never the
other way round, never anything between blocks.

Matches the :class:`MiniCPMH Fused` selection contract so
:class:`LlmModel` can select it via ``LlmConfig(backend="gemma")``,
with two deliberate differences (the GGUF repo ships no tokenizer
files, so the ids path cannot work):

- ``is_sidecar = True``: :class:`LlmModel` passes ``(messages, tools)``
  straight through instead of encoding to ids first.
- ``chat`` / ``chat_stream`` speak text; streamed pieces carry
  ``token_id=-1`` upstream (callers already fall back to joined pieces).

RAM preflight fails fast below ``min_ram_gb`` (same spirit as
``assert_cuda_leg``): a swapped 5.3GB model is worse than no model.
"""

import atexit
import json
import os
import subprocess
import time
import urllib.request

__all__ = [
    "GemmaLlamaCpp",
    "compose_raw",
    "to_openai_tools",
    "resolve_gguf",
    "resolve_server_bin",
    "mem_available_gb",
]

REPO_ID = "lmstudio-community/gemma-4-E4B-it-GGUF"
GGUF_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"
GGUF_BYTES = 5335291936  # exact file size; completeness check after download

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


def resolve_gguf(explicit=None):
    """Find the Q4_K_M file: explicit path, else HF cache download."""
    if explicit and str(explicit) != "auto":
        p = os.path.abspath(os.path.expanduser(str(explicit)))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"gemma gguf not found: {p}")
        return p
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo_id=REPO_ID, allow_patterns=[GGUF_FILE])
    p = os.path.join(root, GGUF_FILE)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"gemma gguf missing after download: {p}")
    return p


def resolve_server_bin(explicit=None):
    """Find llama-server: explicit path, ./third_party, PATH, else error."""
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
    for p in cands:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(
        "llama-server not found. Pass LlmConfig(gemma_bin='/path/to/llama-server') "
        "or build llama.cpp (cmake -DGGML_NATIVE=ON, target llama-server).")


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


def compose_raw(content, tool_calls):
    """Compose parser-target raw text: content + native tool blocks.

    ``tool_calls`` are OpenAI-style ``{"name":..., "arguments": str|dict}``.
    """
    parts = []
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


class GemmaLlamaCpp:
    """CPU sidecar leg. ``warm()`` attaches or spawns; ``close()`` frees."""

    is_sidecar = True

    def __init__(self, gguf_path="auto", host="127.0.0.1", port=8080,
                 n_ctx=4096, threads=0, server_bin="auto",
                 min_ram_gb=6.0, startup_timeout=180,
                 _post_fn=None, _stream_fn=None, _popen=None):
        self.gguf_path = gguf_path
        self.host = host
        self.port = int(port)
        self.n_ctx = int(n_ctx)
        self.threads = int(threads)
        self.server_bin = server_bin
        self.min_ram_gb = float(min_ram_gb)
        self.startup_timeout = float(startup_timeout)
        self._post_fn = _post_fn
        self._stream_fn = _stream_fn
        self._popen = _popen or subprocess.Popen
        self._proc = None
        self.base_url = f"http://{host}:{int(port)}"

    # -- lifecycle ------------------------------------------------------
    def _health(self) -> bool:
        try:
            with urllib.request.urlopen(
                    self.base_url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def _serving_model(self) -> str:
        try:
            with urllib.request.urlopen(
                    self.base_url + "/v1/models", timeout=10) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            items = data.get("data", [])
            return str(items[0].get("id", "")) if items else ""
        except Exception:
            return ""

    def warm(self):
        """Attach to a healthy server or spawn one. Raises on failure."""
        if self._health():
            return self
        free = mem_available_gb()
        if 0.0 < free < self.min_ram_gb:
            raise MemoryError(
                f"ABORT: {free:.1f}GB RAM free, need >={self.min_ram_gb:.0f}GB "
                f"for the 5.3GB Q4 model + context. Close apps and retry. "
                f"Refusing swap-death on purpose.")
        gguf = resolve_gguf(self.gguf_path)
        size = os.path.getsize(gguf)
        if size != GGUF_BYTES:
            raise ValueError(
                f"ABORT: {gguf} is {size} bytes, expected {GGUF_BYTES} "
                f"(incomplete download?). Re-fetch and retry.")
        if self._health():
            return self
        bin_path = resolve_server_bin(self.server_bin)
        cmd = [bin_path, "-m", gguf, "--port", str(self.port),
               "-c", str(self.n_ctx), "--n-gpu-layers", "0",
               "--log-disable"]
        if self.threads > 0:
            cmd += ["-t", str(self.threads)]
        logf = open(os.devnull, "w")
        self._proc = self._popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.close)
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.startup_timeout:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"ABORT: llama-server exited during load "
                    f"(rc={self._proc.poll()}).")
            if self._health():
                return self
            time.sleep(2.0)
        self.close()
        raise TimeoutError(
            f"ABORT: llama-server not healthy after {self.startup_timeout:.0f}s.")

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
        if stop:
            p["stop"] = list(stop)
        return p

    def chat(self, messages, tools=None, max_tokens=320, stop=None):
        """One full turn. Returns dict with composed ``text`` (never raises
        on model content; raises on transport/server errors)."""
        out = self._post("/v1/chat/completions",
                         self._payload(messages, tools, max_tokens, stop, False),
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
            "text": compose_raw(msg.get("content"), tcs),
            "tool_calls": tcs,
            "ttft": float(t.get("prompt_ms", 0.0) or 0.0) / 1000.0,
            "decode_tps": (1000.0 * n / ms) if ms > 0 and n else 0.0,
        }

    def chat_stream(self, acc, messages, tools=None, max_tokens=320, stop=None):
        """Yield live content pieces, then one composed tool block.

        Stats land in ``acc`` (``ttft``/``decode_tps``/``tool_calls``).
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
                yield str(content)
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
            yield tail
