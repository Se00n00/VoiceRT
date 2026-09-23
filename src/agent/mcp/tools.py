"""Canonical extra-tool implementations: python_exec, fetch, web_search.

Single-definition rule: these plain functions are THE implementations.
- ``server.py`` (same package) exposes them over MCP as thin wrappers
  (arg names are the agent-facing ones: ``code`` / ``path`` / ``pattern``).
- ``client.py`` loads them into the agent as LangChain tools through a
  local MCP stdio server (``python -m src.agent.mcp.server``).

All functions are sync and total (never raise — errors become strings).
File/shell work is covered by deepagents' built-in backend tools, so this
module carries only what the backend lacks: host python, web fetch,
web search.

Search providers (``web_search``): TinyFish Search API first
(``GET https://api.search.tinyfish.ai``, ``X-API-Key`` from
``TINYFISH_API_KEY``), DuckDuckGo fallback. ``SEARCH_PROVIDER`` forces
``tinyfish`` / ``ddg``; default ``auto`` picks TinyFish when a key is
present. TinyFish calls respect the free-tier quota (30/min + 500/hr)
and a 15-min result cache, both file-backed so they hold across the
per-call MCP server processes.
"""
import html
import re

__all__ = [
    "fetch",
    "python_exec",
    "web_search",
]


def _load_dotenv() -> None:
    """Load repo-root ``.env`` into the environment (first import only).

    Simple ``KEY=VALUE`` lines; blanks and ``#`` comments skipped, single
    or double quotes stripped. Real environment wins (``setdefault``), so
    exported vars and CI secrets always override the file. Best-effort:
    a missing/unreadable file is normal (key-less DDG fallback).
    """
    import os as _os

    if _os.environ.get("VOICE_DOTENV_LOADED"):
        return
    try:
        here = _os.path.abspath(__file__)  # .../voice-pipeline/src/agent/mcp/tools.py
        root = _os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.dirname(here))))
        path = _os.path.join(root, ".env")
        with open(path, "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip("'").strip('"').strip()
                if k and k not in _os.environ:
                    _os.environ[k] = v
    except Exception:
        pass
    finally:
        try:
            _os.environ.setdefault("VOICE_DOTENV_LOADED", "1")
        except Exception:
            pass


_load_dotenv()


def python_exec(code: str, timeout: int = 30) -> str:
    """Run Python code on this host. Never raises.

    Same trust level as shell execution — for local use only. The source
    is base64-piped to ``python3 -`` so arbitrary payloads need no shell
    quoting. One-shot per call with a process kill on timeout.
    """
    import base64
    import subprocess
    import sys

    src = str(code or "")
    if not src.strip():
        return "error: empty code"
    try:
        t = max(1, int(timeout))
    except Exception:
        t = 30
    b64 = base64.b64encode(src.encode("utf-8")).decode("ascii")
    cmd = [sys.executable or "python3", "-c",
           ("import base64,sys; exec(compile(base64.b64decode(sys.argv[1])"
            ".decode('utf-8'), '<agent>', 'exec'))"), b64]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=t)
    except subprocess.TimeoutExpired:
        return f"timeout after {t}s"
    except Exception as exc:
        return f"error: {exc}"
    out = ((p.stdout or "") + (p.stderr or "")).strip() or "(no output)"
    return f"rc={p.returncode}\n{out[:6000]}"


# -- web: fetch --------------------------------------------------------

_FETCH_BYTES = 200_000


def _strip_html(page: str) -> str:
    """Drop scripts/styles/tags, unescape, collapse whitespace."""
    page = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1\s*>", " ", page)
    page = re.sub(r"(?s)<!--.*?-->", " ", page)
    page = re.sub(r"<[^>]+>", " ", page)
    text = html.unescape(page)
    return re.sub(r"\s+", " ", text).strip()


def fetch(url: str, max_chars: int = 6000, timeout: int = 20) -> str:
    """Fetch an http(s) URL and return its text. Never raises."""
    import httpx

    url = str(url or "").strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return "error: only http(s) URLs may be fetched"
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "voice-pipeline/fetch"})
        r.raise_for_status()
    except Exception as exc:
        return f"error: fetch failed ({exc})"[:500]
    try:
        raw = r.content[:_FETCH_BYTES]
        ctype = (r.headers.get("content-type", "") or "").lower()
        text = raw.decode("utf-8", errors="replace")
        if "html" in ctype:
            text = _strip_html(text)
        else:
            text = re.sub(r"\s+", " ", text).strip()
    except Exception as exc:
        return f"error: decode failed ({exc})"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text or "(empty page)"


# -- direct web search (no SearXNG instance needed) -----------------------

_DDG_URL = "https://html.duckduckgo.com/html/"
_DDG_RESULT_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE)
_DDG_SNIPPET_RE = re.compile(
    r'class="result__snippet"[^>]*>(.*?)</(?:a|div)',
    re.DOTALL | re.IGNORECASE)
_DDG_TAG_RE = re.compile(r"<[^>]+>")


def _ddg_link(raw: str) -> str:
    """Unwrap DDG redirect links (//duckduckgo.com/l/?uddg=...) to the URL."""
    import urllib.parse

    raw = html.unescape(str(raw or "").strip())
    try:
        q = urllib.parse.parse_qs(
            urllib.parse.urlsplit(raw).query).get("uddg", [""])
        if q and q[0]:
            return q[0]
    except Exception:
        pass
    return raw


def _ddg_search(query: str, n: int, timeout: int) -> str:
    """Web search (DuckDuckGo HTML endpoint, no keys). Never raises."""
    import httpx

    try:
        r = httpx.get(_DDG_URL, params={"q": query}, timeout=timeout,
                      follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 voice-pipeline/websearch"})
        r.raise_for_status()
        page = r.content.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"error: web search failed ({exc})"[:400]
    anchors = _DDG_RESULT_RE.findall(page)
    snippets = _DDG_SNIPPET_RE.findall(page)
    lines = []
    for i, (href, title) in enumerate(anchors[:n]):
        title = re.sub(r"\s+", " ", _DDG_TAG_RE.sub(" ", title)).strip()
        url = _ddg_link(href)
        snip = ""
        if i < len(snippets):
            snip = re.sub(r"\s+", " ",
                          _DDG_TAG_RE.sub(" ", html.unescape(snippets[i]))
                          ).strip()[:300]
        if title or url:
            lines.append(f"- {title}\n  {url}\n  {snip}".rstrip())
    return "\n".join(lines) or "(no results)"


# -- TinyFish search (metered) + shared quota/cache ----------------------
#
# Free tier: 30 requests/min (per docs) and 500 requests/hour. The MCP
# tool server runs each call in a FRESH process, so a process-local
# limiter would be useless — quota timestamps and cached results live in
# one JSON state file (``VOICE_SEARCH_STATE`` override, else
# ``~/.cache/voice-pipeline/search_state.json``), guarded by fcntl.
# Cache hits and DDG calls consume no TinyFish quota. Everything here is
# best-effort and total: state failures fail OPEN (allow the call) so a
# full disk never breaks the agent loop.

_TINYFISH_URL = "https://api.search.tinyfish.ai"

_MINUTE_QUOTA = 30
_MINUTE_WINDOW_S = 60.0
_HOUR_QUOTA = 500
_HOUR_WINDOW_S = 3600.0
_SEARCH_CACHE_TTL_S = 900.0
_SEARCH_CACHE_CAP = 200


def _tinyfish_key() -> str:
    """API key from the environment only. Never logged, never stored."""
    import os as _os

    return (_os.environ.get("TINYFISH_API_KEY", "") or "").strip()


def _search_state_path() -> str:
    import os as _os

    override = (_os.environ.get("VOICE_SEARCH_STATE", "") or "").strip()
    if override:
        return override
    base = (_os.environ.get("XDG_CACHE_HOME", "") or "").strip() \
        or _os.path.join(_os.path.expanduser("~"), ".cache")
    return _os.path.join(base, "voice-pipeline", "search_state.json")


def _load_search_state(path: str) -> dict:
    try:
        import json as _json

        with open(path, "r", encoding="utf-8") as f:
            st = _json.load(f)
        if not isinstance(st, dict):
            return {}
        calls = st.get("calls")
        cache = st.get("cache")
        return {
            "calls": [float(t) for t in calls] if isinstance(calls, list) else [],
            "cache": dict(cache) if isinstance(cache, dict) else {},
        }
    except Exception:
        return {}


def _save_search_state(path: str, state: dict) -> None:
    try:
        import json as _json
        import os as _os

        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            _json.dump(state, f)
        _os.replace(tmp, path)
    except Exception:
        pass


def _with_search_state(fn):
    """Run ``fn(state)`` under an exclusive file lock; save afterwards.

    On any failure (missing fcntl, bad disk) run unlocked on a scratch
    state and skip the save — fail open, never raise.
    """
    import os as _os

    path = _search_state_path()
    try:
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a+", encoding="utf-8") as f:
            try:
                import fcntl as _fcntl

                _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
                locked = True
            except Exception:
                locked = False
            try:
                f.seek(0)
                import json as _json

                try:
                    raw = _json.load(f)
                except Exception:
                    raw = {}
                state = {
                    "calls": [float(t) for t in raw.get("calls", [])]
                    if isinstance(raw.get("calls"), list) else [],
                    "cache": dict(raw["cache"])
                    if isinstance(raw.get("cache"), dict) else {},
                }
                out = fn(state)
                try:
                    f.seek(0)
                    f.truncate()
                    _json.dump(state, f)
                except Exception:
                    pass
                return out
            finally:
                if locked:
                    try:
                        _fcntl.flock(f.fileno(), _fcntl.LOCK_UN)
                    except Exception:
                        pass
    except Exception:
        return fn({"calls": [], "cache": {}})


def _cache_key(provider: str, query: str, n: int) -> str:
    import hashlib as _hl

    return _hl.sha256(f"{provider}|{query}|{n}".encode("utf-8")).hexdigest()[:32]


def _cache_get(provider: str, query: str, n: int):
    import time as _time

    def _get(state):
        now = _time.time()
        ent = state.get("cache", {}).get(_cache_key(provider, query, n))
        if isinstance(ent, dict) and now - float(ent.get("at", 0)) < _SEARCH_CACHE_TTL_S:
            return str(ent.get("result", ""))
        return None

    try:
        return _with_search_state(_get)
    except Exception:
        return None


def _cache_put(provider: str, query: str, n: int, result: str) -> None:
    import time as _time

    def _put(state):
        cache = state.setdefault("cache", {})
        cache[_cache_key(provider, query, n)] = {"at": _time.time(), "result": result}
        while len(cache) > _SEARCH_CACHE_CAP:
            oldest = min(cache, key=lambda k: float(cache[k].get("at", 0)))
            del cache[oldest]

    try:
        _with_search_state(_put)
    except Exception:
        pass


def _quota_check():
    """Minute+hour sliding windows over recorded TinyFish calls.

    Returns ``(allowed, retry_in_s)``. State failures fail open.
    """
    import time as _time

    def _check(state):
        now = _time.time()
        calls = [t for t in state.get("calls", [])
                 if now - t < _HOUR_WINDOW_S]
        state["calls"] = calls
        minute = [t for t in calls if now - t < _MINUTE_WINDOW_S]
        if len(minute) >= _MINUTE_QUOTA:
            return False, max(1, int(min(minute) + _MINUTE_WINDOW_S - now) + 1)
        if len(calls) >= _HOUR_QUOTA:
            return False, max(1, int(min(calls) + _HOUR_WINDOW_S - now) + 1)
        return True, 0

    try:
        return _with_search_state(_check)
    except Exception:
        return True, 0


def _quota_record() -> None:
    import time as _time

    def _rec(state):
        state.setdefault("calls", []).append(_time.time())

    try:
        _with_search_state(_rec)
    except Exception:
        pass


def _tinyfish_search(query: str, n: int, timeout: int) -> str:
    """Search via the TinyFish Search API. Never raises.

    Auth: ``X-API-Key`` from ``TINYFISH_API_KEY`` (env only). Consumes
    one quota unit per attempt; results are cached (see above).
    """
    import httpx

    key = _tinyfish_key()
    if not key:
        return ("error: tinyfish search needs TINYFISH_API_KEY in the "
                "environment (create one at agent.tinyfish.ai/api-keys). "
                "Fallback: SEARCH_PROVIDER=ddg.")
    allowed, retry_in = _quota_check()
    if not allowed:
        return (f"error: tinyfish search quota exhausted "
                f"(30/min, 500/hr free tier); retry in ~{retry_in}s.")
    try:
        r = httpx.get(_TINYFISH_URL, params={"query": query},
                      timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "voice-pipeline/websearch",
                               "X-API-Key": key})
    except Exception as exc:
        _quota_record()
        return f"error: tinyfish search failed ({exc})"[:400]
    _quota_record()
    code = getattr(r, "status_code", 200)
    if code == 401:
        return "error: tinyfish search rejected the API key (401)"
    if code == 402:
        return "error: tinyfish search needs Search API access (402)"
    if code == 429:
        return ("error: tinyfish search rate-limited (429); "
                "backing off, retry in ~60s.")
    if code in (403, 404):
        return f"error: tinyfish search unavailable ({code})"
    try:
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        hint = "; retry with backoff" if code >= 500 else ""
        return f"error: tinyfish search bad response ({exc}){hint}"[:400]
    try:
        results = payload.get("results", []) or []
    except Exception:
        return "error: tinyfish search bad response shape"
    lines = []
    for item in results[:n]:
        if not isinstance(item, dict):
            continue
        title = re.sub(r"\s+", " ", str(item.get("title", "") or "")).strip()
        url = str(item.get("url", "") or "").strip()
        snippet = re.sub(r"\s+", " ",
                         str(item.get("snippet", "") or "")).strip()[:300]
        if title or url:
            lines.append(f"- {title}\n  {url}\n  {snippet}".rstrip())
    text = "\n".join(lines) or "(no results)"
    _cache_put("tinyfish", query, n, text)
    return text


def web_search(query: str, count: int = 5, timeout: int = 15) -> str:
    """Web search. Never raises.

    Provider via ``SEARCH_PROVIDER``: ``auto`` (default — TinyFish when
    ``TINYFISH_API_KEY`` is set, else DuckDuckGo), ``tinyfish``, ``ddg``.
    Results are cached per provider/query (TTL 15 min); TinyFish calls
    additionally respect the 30/min + 500/hr free-tier quota.
    """
    import os as _os

    q = str(query or "").strip()
    if not q:
        return "error: empty query"
    try:
        n = max(1, min(int(count), 10))
    except Exception:
        n = 5
    want = (_os.environ.get("SEARCH_PROVIDER", "") or "").strip().lower() or "auto"
    if want == "tinyfish" and not _tinyfish_key():
        return ("error: SEARCH_PROVIDER=tinyfish but TINYFISH_API_KEY is "
                "not set; unset SEARCH_PROVIDER for auto fallback.")
    provider = "tinyfish" if (want == "tinyfish"
                              or (want == "auto" and _tinyfish_key())) else "ddg"
    hit = _cache_get(provider, q, n)
    if hit:
        return hit
    if provider == "tinyfish":
        return _tinyfish_search(q, n, timeout)
    text = _ddg_search(q, n, timeout)
    if text and not text.startswith("error:"):
        _cache_put("ddg", q, n, text)
    return text
