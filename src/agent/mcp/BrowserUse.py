"""
Canonical BrowserUse tool implementations: web fetch, search, browser.

These plain functions are THE BrowserUse tool implementations —
:mod:`src.agent.mcp.mcp` registers them over MCP via ``app.add_tool``.
All functions are sync and total (never raise — errors become strings).

contract, dependencies, who calls it: webfetch (page as text),
websearch (TinyFish API first with quota + cache, DuckDuckGo
fallback), open_url/download plus Playwright Chromium actions;
depends on httpx (and playwright only for browser_*); used by the
MCP registry. Secrets never live here — keys travel via environment.

Search providers (``websearch``): TinyFish Search API first
(``GET https://api.search.tinyfish.ai``, ``X-API-Key`` from
``TINYFISH_API_KEY``), DuckDuckGo fallback. ``SEARCH_PROVIDER`` forces
``tinyfish`` / ``ddg``; default ``auto`` picks TinyFish when a key is
present. TinyFish calls respect the free-tier quota (30/min + 500/hr)
and a 15-min result cache, both file-backed so they hold across
server restarts.
"""
import asyncio as _asyncio
import concurrent.futures as _futures
import hashlib as _hl
import html as _html
import json as _json
import os as _os
import re as _re
import shutil as _shutil
import subprocess as _sp
import time as _time
import urllib.parse as _urlparse

import httpx as _httpx

try:
    import fcntl as _fcntl
except ImportError:  # no fcntl on Windows; locked paths fail open
    _fcntl = None

__all__ = [
    "webfetch",
    "websearch",
    "open_url",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "download_file",
]

_SNAPSHOT_CHARS = 2000


def _load_dotenv() -> None:
    """Load repo-root ``.env`` into the environment (first import only).

    Simple ``KEY=VALUE`` lines; blanks and ``#`` comments skipped, single
    or double quotes stripped. Real environment wins (``setdefault``), so
    exported vars and CI secrets always override the file. Best-effort:
    a missing/unreadable file is normal (key-less DDG fallback).
    """
    if _os.environ.get("VOICE_DOTENV_LOADED"):
        return
    try:
        here = _os.path.abspath(__file__)
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


def _to_int(value, default: int) -> int:
    """int or the default (digit check, no coercion tricks).

    ..
    """
    s = str(value).strip()
    return int(s) if s.lstrip("-").isdigit() else default


def _check_url(url: str) -> str:
    """Trimmed URL or "" unless http(s).

    ..
    """
    u = str(url or "").strip()
    if not _re.match(r"^https?://", u, _re.IGNORECASE):
        return ""
    return u


# -- webfetch ----------------------------------------------------------

_FETCH_BYTES = 200_000


def _strip_html(page: str) -> str:
    """Drop scripts/styles/tags, unescape, collapse whitespace."""
    page = _re.sub(r"(?is)<(script|style)[^>]*>.*?</\1\s*>", " ", page)
    page = _re.sub(r"(?s)<!--.*?-->", " ", page)
    page = _re.sub(r"<[^>]+>", " ", page)
    text = _html.unescape(page)
    return _re.sub(r"\s+", " ", text).strip()


def webfetch(url: str, max_chars: int = 6000) -> str:
    """Fetch an http(s) URL and return its text. Never raises."""
    link = str(url or "").strip()
    if not _re.match(r"^https?://", link, _re.IGNORECASE):
        return "error: only http(s) URLs may be fetched"
    try:
        r = _httpx.get(link, timeout=20, follow_redirects=True,
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
            text = _re.sub(r"\s+", " ", text).strip()
    except Exception as exc:
        return f"error: decode failed ({exc})"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text or "(empty page)"


# -- websearch: DuckDuckGo fallback --------------------------------------

_DDG_URL = "https://html.duckduckgo.com/html/"
_DDG_RESULT_RE = _re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    _re.DOTALL | _re.IGNORECASE)
_DDG_SNIPPET_RE = _re.compile(
    r'class="result__snippet"[^>]*>(.*?)</(?:a|div)',
    _re.DOTALL | _re.IGNORECASE)
_DDG_TAG_RE = _re.compile(r"<[^>]+>")


def _ddg_link(raw: str) -> str:
    """Unwrap DDG redirect links (//duckduckgo.com/l/?uddg=...) to the URL."""

    raw = _html.unescape(str(raw or "").strip())
    try:
        q = _urlparse.parse_qs(
            _urlparse.urlsplit(raw).query).get("uddg", [""])
        if q and q[0]:
            return q[0]
    except Exception:
        pass
    return raw


def _ddg_search(query: str, n: int, timeout: int) -> str:
    """Web search (DuckDuckGo HTML endpoint, no keys). Never raises."""

    try:
        r = _httpx.get(_DDG_URL, params={"q": query}, timeout=timeout,
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
        title = _re.sub(r"\s+", " ", _DDG_TAG_RE.sub(" ", title)).strip()
        url = _ddg_link(href)
        snip = ""
        if i < len(snippets):
            snip = _re.sub(r"\s+", " ",
                           _DDG_TAG_RE.sub(" ", _html.unescape(snippets[i]))
                           ).strip()[:300]
        if title or url:
            lines.append(f"- {title}\n  {url}\n  {snip}".rstrip())
    return "\n".join(lines) or "(no results)"


# -- websearch: TinyFish (metered) + shared quota/cache ------------------
#
# Free tier: 30 requests/min (per docs) and 500 requests/hour. Quota
# timestamps and cached results live in one JSON state file
# (``VOICE_SEARCH_STATE`` override, else
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

    return (_os.environ.get("TINYFISH_API_KEY", "") or "").strip()


def _search_state_path() -> str:

    override = (_os.environ.get("VOICE_SEARCH_STATE", "") or "").strip()
    if override:
        return override
    base = (_os.environ.get("XDG_CACHE_HOME", "") or "").strip() \
        or _os.path.join(_os.path.expanduser("~"), ".cache")
    return _os.path.join(base, "voice-pipeline", "search_state.json")


def _with_search_state(fn):
    """Run ``fn(state)`` under an exclusive file lock; save afterwards.

    On any failure (missing fcntl, bad disk) run unlocked on a scratch
    state and skip the save — fail open, never raise.
    """

    path = _search_state_path()
    try:
        _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a+", encoding="utf-8") as f:
            try:
                _fcntl.flock(f.fileno(), _fcntl.LOCK_EX)
                locked = True
            except Exception:
                locked = False
            try:
                f.seek(0)

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

    return _hl.sha256(f"{provider}|{query}|{n}".encode("utf-8")).hexdigest()[:32]


def _cache_get(provider: str, query: str, n: int):

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
        r = _httpx.get(_TINYFISH_URL, params={"query": query},
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
        title = _re.sub(r"\s+", " ", str(item.get("title", "") or "")).strip()
        url = str(item.get("url", "") or "").strip()
        snippet = _re.sub(r"\s+", " ",
                          str(item.get("snippet", "") or "")).strip()[:300]
        if title or url:
            lines.append(f"- {title}\n  {url}\n  {snippet}".rstrip())
    text = "\n".join(lines) or "(no results)"
    _cache_put("tinyfish", query, n, text)
    return text


def websearch(query: str, count: int = 5) -> str:
    """Web search, titles/URLs/snippets. Never raises.

    Provider via ``SEARCH_PROVIDER``: ``auto`` (default — TinyFish when
    ``TINYFISH_API_KEY`` is set, else DuckDuckGo), ``tinyfish``, ``ddg``.
    Results are cached per provider/query (TTL 15 min); TinyFish calls
    additionally respect the 30/min + 500/hr free-tier quota.
    """
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
        return _tinyfish_search(q, n, 15)
    text = _ddg_search(q, n, 15)
    if text and not text.startswith("error:"):
        _cache_put("ddg", q, n, text)
    return text


# -- open_url / download -------------------------------------------------

def _opener() -> str:
    """OS URL-opener binary name ("" when none installed).

    ..
    """
    if _os.name == "nt":
        return "cmd"
    if _os.name == "posix" and _os.uname().sysname != "Linux":
        return "open"
    return "xdg-open" if _shutil.which("xdg-open") else ""


def open_url(url: str) -> str:
    """Open a URL in the OS default browser (detached). Never raises."""
    try:
        u = _check_url(url)
        if not u:
            return "error: only http(s) URLs may be opened"
        opener = _opener()
        if not opener:
            return "error: no URL opener installed"
        if opener == "cmd":
            _sp.Popen(["cmd", "/c", "start", "", u],
                      stdin=_sp.DEVNULL, stdout=_sp.DEVNULL,
                      stderr=_sp.DEVNULL, start_new_session=True)
        else:
            _sp.Popen([opener, u], stdin=_sp.DEVNULL, stdout=_sp.DEVNULL,
                      stderr=_sp.DEVNULL, start_new_session=True)
        return f"opened {u}"
    except Exception as exc:
        return f"error: open failed ({exc})"[:300]


def download_file(url: str, dest: str, max_bytes: int = 50_000_000,
                  timeout: int = 60) -> str:
    """Download an http(s) URL to ``dest``. Never raises.

    Streams to disk (no giant RAM spike), aborts past ``max_bytes``.
    ``dest`` may be a directory (basename taken from the URL).
    """
    u = _check_url(url)
    if not u:
        return "error: only http(s) URLs may be downloaded"
    raw = str(dest or "").strip()
    if not raw:
        return "error: empty dest"
    cap = max(1024, min(_to_int(max_bytes, 50_000_000), 500_000_000))
    t = max(5, _to_int(timeout, 60))
    d = _os.path.abspath(_os.path.expanduser(raw))
    try:
        if raw.endswith((_os.sep, "/")) or _os.path.isdir(d):
            base = u.rstrip("/").split("/")[-1].split("?")[0] or "download"
            d = _os.path.join(d, base)
        parent = _os.path.dirname(d)
        if parent:
            _os.makedirs(parent, exist_ok=True)
        n = 0
        with _httpx.stream("GET", u, timeout=t, follow_redirects=True,
                           headers={"User-Agent": "voice-pipeline/download"}) as r:
            r.raise_for_status()
            with open(d, "wb") as f:
                for chunk in r.iter_bytes(65536):
                    n += len(chunk)
                    if n > cap:
                        raise ValueError(
                            f"over cap ({cap} bytes), partial file kept")
                    f.write(chunk)
    except Exception as exc:
        return f"error: download failed ({exc})"[:300]
    return f"saved {d} ({n} bytes)"


# -- playwright-backed browser actions -------------------------------------

def _playwright():
    """Playwright async API or None (import guard, the only import try).

    ..
    """
    try:
        from playwright.async_api import async_playwright
        return async_playwright
    except Exception:
        return None


def _run_async(coro, timeout_s: int):
    """Drive one coroutine to completion from sync code.

    Plain ``asyncio.run`` when no loop runs in this thread; otherwise a
    helper thread with its own loop (a sync caller cannot share the
    running loop, and the sync Playwright API refuses to run under it).
    ..
    """
    try:
        _asyncio.get_running_loop()
    except RuntimeError:
        return _asyncio.run(coro)
    with _futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_asyncio.run, coro).result(timeout=timeout_s)


async def _drive_async(url: str, op: tuple, timeout_ms: int,
                       headless: bool) -> str:
    """Async core: launch Chromium, run ``op``, snapshot, close.

    ``op`` is ("click", sel, ms), ("type", sel, text, submit, ms), or
    ("scroll", dy). The browser always closes (finally), so a failed
    action still reports instead of leaking a process.
    ..
    """
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=bool(headless))
        try:
            page = await browser.new_page()
            await page.goto(url, timeout=timeout_ms,
                            wait_until="domcontentloaded")
            kind = op[0]
            if kind == "click":
                await page.click(op[1], timeout=op[2])
            elif kind == "type":
                await page.fill(op[1], op[2], timeout=op[4])
                if op[3]:
                    await page.keyboard.press("Enter")
            elif kind == "scroll":
                await page.evaluate(f"window.scrollBy(0, {int(op[1])})")
            title = await page.title()
            cur = page.url
            body = await page.evaluate(
                "() => (document.body && document.body.innerText) || ''")
        finally:
            await browser.close()
    body = _re.sub(r"\s+", " ", str(body)).strip()[:_SNAPSHOT_CHARS]
    return f"title: {title}\nurl: {cur}\n{body or '(no text)'}"


def _drive(url: str, op: tuple, timeout: int, headless: bool) -> str:
    """Launch Chromium, goto url, run ``op``, snapshot, close.

    Sync wrapper around the async driver: safe with or without a
    running asyncio loop in this thread. Total — errors become strings.
    ..
    """
    try:
        if _playwright() is None:
            return ("error: browser automation needs playwright in the bridge "
                    "venv (.venv/bin/python -m pip install playwright && "
                    ".venv/bin/python -m playwright install chromium)")
        u = _check_url(url)
        if not u:
            return "error: only http(s) URLs may be driven"
        t = max(5, min(_to_int(timeout, 15), 120))
        return _run_async(_drive_async(u, op, t * 1000, headless), t + 60)
    except Exception as exc:
        return f"error: browser action failed ({exc})"[:400]


def browser_click(url: str, selector: str, timeout: int = 15,
                  headless: bool = True) -> str:
    """Click a CSS ``selector`` on ``url``; report the page after."""
    sel = str(selector or "").strip()
    if not sel:
        return "error: empty selector"
    ms = max(5, min(_to_int(timeout, 15), 120)) * 1000
    return _drive(url, ("click", sel, ms), timeout, headless)


def browser_type(url: str, selector: str, text: str, submit: bool = False,
                 timeout: int = 15, headless: bool = True) -> str:
    """Fill ``text`` into a CSS ``selector`` on ``url`` (+Enter if submit)."""
    sel = str(selector or "").strip()
    if not sel:
        return "error: empty selector"
    src = str(text or "")
    if len(src) > 2000:
        return "error: text over 2000 chars (split it up)"
    ms = max(5, min(_to_int(timeout, 15), 120)) * 1000
    return _drive(url, ("type", sel, src, bool(submit), ms), timeout,
                  headless)


def browser_scroll(url: str, direction: str = "down", pixels: int = 800,
                   timeout: int = 15, headless: bool = True) -> str:
    """Scroll ``url`` up/down by pixels; report the page after."""
    d = (str(direction or "down").strip().lower() or "down")
    if d not in ("up", "down"):
        return "error: direction must be 'up' or 'down'"
    px = max(1, min(_to_int(pixels, 800), 10000))
    dy = px if d == "down" else -px
    return _drive(url, ("scroll", dy), timeout, headless)
