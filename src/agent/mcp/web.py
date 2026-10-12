"""Canonical web-automation implementations (browser + download).

Single-definition rule (same as :mod:`src.agent.mcp.tools`): these plain
functions are THE implementations — :mod:`src.agent.mcp.server` exposes
them over MCP as thin wrappers. All functions are sync and total (never
raise — errors become strings).

- ``search_web`` / ``extract_page`` delegate to the canonical
  :mod:`src.agent.mcp.tools` implementations (provider quota + cache
  live there — one definition, no drift).
- ``open_url`` hands the URL to the OS default browser (detached).
- ``download_file`` streams to disk with a byte cap (httpx, in
  requirements).
- ``browser_click`` / ``browser_type`` / ``browser_scroll`` drive a real
  Chromium via Playwright (needs the bridge venv: ``.venv/bin/python -m
  pip install playwright`` + ``.venv/bin/python -m playwright install
  chromium`` — never bare system pip, it is externally managed here).
  Each call is stateless — launch,
  navigate to ``url``, act, report a compact snapshot, close — which
  fits the MCP server's fresh-process-per-call shape with no daemon.
"""
import asyncio as _asyncio
import concurrent.futures as _futures
import os as _os
import re as _re
import shutil as _shutil
import subprocess as _sp

from src.agent.mcp.tools import fetch as _fetch
from src.agent.mcp.tools import web_search as _web_search

__all__ = [
    "search_web",
    "open_url",
    "extract_page",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "download_file",
]

_SNAPSHOT_CHARS = 2000


def _check_url(url: str) -> str:
    u = str(url or "").strip()
    if not _re.match(r"^https?://", u, _re.IGNORECASE):
        return ""
    return u


def _to_int(value, default: int) -> int:
    """int or the default (digit check, no coercion tricks).

    ..
    """
    s = str(value).strip()
    return int(s) if s.lstrip("-").isdigit() else default


def search_web(query: str, count: int = 5) -> str:
    """Web search (titles/URLs/snippets). Never raises.

    Same providers/quota/cache as the canonical ``web_search`` — this
    is the web-category alias of that one definition.
    """
    return _web_search(query, count=max(1, min(_to_int(count, 5), 10)))


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


def extract_page(url: str, max_chars: int = 6000) -> str:
    """Extract an http(s) page as text. Never raises.

    Canonical fetch lives in :mod:`src.agent.mcp.tools` — this is the
    web-category alias of that one definition.
    """
    return _fetch(url, max_chars=max(256, min(_to_int(max_chars, 6000),
                                              100_000)))


def download_file(url: str, dest: str, max_bytes: int = 50_000_000,
                  timeout: int = 60) -> str:
    """Download an http(s) URL to ``dest``. Never raises.

    Streams to disk (no giant RAM spike), aborts past ``max_bytes``.
    ``dest`` may be a directory (basename taken from the URL).
    """
    import httpx

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
        with httpx.stream("GET", u, timeout=t, follow_redirects=True,
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
