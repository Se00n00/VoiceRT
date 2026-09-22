"""Canonical MCP tool implementations: scoped filesystem, docker exec, web.

Single-definition rule: these plain functions are THE implementations.
- ``server.py`` (same package) exposes them over MCP as thin wrappers.
- ``src/agent/terminal.py`` ``run_command()`` delegates its file ops here
  with ``roots=(turn cwd,)``; ``fetch``/``searxng`` dispatch here too.
- ``build_terminal_tools()`` wraps them for the turn-scoped gated
  LangChain toolset (confirm gates live there, not here).

All functions are sync and total (never raise — errors become strings).
Filesystem functions take explicit ``roots``; the MCP surface defaults to
``~/Documents`` + ``~/Projects`` (``MCP_FS_ROOTS`` env overrides with an
``os.pathsep``-separated list).
"""
import html
import os
import re
import tempfile
import threading

from src.sandbox.docker import DockerSandbox, DockerSandboxError

__all__ = [
    "DEFAULT_ROOTS",
    "default_roots",
    "fs_read",
    "fs_write",
    "fs_edit",
    "fs_list",
    "fs_grep",
    "docker_exec",
    "fetch",
    "python_exec",
    "searxng_search",
]


def default_roots() -> list:
    """Allowed filesystem roots: env override, else ~/Documents+~/Projects.

    Only existing directories are kept — a missing home dir is not a
    usable root, and surfacing "(none configured)" beats late ENOENTs.
    """
    env = os.environ.get("MCP_FS_ROOTS", "")
    if env.strip():
        cands = [os.path.abspath(os.path.expanduser(p.strip()))
                 for p in env.split(os.pathsep) if p.strip()]
    else:
        cands = [os.path.abspath(os.path.expanduser(p))
                 for p in ("~/Documents", "~/Projects")]
    return [d for d in cands if os.path.isdir(d)]


def _resolve(path: str, roots: list) -> str | None:
    """Abspath if ``path`` (file or would-be file) lives under roots."""
    full = os.path.abspath(os.path.join(roots[0], path or ".") if roots else "")
    if not roots:
        return None
    # absolute paths stay absolute; relative ones resolve under first root
    full = os.path.abspath(path) if os.path.isabs(path or "") else full
    for base in roots:
        base = os.path.abspath(base)
        if full == base or full.startswith(base + os.sep):
            return full
    return None


def _deny(roots: list) -> str:
    shown = ", ".join(roots) if roots else "(none configured)"
    return f"denied: path escapes allowed roots ({shown})"


def fs_read(path: str, roots=None, out_cap: int = 6000) -> str:
    """Read a text file under roots."""
    roots = list(roots) if roots else default_roots()
    full = _resolve(path or "", roots)
    if not full:
        return _deny(roots)
    try:
        with open(full, "r", errors="replace") as f:
            data = f.read(out_cap + 1)
    except FileNotFoundError:
        return f"error: no such file {(path or '').strip()}"
    except Exception as exc:
        return f"error: {exc}"
    if len(data) > out_cap:
        data = data[:out_cap] + "\n…[truncated]"
    return data or "(empty file)"


def fs_write(path: str, text: str, roots=None) -> str:
    """Write text to a file under roots (parents created)."""
    roots = list(roots) if roots else default_roots()
    full = _resolve(path or "", roots)
    if not full or not (path or "").strip():
        return _deny(roots)
    try:
        os.makedirs(os.path.dirname(full) or roots[0], exist_ok=True)
        with open(full, "w") as f:
            f.write(text)
    except Exception as exc:
        return f"error: {exc}"
    return f"wrote {len(text)} chars to {(path or '').strip()}"


def fs_edit(path: str, anchor: str, text: str, roots=None) -> str:
    """Anchored patch: replace ``anchor`` verbatim with ``text``."""
    from src.agent.terminal import _edit_diff

    roots = list(roots) if roots else default_roots()
    full = _resolve(path or "", roots)
    if not full:
        return _deny(roots)
    try:
        with open(full, "r", errors="replace") as f:
            src = f.read()
    except FileNotFoundError:
        return f"error: no such file {(path or '').strip()}"
    except Exception as exc:
        return f"error: {exc}"
    if anchor not in src:
        return "error: anchor not found (must copy verbatim from the file)"
    if src.count(anchor) > 1:
        return "error: anchor not unique (appears %d times)" % src.count(anchor)
    diff = _edit_diff(src, anchor, text)
    try:
        with open(full, "w") as f:
            f.write(src.replace(anchor, text, 1))
    except Exception as exc:
        return f"error: {exc}"
    return "patched %s with:\n%s" % ((path or "").strip(), diff)


def fs_list(path: str = ".", roots=None) -> str:
    """List directory entries under roots."""
    roots = list(roots) if roots else default_roots()
    full = _resolve(path or ".", roots)
    if not full:
        return _deny(roots)
    try:
        names = sorted(os.listdir(full))
    except Exception as exc:
        return f"error: {exc}"
    return "\n".join(names[:200]) or "(empty)"


def fs_grep(pattern: str, path: str = ".", roots=None, rel_base: str | None = None) -> str:
    """Grep a regex across files under roots (skips .git/__pycache__).

    Hits display as ``rel:line: text`` relative to ``rel_base`` (defaults
    to the searched dir itself; the turn runner passes its cwd so hits
    read relative to the working tree).
    """
    roots = list(roots) if roots else default_roots()
    full = _resolve(path or ".", roots)
    if not full:
        return _deny(roots)
    try:
        pat = re.compile(pattern)
    except Exception as exc:
        return f"error: bad pattern: {exc}"
    # anchor display base: the resolved dir itself when it's a dir,
    # else its parent (mirrors the turn-cwd display form path:line:)
    walk = full if os.path.isdir(full) else os.path.dirname(full)
    base = rel_base or walk
    hits = []
    for root, _, files in os.walk(walk):
        rel_root = os.path.relpath(root, base)
        if rel_root.startswith(".git") or "__pycache__" in rel_root:
            continue
        for name in files:
            f = os.path.join(root, name)
            try:
                txt = open(f, "r", errors="replace").read()
            except Exception:
                continue
            for lno, line in enumerate(txt.splitlines(), 1):
                if pat.search(line):
                    hits.append(f"{os.path.relpath(f, base)}:{lno}: {line.strip()[:200]}")
                    if len(hits) >= 80:
                        break
            if len(hits) >= 80:
                break
        if len(hits) >= 80:
            break
    return "\n".join(hits) or "(no matches)"


# -- docker exec (single tool, timeout, no native shell) ----------------

_BOX = {"sb": None, "tmp": None}
_BOX_LOCK = threading.Lock()
def _box():
    """Process-wide lazy container (empty tmp mount, nothing host-visible)."""
    with _BOX_LOCK:
        if _BOX["sb"] is None:
            tmp = tempfile.mkdtemp(prefix="mbox-")
            _BOX["tmp"] = tmp
            _BOX["sb"] = DockerSandbox(host_cwd=tmp, pull=True)
            _BOX["sb"].ensure_running()
        return _BOX["sb"]


def docker_exec(command: str, timeout: int = 30) -> str:
    """Run one command inside the shared container. Never raises.

    One-shot (no persistent shell, no jobs): each call is an isolated
    ``bash -c`` with a container-side kill on timeout. The container sees
    only an empty workdir — use ``fetch``/fs tools to move data.
    """
    from src.sandbox.docker import DockerSandboxError

    cmd = str(command or "").strip()
    if not cmd:
        return "error: empty command"
    try:
        t = max(1, int(timeout))
    except Exception:
        t = 30
    try:
        sb = _box()
    except Exception as exc:
        # DockerSandboxError or anything else: never raise, degrade to text
        return f"error: docker unavailable ({exc})"
    try:
        res = sb.execute(cmd, timeout=t)
    except Exception as exc:
        return f"error: {exc}"
    out = (res.output or "").strip() or "(no output)"
    tail = f"\n…[truncated]" if getattr(res, "truncated", False) else ""
    return f"rc={res.exit_code}\n{out[:6000]}{tail}"


def python_exec(code: str, timeout: int = 30) -> str:
    """Run Python code inside the shared container. Never raises.

    Same isolation as :func:`docker_exec` (no shell, no host files): the
    source is base64-piped to ``python3 -`` so arbitrary payloads need no
    shell quoting. One-shot per call with a container-side kill on timeout.
    """
    import base64

    src = str(code or "")
    if not src.strip():
        return "error: empty code"
    try:
        t = max(1, int(timeout))
    except Exception:
        t = 30
    try:
        sb = _box()
    except Exception as exc:
        return f"error: docker unavailable ({exc})"
    b64 = base64.b64encode(src.encode("utf-8")).decode("ascii")
    cmd = f"printf %s {b64} | base64 -d | python3 -"
    try:
        res = sb.execute(cmd, timeout=t)
    except Exception as exc:
        return f"error: {exc}"
    out = (res.output or "").strip() or "(no output)"
    tail = f"\n…[truncated]" if getattr(res, "truncated", False) else ""
    return f"rc={res.exit_code}\n{out[:6000]}{tail}"


# -- offline-friendly web: fetch + searxng -------------------------------

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
        text = raw.decode("utf-8", errors="replace") if "html" in ctype else raw.decode("utf-8", errors="replace")
        if "html" in ctype:
            text = _strip_html(text)
        else:
            text = re.sub(r"\s+", " ", text).strip()
    except Exception as exc:
        return f"error: decode failed ({exc})"
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text or "(empty page)"


def searxng_url() -> str:
    """Configured SearXNG base URL (env ``SEARXNG_URL`` wins)."""
    return (os.environ.get("SEARXNG_URL", "") or "").strip() or "http://127.0.0.1:8888"


def searxng_search(query: str, count: int = 5, base_url: str | None = None,
                   timeout: int = 15) -> str:
    """Web search via SearXNG JSON API. Never raises; degrades gracefully."""
    import httpx

    q = str(query or "").strip()
    if not q:
        return "error: empty query"
    try:
        n = max(1, min(int(count), 10))
    except Exception:
        n = 5
    base = (base_url or "").strip() or searxng_url()
    try:
        r = httpx.get(f"{base.rstrip('/')}/search",
                      params={"q": q, "format": "json", "language": "en"},
                      timeout=timeout,
                      headers={"User-Agent": "voice-pipeline/searxng"})
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:
        return (f"error: searxng unreachable at {base} ({exc}). "
                f"Set SEARXNG_URL to a running instance."[:400])
    try:
        results = payload.get("results", []) or []
    except Exception:
        return "error: bad searxng response"
    lines = []
    for item in results[:n]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "").strip()
        url = str(item.get("url", "") or "").strip()
        snippet = re.sub(r"\s+", " ", str(item.get("content", "") or "")).strip()[:300]
        if title or url:
            lines.append(f"- {title}\n  {url}\n  {snippet}".rstrip())
    return "\n".join(lines) or "(no results)"


DEFAULT_ROOTS = ("~/Documents", "~/Projects")
"""Default MCP filesystem scope (expanded at call time)."""
