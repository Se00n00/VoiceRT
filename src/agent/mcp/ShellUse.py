"""
Canonical ShellUse tool implementations: shell, python, files.

These plain functions are THE ShellUse tool implementations —
:mod:`src.agent.mcp.mcp` registers them over MCP via ``app.add_tool``.
All functions are sync and total (never raise — errors become strings).

contract, dependencies, who calls it: unified bash (foreground,
background, poll), host python_exec, content grep, and file
read/write/edit/list/glob/move/delete; depends on stdlib plus
:mod:`src.agent.shell` (persistent shell + jobs, stdlib only);
used by the MCP registry. Relative paths anchor at ``VOICE_WORKDIR``.
"""
import base64 as _base64
import os as _os
import pathlib as _pathlib
import re as _re
import shutil as _shutil
import subprocess as _sp
import sys as _sys

__all__ = [
    "bash",
    "python_exec",
    "read",
    "write",
    "edit",
    "list",
    "glob",
    "grep",
    "move",
    "delete",
]

_SHELL = None
_JOBS = None


def _workdir() -> str:
    """Harness workdir anchor for relative commands.

    ..
    """
    base = (_os.environ.get("VOICE_WORKDIR", "") or "").strip()
    if base and _os.path.isdir(base):
        return base
    return _os.getcwd()


def _resolve(path: str) -> str:
    """Expand user vars and absolutize. Empty stays empty (caller errors).

    Relative paths anchor at ``VOICE_WORKDIR`` when set (the harness
    sets it to the agent workdir, so model-relative paths land where
    the shell backend runs); else the process cwd.
    """
    p = str(path or "").strip()
    if not p:
        return ""
    p = _os.path.expanduser(_os.path.expandvars(p))
    if not _os.path.isabs(p):
        base = (_os.environ.get("VOICE_WORKDIR", "") or "").strip()
        if base:
            p = _os.path.join(base, p)
    return _os.path.abspath(p)


def _to_int(value, default: int) -> int:
    """int clamp input: parsed digits or the default (plain ifs).

    ..
    """
    s = str(value).strip()
    return int(s) if s.lstrip("-").isdigit() else default


def _shell():
    """Module singleton PersistentShell anchored at the workdir.

    ..
    """
    global _SHELL
    try:
        if _SHELL is None:
            from src.agent.shell import PersistentShell
            _SHELL = PersistentShell(cwd=_workdir(), timeout_s=30.0)
        return _SHELL
    except Exception:
        return None


def _jobs():
    """Module singleton JobManager for background bash jobs.

    ..
    """
    global _JOBS
    try:
        if _JOBS is None:
            from src.agent.shell import JobManager
            _JOBS = JobManager(cwd=_workdir())
        return _JOBS
    except Exception:
        return None


def bash(command: str = "", background: bool = False,
         job_id: str = "") -> str:
    """Run, start, or poll a bash command. Never raises.

    job_id polls; background starts a job; else foreground exec.
    ..
    """
    jid = str(job_id or "").strip()
    if jid:
        mgr = _jobs()
        if mgr is None:
            return "error: job backend unavailable"
        try:
            out = mgr.poll(jid)
        except Exception as exc:
            return f"error: poll failed ({exc})"[:300]
        state = "running" if out.get("running") else "done"
        rc = out.get("rc")
        text = str(out.get("output", "") or "").strip() or "(no output)"
        return f"{jid} [{state} rc={rc}]\n{text[:6000]}"
    cmd = str(command or "").strip()
    if not cmd:
        return "error: empty command (need command or job_id)"
    if bool(background):
        mgr = _jobs()
        if mgr is None:
            return "error: job backend unavailable"
        try:
            new_id = mgr.start(cmd)
        except Exception as exc:
            return f"error: start failed ({exc})"[:300]
        return f"started {new_id}: {cmd[:200]}"
    sh = _shell()
    if sh is not None:
        try:
            rc, out = sh.exec(cmd)
            return f"rc={rc}\n{(out or '').strip() or '(no output)'}"[:8000]
        except Exception:
            pass
    try:
        p = _sp.run(cmd, shell=True, executable="/bin/bash",
                    cwd=_workdir(), capture_output=True,
                    text=True, timeout=30)
    except _sp.TimeoutExpired:
        return "timeout after 30s"
    except Exception as exc:
        return f"error: {exc}"[:300]
    out = ((p.stdout or "") + (p.stderr or "")).strip() or "(no output)"
    return f"rc={p.returncode}\n{out[:6000]}"


def _python_exec_box(container: str, src: str, timeout: int) -> str:
    """Run code in the turn container via ``docker exec``. Never raises.

    The container shares the turn workdir at ``/work`` and always has
    ``python3`` (image contract), so the same base64 bootstrap runs
    unquoted on either side of the wall.
    ..
    """
    b64 = _base64.b64encode(src.encode("utf-8")).decode("ascii")
    argv = ["docker", "exec", "-i", container, "python3", "-c",
            ("import base64,sys; exec(compile(base64.b64decode(sys.argv[1])"
             ".decode('utf-8'), '<agent>', 'exec'))"), b64]
    try:
        p = _sp.run(argv, capture_output=True, text=True,
                    timeout=timeout + 15,
                    stdin=_sp.DEVNULL)
    except FileNotFoundError:
        return "error: docker CLI not found ('docker'). Install Docker first."
    except _sp.TimeoutExpired:
        return f"timeout after {timeout}s"
    except Exception as exc:
        return f"error: {exc}"
    if p.returncode != 0 and "No such container" in (p.stderr or ""):
        return ("error: sandbox container is gone "
                f"({container}); retry the turn.")
    out = ((p.stdout or "") + (p.stderr or "")).strip() or "(no output)"
    return f"rc={p.returncode}\n{out[:6000]}"


def python_exec(code: str, timeout: int = 30) -> str:
    """Run Python code, in the turn container when sandboxed. Never raises.

    Same trust level as shell execution — for local use only. The source
    is base64-piped to ``python3 -`` so arbitrary payloads need no shell
    quoting. One-shot per call with a process kill on timeout.
    ``VOICE_SANDBOX_CONTAINER`` (set per sandboxed turn) switches this
    from host ``sys.executable`` to ``docker exec`` in that container.
    ..
    """
    src = str(code or "")
    if not src.strip():
        return "error: empty code"
    try:
        t = max(1, int(timeout))
    except Exception:
        t = 30
    box = (_os.environ.get("VOICE_SANDBOX_CONTAINER") or "").strip()
    if box:
        return _python_exec_box(box, src, t)
    b64 = _base64.b64encode(src.encode("utf-8")).decode("ascii")
    cmd = [_sys.executable or "python3", "-c",
           ("import base64,sys; exec(compile(base64.b64decode(sys.argv[1])"
            ".decode('utf-8'), '<agent>', 'exec'))"), b64]
    try:
        p = _sp.run(cmd, capture_output=True, text=True, timeout=t)
    except _sp.TimeoutExpired:
        return f"timeout after {t}s"
    except Exception as exc:
        return f"error: {exc}"
    out = ((p.stdout or "") + (p.stderr or "")).strip() or "(no output)"
    return f"rc={p.returncode}\n{out[:6000]}"


def _grep_rg(pattern: str, root: str, limit: int) -> str | None:
    """ripgrep fast path; None when rg missing or failed.

    ..
    """
    rg = _shutil.which("rg")
    if not rg:
        return None
    try:
        p = _sp.run([rg, "-n", "--no-heading", "-m", str(limit),
                     "--", pattern, root],
                    capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if p.returncode not in (0, 1):
        return None
    lines = (p.stdout or "").strip().splitlines()
    if not lines:
        return "(no matches)"
    return "\n".join(lines[:limit])


def grep(pattern: str, path: str = ".") -> str:
    """Search file contents for a pattern. Never raises.

    ripgrep when installed, stdlib walk otherwise; capped output.
    ..
    """
    pat = str(pattern or "").strip()
    if not pat:
        return "error: empty pattern"
    root = _resolve(path)
    if not _os.path.lexists(root):
        return f"error: not found: {root}"
    try:
        rx = _re.compile(pat)
    except _re.error as exc:
        return f"error: bad regex ({exc})"[:200]
    hit = _grep_rg(pat, root, 100)
    if hit is not None:
        return hit[:8000]
    out = []
    try:
        if _os.path.isfile(root):
            files = [root]
        else:
            files = []
            for dirpath, _, filenames in _os.walk(root):
                if "/.git/" in dirpath or "/.venv/" in dirpath:
                    continue
                for fn in filenames:
                    files.append(_os.path.join(dirpath, fn))
                    if len(files) > 2000:
                        break
        for fp in files:
            try:
                with open(fp, "rb") as f:
                    head = f.read(8192)
                if b"\x00" in head:
                    continue
                with open(fp, "r", encoding="utf-8",
                           errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            rel = _os.path.relpath(fp, root)
                            out.append(f"{rel}:{i}:{line.strip()}"[:300])
                            if len(out) >= 100:
                                return "\n".join(out)
            except Exception:
                continue
    except Exception as exc:
        return f"error: grep failed ({exc})"[:300]
    return "\n".join(out)[:8000] if out else "(no matches)"


def read(path: str, offset: int = 1, limit: int = 100,
         max_chars: int = 12000) -> str:
    """Read text lines (1-based ``offset``, at most ``limit``). Never raises.

    Refuses binary files (NUL byte in the head). Whole output capped at
    ``max_chars``.
    """
    try:
        p = _resolve(path)
        if not p:
            return "error: empty path"
        if not _os.path.lexists(p):
            return f"error: not found: {p}"
        off = max(1, _to_int(offset, 1))
        lim = max(1, min(_to_int(limit, 100), 2000))
        cap = max(256, min(_to_int(max_chars, 12000), 100_000))
        with open(p, "rb") as f:
            if b"\x00" in f.read(8192):
                return f"error: binary file (not text): {p}"
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        if off > total:
            return f"{p} [{total} lines — offset {off} past end]"
        chunk = "".join(lines[off - 1:off - 1 + lim])
        if len(chunk) > cap:
            chunk = chunk[:cap] + "\n…[truncated]"
        return (f"{p} [lines {off}-{min(off + lim - 1, total)} of {total}]\n"
                + (chunk or "(empty)"))
    except Exception as exc:
        return f"error: cannot read {path} ({exc})"[:300]


def write(path: str, content: str,
          make_parents: bool = True) -> str:
    """Write (create or overwrite) a text file. Never raises."""
    try:
        p = _resolve(path)
        if not p:
            return "error: empty path"
        if _os.path.isdir(p):
            return f"error: is a directory: {p}"
        if make_parents:
            parent = _os.path.dirname(p)
            if parent:
                _os.makedirs(parent, exist_ok=True)
        data = str(content or "")
        with open(p, "w", encoding="utf-8") as f:
            f.write(data)
        return f"wrote {len(data.encode('utf-8'))} bytes to {p}"
    except Exception as exc:
        return f"error: cannot write {path} ({exc})"[:300]


def edit(path: str, old_string: str, new_string: str,
         replace_all: bool = False) -> str:
    """Exact string replacement in a text file. Never raises.

    ``old_string`` missing -> error; matching more than once without
    ``replace_all`` -> error (narrow it down instead of guessing).
    """
    try:
        p = _resolve(path)
        if not p:
            return "error: empty path"
        if not _os.path.lexists(p):
            return f"error: not found: {p}"
        old = str(old_string or "")
        if not old:
            return "error: empty old_string (nothing to match)"
        new = str(new_string or "")
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        n = text.count(old)
        if n == 0:
            return "error: old_string not found in file"
        if n > 1 and not replace_all:
            return (f"error: old_string matches {n} times "
                    f"(narrow it, or set replace_all=true)")
        text = text.replace(old, new) if replace_all \
            else text.replace(old, new, 1)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        return (f"replaced {n} occurrence(s) in {p}"
                if replace_all else f"replaced 1 occurrence in {p}")
    except Exception as exc:
        return f"error: cannot edit {path} ({exc})"[:300]


def list(path: str = ".", show_hidden: bool = False,
         limit: int = 200) -> str:
    """List a directory (dirs first, trailing ``/``). Never raises."""
    try:
        d = _resolve(path)
        if not d:
            return "error: empty path"
        if not _os.path.lexists(d):
            return f"error: not found: {d}"
        if not _os.path.isdir(d):
            return f"error: not a directory: {d}"
        lim = max(1, min(_to_int(limit, 200), 1000))
        names = _os.listdir(d)
        if not bool(show_hidden):
            names = [n for n in names if not n.startswith(".")]
        dirs = sorted(n for n in names
                      if _os.path.isdir(_os.path.join(d, n)))
        files = sorted(n for n in names
                       if not _os.path.isdir(_os.path.join(d, n)))
        rows = [f"{n}/" for n in dirs]
        for n in files:
            sz = _os.path.getsize(_os.path.join(d, n))
            rows.append(f"{n} ({sz}B)")
        total = len(rows)
        shown = rows[:lim]
        tail = f"\n…[{total - lim} more]" if total > lim else ""
        return f"{d} [{total} entries]\n" + "\n".join(shown) + tail
    except Exception as exc:
        return f"error: cannot list {path} ({exc})"[:300]


def glob(pattern: str, path: str = ".",
         limit: int = 50) -> str:
    """Glob-search for file names under ``path`` (recursive). Never raises.

    ``pattern`` without a wildcard matches substrings
    (``report`` -> ``*report*``). Returns paths relative to ``path``.
    """
    try:
        base = _resolve(path)
        if not base:
            return "error: empty path"
        if not _os.path.isdir(base):
            return f"error: not a directory: {base}"
        lim = max(1, min(_to_int(limit, 50), 500))
        pat = str(pattern or "*").strip() or "*"
        if not any(c in pat for c in "*?["):
            pat = f"*{pat}*"
        hits = sorted(str(p.relative_to(base))
                      for p in _pathlib.Path(base).rglob(pat))[:lim]
        if not hits:
            return "(no matches)"
        tail = f"\n…[capped at {lim}]" if len(hits) == lim else ""
        return "\n".join(hits) + tail
    except Exception as exc:
        return f"error: search failed ({exc})"[:300]


def move(src: str, dst: str) -> str:
    """Move/rename a file or directory (parents created). Never raises."""
    try:
        s = _resolve(src)
        d = _resolve(dst)
        if not s or not d:
            return "error: empty src or dst"
        if not _os.path.lexists(s):
            return f"error: not found: {s}"
        if s == d:
            return "error: src and dst are the same"
        parent = _os.path.dirname(d)
        if parent:
            _os.makedirs(parent, exist_ok=True)
        _shutil.move(s, d)
        return f"moved {s} -> {d}"
    except Exception as exc:
        return f"error: cannot move {src} -> {dst} ({exc})"[:300]


def delete(path: str, recursive: bool = False) -> str:
    """Delete a file, symlink, or directory. Never raises.

    Directories need ``recursive=true`` unless empty. Refuses ``/``
    and the home directory itself.
    """
    try:
        p = _resolve(path)
        if not p:
            return "error: empty path"
        home = _os.path.expanduser("~")
        if p in ("/", home):
            return f"error: refusing to delete {p}"
        if not _os.path.lexists(p):
            return f"error: not found: {p}"
        if _os.path.islink(p) or _os.path.isfile(p):
            _os.unlink(p)
            return f"deleted file {p}"
        if _os.path.isdir(p):
            if not _os.listdir(p):
                _os.rmdir(p)
                return f"deleted empty directory {p}"
            if not recursive:
                return (f"error: directory not empty: {p} "
                        f"(use recursive=true)")
            _shutil.rmtree(p)
            return f"deleted directory tree {p}"
        return f"error: not a file or directory: {p}"
    except Exception as exc:
        return f"error: cannot delete {path} ({exc})"[:300]
