"""Canonical filesystem implementations (host paths, stdlib only).

Single-definition rule (same as :mod:`src.agent.mcp.tools`): these plain
functions are THE implementations — :mod:`src.agent.mcp.server` exposes
them over MCP as thin wrappers. All functions are sync and total (never
raise — errors become strings).

Same trust level as shell execution: for local use only. Reads are
capped (chars/lines), listings and searches are capped (entries), and
:func:`delete_file` refuses ``/`` and the home directory itself.
"""
import os as _os
import pathlib as _pathlib
import shutil as _shutil

__all__ = [
    "list_directory",
    "search_files",
    "read_file",
    "write_file",
    "edit_file",
    "move_file",
    "delete_file",
]


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


def list_directory(path: str = ".", show_hidden: bool = False,
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


def search_files(root: str = ".", pattern: str = "*",
                 limit: int = 50) -> str:
    """Glob-search under ``root`` (recursive). Never raises.

    ``pattern`` without a wildcard matches substrings
    (``report`` -> ``*report*``). Returns paths relative to ``root``.
    """
    try:
        base = _resolve(root)
        if not base:
            return "error: empty root"
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


def read_file(path: str, offset: int = 1, limit: int = 100,
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


def write_file(path: str, content: str,
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


def edit_file(path: str, old_string: str, new_string: str,
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


def move_file(src: str, dst: str) -> str:
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


def delete_file(path: str, recursive: bool = False) -> str:
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
