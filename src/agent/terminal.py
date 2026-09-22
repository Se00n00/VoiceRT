"""Terminal tools + execution for the unified agent (no harness here).

The turn loop lives in :class:`src.main.VoiceAgent`, which builds a
deepagents agent over :func:`build_terminal_tools` each turn. This module
keeps the pieces with their own unit tests:

- :func:`run_command` — one validated action, host or sandbox shell
- :func:`build_terminal_tools` — local LangChain tools over a TurnCtx
- policy gate + confirm plumbing (``_guarded``), diff previews, speak head
"""
import asyncio
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from src.tools.terminal import (
    TerminalAction,
    check_policy,
)

__all__ = [
    "TurnCtx",
    "build_terminal_tools",
    "run_command",
]


def _edit_diff(src: str, anchor: str, new: str, ctx: int = 3) -> str:
    """Unified diff preview for an anchored edit. Never raises."""
    try:
        import difflib
        a = (anchor.splitlines() or [""])[0][:80]
        b = (new.splitlines() or [""])[0][:80]
        diff = difflib.unified_diff(
            src.splitlines(), src.replace(anchor, new, 1).splitlines(),
            fromfile="before", tofile="after", n=ctx, lineterm="")
        out = "\n".join(list(diff)[:40])
        return out or f"- {a}\n+ {b}"
    except Exception:
        return f"- {anchor[:200]}\n+ {new[:200]}"


def _write_diff(path: str, old: str, new: str) -> str:
    try:
        import difflib
        diff = difflib.unified_diff(
            old.splitlines(), new.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", n=3, lineterm="")
        out = "\n".join(list(diff)[:50])
        return out or "(new file)"
    except Exception:
        return "(diff unavailable)"


def run_command(action: TerminalAction, cwd: str = ".",
                 timeout_s: float = 30.0, out_cap: int = 6000,
                 shell=None, jobs=None) -> str:
    """Execute one validated action. No weights. Never raises.

    If a :class:`PersistentShell` is supplied, exec/exec_bg/poll use it
    (persistent CWD/env; bg jobs share the turn). Otherwise a one-shot
    subprocess is used (tests).
    """
    jm = jobs if jobs is not None else getattr(shell, "jobs", None) if shell else None
    try:
        if action.op == "exec":
            if shell is not None:
                rc, out = shell.exec(action.command, timeout_s=timeout_s)
                if len(out) > out_cap:
                    out = out[:out_cap] + f"\n…[truncated {len(out)} chars]"
                return f"rc={rc}\n{out.strip() or '(no output)'}"
            p = subprocess.run(
                action.command, shell=True, cwd=cwd or ".",
                capture_output=True, text=True, timeout=timeout_s)
            out = (p.stdout or "") + (p.stderr or "")
            if len(out) > out_cap:
                out = out[:out_cap] + f"\n…[truncated {len(out)} chars]"
            return f"rc={p.returncode}\n{out.strip() or '(no output)'}"
        if action.op == "python_exec":
            # Python in an isolated container (canonical impl; no confirm:
            # read-only-equivalent compute, policy allows it outright)
            from src.agent.mcp.tools import python_exec as run_python

            return run_python(action.code.strip())
        if action.op == "fetch":
            # web page as text (canonical impl; policy already allowed it)
            from src.agent.mcp.tools import fetch as fetch_url

            return fetch_url(action.path.strip())
        if action.op == "searxng":
            # web search via local SearXNG (canonical impl)
            from src.agent.mcp.tools import searxng_search

            return searxng_search(action.pattern.strip())
        if action.op == "exec_bg":
            if jm is None:
                return "error: background jobs unavailable (no shell context)"
            jid = jm.start(action.command)
            return f"started {jid}: {action.command[:200]}"
        if action.op == "poll":
            if jm is None:
                return "error: background jobs unavailable (no shell context)"
            r = jm.poll(action.command)
            prefix = "running" if r.get("running") else f"done rc={r.get('rc')}"
            return f"{prefix} {r.get('job','')}\n{(r.get('output','') or '').strip() or '(no output)'}"
        if action.op == "edit":
            # canonical impl (roots = turn cwd); strings verified by suite
            from src.agent.mcp.tools import fs_edit

            base = os.path.abspath(cwd or ".")
            return fs_edit(action.path.strip(), action.anchor, action.text,
                           roots=[base])
        if action.op == "grep":
            from src.agent.mcp.tools import fs_grep

            base = os.path.abspath(cwd or ".")
            return fs_grep(action.pattern, action.path.strip() or ".",
                           roots=[base], rel_base=base)
        if action.op == "list":
            from src.agent.mcp.tools import fs_list

            base = os.path.abspath(cwd or ".")
            return fs_list(action.path.strip() or ".", roots=[base])
        if action.op == "read":
            from src.agent.mcp.tools import fs_read

            base = os.path.abspath(cwd or ".")
            return fs_read(action.path.strip(), roots=[base], out_cap=out_cap)
        if action.op == "write":
            from src.agent.mcp.tools import fs_write

            base = os.path.abspath(cwd or ".")
            return fs_write(action.path.strip(), action.text, roots=[base])
        return f"unknown op: {action.op}"
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s:.0f}s"
    except Exception as exc:  # never break the agent loop
        return f"error: {exc}"


def _speak_head(reply: str, limit: int = 280) -> str:
    """Head of a reply for voicing; full text stays on screen/in memory."""
    t = " ".join(str(reply or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    dot = cut.rfind(". ")
    head = (cut[:dot + 1] if dot > 120 else cut).strip()
    return f"{head} … full output is on screen."


@dataclass
class TurnCtx:
    """Per-turn execution context shared by the local tools."""

    shell: Any = None
    cwd: str = "."
    approvals: set = field(default_factory=set)
    confirm_fn: Any = None
    loop: Any = None
    cfg: Any = None
    sink: list = field(default_factory=list)  # (kind, data) for confirm/deny UI events
    sandbox: Any = None  # DockerSandbox for this turn, if enabled


def _approval_key(action: TerminalAction) -> str:
    cmd = getattr(action, "command", "") or ""
    if cmd.strip():
        return cmd.strip().split()[0][:40]
    return getattr(action, "op", "")[:40]


def _resolve_confirm(ctx: TurnCtx, action: TerminalAction):
    """Ask confirm_fn (sync or async). Returns 'always' | True | False."""
    fn = ctx.confirm_fn
    if fn is None:
        return False
    try:
        if asyncio.iscoroutinefunction(fn):
            fut = asyncio.run_coroutine_threadsafe(fn(action), ctx.loop)
            res = fut.result(timeout=300)
        else:
            res = fn(action)
    except Exception:
        return False
    if isinstance(res, str) and res.lower() in ("always", "a", "allowlist"):
        return "always"
    return bool(res)


def _file_preview(action: TerminalAction, cwd: str) -> str:
    """Diff preview for mutating file ops shown in the confirm event."""
    try:
        base = os.path.abspath(cwd or ".")
        full = os.path.abspath(os.path.join(base, action.path.strip())) if action.path.strip() else ""
        if not full or not full.startswith(base) or not os.path.isfile(full):
            return "(new file)" if action.op == "write" else ""
        with open(full, "r", errors="replace") as f:
            old = f.read()
        if action.op == "edit" and action.anchor in old:
            return _edit_diff(old, action.anchor, action.text)[:1200]
        if action.op == "write":
            return _write_diff(action.path.strip(), old, action.text)[:1200]
    except Exception:
        pass
    return ""


def _guarded(ctx: TurnCtx, action: TerminalAction) -> str | None:
    """Policy gate: None = allowed, else the tool-result string to return."""
    cfg = ctx.cfg
    timeout_s = getattr(cfg, "timeout_s", 30.0) if cfg else 30.0
    out_cap = getattr(cfg, "out_cap", 6000) if cfg else 6000
    # approvals override: pre-approved prefixes skip confirm
    verdict, reason = check_policy(action)
    if verdict == "confirm" and ctx.approvals:
        if _approval_key(action) in ctx.approvals or action.op in ctx.approvals:
            verdict, reason = "allow", "pre-approved"
    if verdict == "deny":
        ctx.sink.append(("deny", {"action": action.as_dict(), "reason": reason}))
        return f"Blocked: {reason}."
    if verdict == "confirm":
        preview = _file_preview(action, ctx.cwd) if action.op in ("edit", "write") else ""
        ctx.sink.append(("confirm", {"action": action.as_dict(),
                                     "reason": reason, "preview": preview}))
        ok = _resolve_confirm(ctx, action)
        if ok == "always":
            ctx.approvals.add(_approval_key(action))
        if not ok:
            ctx.sink.append(("deny", {"action": action.as_dict(),
                                      "reason": "denied by user"}))
            return "Cancelled — I did not run that."
    shell = ctx.shell
    cwd = getattr(shell, "cwd", ctx.cwd) if shell is not None else ctx.cwd
    out = run_command(action, cwd=cwd, timeout_s=timeout_s,
                      out_cap=out_cap, shell=shell)
    # track cwd if the persistent shell moved
    try:
        if shell is not None and hasattr(shell, "cwd"):
            ctx.cwd = shell.cwd
    except Exception:
        pass
    # cap context burn; full detail already went to the observation event
    if len(out) > 2000:
        lines = out.splitlines()
        if len(lines) > 60:
            out = "\n".join(lines[:20]) + f"\n…[{len(lines)-40} lines omitted]…\n" + "\n".join(lines[-20:])
            out = out[:2000]
        else:
            out = out[:2000] + "\n…[truncated]"
    return out


def build_terminal_tools(ctx: TurnCtx) -> list:
    """11 local LangChain tools closing over this turn's shell + policy."""
    from langchain_core.tools import StructuredTool

    def _exec(command: str) -> str:
        """Run a bash shell command (persistent CWD/env). Prefer read-only commands."""
        return _guarded(ctx, TerminalAction(op="exec", command=command)) or ""

    def _exec_bg(command: str) -> str:
        """Start a long-running shell command in the background. Poll it later."""
        return _guarded(ctx, TerminalAction(op="exec_bg", command=command)) or ""

    def _poll(command: str) -> str:
        """Poll a background job started by exec_bg. Job id like job-1."""
        return _guarded(ctx, TerminalAction(op="poll", command=command)) or ""

    def _read(path: str) -> str:
        """Read a text file below the working directory."""
        return _guarded(ctx, TerminalAction(op="read", path=path)) or ""

    def _write(path: str, text: str) -> str:
        """Write text to a file below the working directory."""
        return _guarded(ctx, TerminalAction(op="write", path=path, text=text)) or ""

    def _edit(path: str, anchor: str, text: str) -> str:
        """Anchored patch: replace `anchor` verbatim in the file with `text`."""
        return _guarded(ctx, TerminalAction(op="edit", path=path, anchor=anchor, text=text)) or ""

    def _grep(pattern: str, path: str = ".") -> str:
        """Grep a pattern across files below the working directory."""
        return _guarded(ctx, TerminalAction(op="grep", pattern=pattern, path=path)) or ""

    def _list(path: str = ".") -> str:
        """List directory entries below the working directory."""
        return _guarded(ctx, TerminalAction(op="list", path=path)) or ""

    def _python_exec(code: str) -> str:
        """Run Python code inside an isolated container (no shell, no host files)."""
        return _guarded(ctx, TerminalAction(op="python_exec", code=code)) or ""

    def _fetch(path: str) -> str:
        """Fetch a web page (http/https) and return its text."""
        return _guarded(ctx, TerminalAction(op="fetch", path=path)) or ""

    def _searxng(pattern: str) -> str:
        """Web search via the local SearXNG instance. Returns titles, URLs, snippets."""
        return _guarded(ctx, TerminalAction(op="searxng", pattern=pattern)) or ""

    return [
        StructuredTool.from_function(_exec, name="exec"),
        StructuredTool.from_function(_exec_bg, name="exec_bg"),
        StructuredTool.from_function(_poll, name="poll"),
        StructuredTool.from_function(_read, name="read"),
        StructuredTool.from_function(_write, name="write"),
        StructuredTool.from_function(_edit, name="edit"),
        StructuredTool.from_function(_grep, name="grep"),
        StructuredTool.from_function(_list, name="list"),
        StructuredTool.from_function(_python_exec, name="python_exec"),
        StructuredTool.from_function(_fetch, name="fetch"),
        StructuredTool.from_function(_searxng, name="searxng"),
    ]


