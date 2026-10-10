"""Permission policy: one verdict per tool call, both agent paths.

Contract: :class:`PolicyGate` is pure (no I/O, no network) — ``decide``
maps a post-mapping tool name + args to ``(deny|confirm|allow, reason)``.
Terminal built-ins reuse the proven :mod:`src.tools.terminal` policy;
the MCP names get their own rules below. Unknown tools fail OPEN (the
tool node reports them, the model recovers). Enforcement lives in
:mod:`src.agent.middleware.policy`, which short-circuits denies and
routes confirms through the harness confirmer (WS gate on server,
auto-allow + announce on console).

Rule shape: read-only and reversible work runs free; anything mutating,
outward-facing, or session-driving needs a yes; system-breakers and
self-destruct paths never run.
"""

from src.tools.terminal import TerminalAction, check_policy

__all__ = [
    "PolicyGate",
    "DENY_PATHS",
]

DENY_PATHS = ("", "/", "~")
"""delete_file refuses these outright (impl refuses too, belt and braces)."""

_DENY_KEYS = ("ctrl+alt+del", "ctrl+alt+delete", "sysrq", "alt+sysrq")
"""press_key combos that can lock, reboot, or break out. Never run."""


class PolicyGate:
    """One verdict per tool call. Stateless, pure, unit-testable."""

    def decide(self, tool_name: str, args: dict | None) -> tuple[str, str]:
        """Map a tool call to (deny|confirm|allow, reason).

        ``tool_name`` is the post-mapping execution name (execute,
        read_file, screenshot, ...). Total by construction (plain
        str/dict reads, isinstance guard); unknown names allow.
        ..
        """
        if not isinstance(args, dict):
            args = {}
        return self._decide(str(tool_name or "").lower(), args)

    def _terminal(self, op: str, args: dict) -> tuple[str, str] | None:
        """Terminal-family verdict via the proven policy, else None.

        Covers both the legacy op names (exec/read/...) and the
        post-mapping built-in names (execute/read_file/...).
        ..
        """
        get = lambda *ks: next(
            (args[k] for k in ks if str(args.get(k, "") or "").strip()), "")
        if op in ("exec", "execute"):
            action = TerminalAction(op="exec", command=get("command"))
        elif op in ("exec_bg",):
            action = TerminalAction(op="exec_bg", command=get("command"))
        elif op in ("read", "read_file"):
            action = TerminalAction(
                op="read", path=get("path", "file_path"))
        elif op in ("write", "write_file"):
            action = TerminalAction(
                op="write", path=get("path", "file_path"),
                text=get("content", "text"))
        elif op in ("edit", "edit_file"):
            action = TerminalAction(
                op="edit", path=get("path", "file_path"),
                anchor=get("old_string", "anchor") or "x",
                text=get("new_string", "text"))
        elif op in ("list", "ls"):
            action = TerminalAction(op="list", path=get("path"))
        elif op in ("grep",):
            action = TerminalAction(
                op="grep", pattern=get("pattern"), path=get("path"))
        elif op in ("searxng", "web_search"):
            action = TerminalAction(op="searxng", pattern=get("pattern"))
        elif op in ("fetch",):
            action = TerminalAction(op="fetch", path=get("path", "url"))
        elif op in ("python_exec",):
            return "confirm", "arbitrary host code"
        else:
            return None
        return check_policy(action)

    def _mcp(self, name: str, args: dict) -> tuple[str, str] | None:
        """MCP-category verdicts, else None for non-MCP names.

        Read-only tools allow; mutating / outward-facing / session
        driving tools confirm; breaker paths and combos deny.
        ..
        """
        if name in ("screenshot", "click", "type_text", "scroll",
                    "get_active_window", "list_windows", "list_directory",
                    "search_files", "search_web", "extract_page",
                    "open_url", "browser_click", "browser_scroll"):
            return "allow", ""
        if name == "press_key":
            keys = "+".join(
                p.strip().lower()
                for p in str(args.get("key", "") or "").split("+")
                if p.strip())
            if keys in _DENY_KEYS:
                return "deny", f"breaker combo refused: {keys}"
            return "confirm", "drives the user session keys"
        if name in ("write_file", "edit_file", "move_file",
                    "download_file", "open_app"):
            return "confirm", "mutating/outward-facing tool"
        if name == "delete_file":
            p = str(args.get("path", "") or "").strip()
            if p in DENY_PATHS:
                return "deny", f"refusing to delete {p or 'empty path'}"
            return "confirm", "destructive tool"
        if name == "browser_type":
            if bool(args.get("submit", False)):
                return "confirm", "submits a web form"
            return "allow", ""
        return None

    def _decide(self, name: str, args: dict) -> tuple[str, str]:
        """Verdict dispatch: terminal rules, then MCP rules, else allow.

        Duck-typed (no imports of the middleware): terminal ops reuse
        check_policy so deny patterns stay in exactly one place.
        ..
        """
        hit = self._terminal(name, args)
        if hit is not None:
            return hit
        hit = self._mcp(name, args)
        if hit is not None:
            return hit
        return "allow", ""
