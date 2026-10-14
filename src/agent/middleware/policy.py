"""Permission layer: one verdict per tool call, enforced on the loop.

Pluggable hook for the deepagents middleware list (both agent paths).
:class:`PolicyMiddleware` asks :class:`PolicyGate` once per tool call:
``deny`` returns a steering observation WITHOUT running the tool;
``confirm`` awaits the harness confirmer (WS gate on server,
auto-allow + announce on console); ``allow`` runs untouched. Sync and
async paths both implemented (the harness invokes async; sync stays
for unit calls). Never raises.

Rule shape: read-only and reversible work runs free; anything mutating,
outward-facing, or session-driving needs a yes; system-breakers and
self-destruct paths never run. All deny/confirm patterns live in this
one file (no shared policy module).
"""
import inspect as _inspect
import re as _re

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

__all__ = ["PolicyMiddleware", "PolicyGate", "DENY_PATHS"]

DENY_PATHS = ("", "/", "~")
"""delete refuses these outright (impl refuses too, belt and braces)."""

_DENY_KEYS = ("ctrl+alt+del", "ctrl+alt+delete", "sysrq", "alt+sysrq")
"""press_key combos that can lock, reboot, or break out. Never run."""

# System-breakers: always denied, never even asked.
_DENY_RES = [
    r":\s*\(\s*\)\s*\{",          # fork bomb
    r"\brm\s+.*-[a-z]*r[a-z]*f\b.*\s+/\s*( |$)",  # rm -rf /
    r"\brm\s+-rf?\s+/\s*( |$)",
    r"\brm\s+-rf?\s+~/?\s*( |$)",
    r"\bmkfs(\.|s?\s)",
    r"\bdd\s+.*\bof=/dev/",
    r"\b(shutdown|poweroff|reboot|halt)\b",
    r":\s*>\s*/dev/sd",
    r"\bchmod\s+-R\s+777\s+/\s*( |$)",
    r"\bmv\s+.*\s+/\s*dev\b",
]
_DENY_RE = _re.compile("|".join(f"(?:{p})" for p in _DENY_RES))

# Mutating / outward-facing: allowed only with explicit user confirm.
_CONFIRM_RES = [
    r"^\s*sudo\b",
    r"\beval\b",
    r"\$\(",
    r"\b`",
    r"\b(process\s*substitution)\b",
    r"\brm\b",
    r"\bmv\b",
    r"\bdd\b",
    r"\b(chmod|chown)\b",
    r"\b(git\s+push\s+.*--force|git\s+reset\s+--hard)\b",
    r"(curl|wget).*\|\s*(sh|bash)",
    r"\bssh\b",
    r"\b(docker|podman)\b",
    r"\b(systemctl|service)\b",
    r"\b(pip|pip3|npm)\s+(install|uninstall)\b",
    r"\bapt(-get)?\b",
]
_CONFIRM_CMD_RE = _re.compile("|".join(f"(?:{p})" for p in _CONFIRM_RES))


def _unwrap(cmd: str) -> str:
    """One-layer unwrap for deny checks: bash -c 'RM -rf /' etc."""
    s = str(cmd or "").strip()
    m = _re.search(r"bash\s+-c\s+['\"](.*)['\"]", s,
                   flags=_re.IGNORECASE | _re.DOTALL)
    if m:
        return m.group(1)
    # $(rm -rf /) — extract inner
    m = _re.search(r"\$\(\s*(.+?)\s*\)", s, flags=_re.DOTALL)
    if m:
        return m.group(1)
    return s


def _get(args: dict, *keys: str) -> str:
    """First non-empty string value for any key."""
    for k in keys:
        v = args.get(k, "")
        if isinstance(v, str) and v.strip():
            return v
    return ""


class PolicyGate:
    """One verdict per tool call. Stateless, pure, unit-testable."""

    def decide(self, tool_name: str, args: dict | None) -> tuple[str, str]:
        """Map a tool call to (deny|confirm|allow, reason).

        ``tool_name`` is the post-mapping execution name (bash, read,
        screenshot, ...; legacy exec/read_file/ls names map too).
        Total by construction (plain str/dict reads, isinstance
        guard); unknown names allow.
        ..
        """
        if not isinstance(args, dict):
            args = {}
        return self._decide(str(tool_name or "").lower(), args)

    def _decide(self, name: str, args: dict) -> tuple[str, str]:
        """Verdict dispatch: shell, files, web/computer, planning.

        Duck-typed deny/confirm patterns stay in exactly one place
        (this file).
        ..
        """
        if name in ("bash", "exec", "execute", "exec_bg", "poll"):
            cmd = _unwrap(_get(args, "command", "job_id", "job"))
            if _DENY_RE.search(cmd):
                return "deny", "destructive command blocked by policy"
            if _CONFIRM_CMD_RE.search(cmd):
                return "confirm", "mutating/outward-facing command"
            return "allow", ""
        if name in ("read", "read_file", "glob", "search_files",
                     "list", "ls", "list_directory", "grep"):
            return "allow", ""
        if name in ("webfetch", "fetch", "extract_page",
                     "websearch", "searxng", "web_search", "search_web"):
            return "allow", ""
        if name in ("screenshot", "click", "type_text", "scroll",
                     "get_active_window", "list_windows",
                     "open_url", "browser_click", "browser_scroll",
                     "skill", "question", "write_todos"):
            return "allow", ""
        if name in ("write", "write_file"):
            p = _get(args, "path", "file_path")
            if p in ("", "/", "~") or ".." in p.split("/"):
                return "deny", "refusing to write outside a named file below cwd"
            return "confirm", "mutating/outward-facing tool"
        if name in ("edit", "edit_file"):
            return "confirm", "mutating/outward-facing tool"
        if name in ("move", "move_file", "download_file", "open_app",
                     "task", "python_exec"):
            return "confirm", "mutating/outward-facing tool"
        if name in ("delete", "delete_file"):
            p = _get(args, "path")
            if p in DENY_PATHS:
                return "deny", f"refusing to delete {p or 'empty path'}"
            return "confirm", "destructive tool"
        if name == "press_key":
            keys = "+".join(
                p.strip().lower()
                for p in str(args.get("key", "") or "").split("+")
                if p.strip())
            if keys in _DENY_KEYS:
                return "deny", f"breaker combo refused: {keys}"
            return "confirm", "drives the user session keys"
        if name == "browser_type":
            if bool(args.get("submit", False)):
                return "confirm", "submits a web form"
            return "allow", ""
        return "allow", ""


def _call_of(request) -> tuple[str, dict, str]:
    """(name, args, call_id) off a ToolCallRequest (getattr + isinstance).

    ..
    """
    call = getattr(request, "tool_call", None)
    if not isinstance(call, dict):
        return "", {}, ""
    name = str(call.get("name", "") or "")
    args = call.get("args", {}) or {}
    args = dict(args) if isinstance(args, dict) else {}
    return name, args, str(call.get("id", "") or "")


def _observation(content: str, call_id: str) -> ToolMessage:
    """ToolMessage shaped like a tool result (steers, never breaks)."""
    return ToolMessage(content=str(content), tool_call_id=call_id)


class PolicyMiddleware(AgentMiddleware):
    """Gate every tool call through the policy. One class, one job."""

    def __init__(self, gate=None, confirmer=None, announce=None):
        self.gate = gate or PolicyGate()
        self.confirmer = confirmer
        self.announce = announce

    def _deny(self, reason: str, call_id: str) -> ToolMessage:
        """Steering observation for a refused call. Never raises.

        Tells the model what died and what to do instead (answer from
        evidence, or pick a read-only tool) — a bare refusal loops.
        ..
        """
        return _observation(
            f"policy: denied — {reason}. Do NOT retry or rephrase this "
            f"call; answer from evidence in hand, or use a read-only "
            f"tool instead.", call_id)

    def _declined(self, reason: str, call_id: str) -> ToolMessage:
        """Steering observation for a user-declined call. Never raises.

        Same shape as a deny (the model must move on, not argue).
        ..
        """
        return _observation(
            f"policy: declined by user ({reason}). Do NOT retry this "
            f"call; answer from evidence in hand, or try a different "
            f"approach.", call_id)

    def _say(self, text: str) -> None:
        """Best-effort announce hook. Never raises."""
        if self.announce is None:
            return
        try:
            self.announce(str(text))
        except Exception:
            pass

    async def _ask(self, name: str, args: dict):
        """Resolve one confirm through the confirmer. Never raises.

        No confirmer -> auto-allow (announced). "always"/"a"/"allowlist"
        strings count as yes (matches _confirm_wrapper semantics).
        ..
        """
        if self.confirmer is None:
            self._say(f"policy: {name} needs confirm "
                      f"-> auto-allowed (no confirmer)")
            return True
        try:
            fn = self.confirmer
            res = await fn(name, args) if _inspect.iscoroutinefunction(
                fn) else fn(name, args)
            if _inspect.isawaitable(res):
                res = await res
        except Exception as exc:
            self._say(f"policy: confirmer failed ({exc}) -> declined")
            return False
        if isinstance(res, str) and res.lower() in ("always", "a",
                                                    "allowlist"):
            return True
        return bool(res)

    def _ask_sync(self, name: str, args: dict):
        """Sync confirm resolve. Never raises.

        An async-only confirmer cannot run without a loop here, so it
        auto-allows (announced) instead of deadlocking on it.
        ..
        """
        if self.confirmer is None:
            self._say(f"policy: {name} needs confirm "
                      f"-> auto-allowed (no confirmer)")
            return True
        try:
            fn = self.confirmer
            if _inspect.iscoroutinefunction(fn):
                self._say(f"policy: {name} async confirmer in sync "
                          f"path -> auto-allowed")
                return True
            res = fn(name, args)
            if _inspect.isawaitable(res):
                self._say(f"policy: {name} async confirmer in sync "
                          f"path -> auto-allowed")
                return True
        except Exception as exc:
            self._say(f"policy: confirmer failed ({exc}) -> declined")
            return False
        if isinstance(res, str) and res.lower() in ("always", "a",
                                                    "allowlist"):
            return True
        return bool(res)

    def wrap_tool_call(self, request, handler):
        name, args, cid = _call_of(request)
        verdict, reason = self.gate.decide(name, args)
        if verdict == "deny":
            return self._deny(reason or "refused by policy", cid)
        if verdict == "confirm":
            if not self._ask_sync(name, args):
                return self._declined(reason or "needs confirm", cid)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        name, args, cid = _call_of(request)
        verdict, reason = self.gate.decide(name, args)
        if verdict == "deny":
            return self._deny(reason or "refused by policy", cid)
        if verdict == "confirm":
            if not await self._ask(name, args):
                return self._declined(reason or "needs confirm", cid)
        return await handler(request)
