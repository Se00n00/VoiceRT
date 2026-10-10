"""Result verification: trust, but check the observable world.

Pluggable hook for the deepagents middleware list (both agent paths).
:class:`VerifyMiddleware` runs AFTER each tool call and checks the
claim against the filesystem: wrote file -> file exists; moved ->
dst exists and src gone; deleted -> path gone; downloaded/screenshot
-> non-empty file. A mismatch appends a ``verify:`` note to the
observation (the model re-checks next step); three in a row appends
the restate nudge and resets. Matches reset the count. Tools that
already reported ``error:`` and unverifiable tools pass through
untouched. Never raises, never blocks.
"""
from contextvars import ContextVar as _ContextVar
import os as _os

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from src.agent.mcp.filesystem import _resolve as _anchor
from src.prompts.chat import VERIFY_RETRY

__all__ = ["VerifyMiddleware", "verify_strikes"]

verify_strikes: _ContextVar = _ContextVar("verify_strikes", default=0)
"""Consecutive verify mismatches (ContextVar, reset on match)."""


def _text_of(result) -> str:
    """ToolMessage content -> plain text (handles block lists).

    ..
    """
    content = getattr(result, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "") if isinstance(b, dict) else b)
            for b in content)
    return str(content or "")


def _call_of(request) -> tuple[str, dict]:
    """(name, args) off a ToolCallRequest (getattr + isinstance).

    ..
    """
    call = getattr(request, "tool_call", None)
    if not isinstance(call, dict):
        return "", {}
    name = str(call.get("name", "") or "")
    args = call.get("args", {}) or {}
    return name, dict(args) if isinstance(args, dict) else {}


class VerifyMiddleware(AgentMiddleware):
    """Check tool claims against the world. One class, one job."""

    def check(self, name: str, args: dict, text: str):
        """Return (expected, observed) on mismatch, else None.

        The single os-read boundary: every probe below runs inside
        this one guard. ``error:`` results are the tool's business.
        ..
        """
        try:
            if not isinstance(args, dict):
                return None
            return self._check(str(name or ""), args, str(text or ""))
        except Exception:
            return None

    def _check(self, name: str, args: dict, text: str):
        """Mismatch detector by tool. None = verified or unverifiable.

        Read-only probes, capped. Paths resolve exactly like the
        implementations (VOICE_WORKDIR anchor); content args stay
        verbatim, never resolved.
        ..
        """
        if text.startswith("error:"):
            return None
        get = lambda k: _anchor(args.get(k, "") or "")
        raw = lambda k: str(args.get(k, "") or "")
        exists = lambda p: bool(p) and _os.path.lexists(p)
        size = lambda p: _os.path.getsize(p) if p else -1
        if name == "write_file":
            p = get("path")
            if not exists(p):
                return f"file at {p} exists", f"nothing at {p}"
            if raw("content") and size(p) <= 0:
                return f"file at {p} non-empty", f"empty file at {p}"
            return None
        if name == "edit_file":
            p = get("path")
            want = raw("new_string")
            if not exists(p):
                return f"file at {p} exists", f"nothing at {p}"
            if want:
                with open(p, "r", encoding="utf-8",
                          errors="replace") as f:
                    body = f.read(200000)
                if want[:4000] not in body:
                    return (f"{want[:60]!r} in {p}",
                            f"replacement not found in {p}")
            return None
        if name == "move_file":
            s, d = get("src"), get("dst")
            if not exists(d):
                return f"moved file at {d}", f"nothing at {d}"
            if exists(s):
                return f"{s} gone after move", f"{s} still present"
            return None
        if name == "delete_file":
            p = get("path")
            if exists(p):
                return f"{p} deleted", f"{p} still present"
            return None
        if name in ("download_file", "screenshot"):
            p = get("dest") or get("path")
            if not exists(p) or size(p) <= 0:
                return f"non-empty file at {p}", \
                    f"missing/empty file at {p or '?'}"
            return None
        return None

    def _apply(self, request, result):
        """Append the verify note on mismatch; track strikes.

        Returns the (possibly new) ToolMessage. Three consecutive
        mismatches add the restate nudge and reset the count.
        ..
        """
        name, args = _call_of(request)
        text = _text_of(result)
        miss = self.check(name, args, text)
        if miss is None:
            verify_strikes.set(0)
            return result
        expected, observed = miss
        n = verify_strikes.get()
        n = n + 1 if isinstance(n, int) else 1
        verify_strikes.set(n)
        note = (f"\nverify: called {name} expecting {expected}, "
                f"but observed {observed}. Re-check with "
                f"list_directory/read_file before acting further.")
        if n >= 3:
            note += " " + VERIFY_RETRY
            verify_strikes.set(0)
        cid = str(getattr(result, "tool_call_id", "") or "")
        return ToolMessage(content=text + note, tool_call_id=cid)

    def wrap_tool_call(self, request, handler):
        return self._apply(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._apply(request, await handler(request))
