"""Error-hint gate: bad calls and bad results steer, never stop.

Pluggable hook for the deepagents middleware list (both agent paths).
:class:`ErrorHintMiddleware` wraps tool execution: a raised error
(arg validation, transport failure, tool crash) becomes a normal
``ToolMessage`` carrying the tool name, the failure, and its required
args — plus results that already report ``error:`` gain the same usage
line. The turn continues: the model fixes the call or answers from
evidence instead of dying on an exception. Never raises.
"""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

__all__ = ["ErrorHintMiddleware"]


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


def _usage(name: str) -> str:
    """One-line usage hint (required args) for a known tool, else "".

    Specs load lazily so this module stays import-cheap; unknown names
    get no hint (their own error text stands).
    ..
    """
    if not name:
        return ""
    try:
        from src.agent.mcp.specs import MCP_TOOL_DEFS
    except Exception:
        return ""
    try:
        spec = next((t for t in MCP_TOOL_DEFS if t.get("name") == name), None)
        if spec is None:
            return ""
        params = (spec.get("parameters", {}) or {})
        required = params.get("required", []) or []
        if required:
            return f"usage: {name}({', '.join(required)})"
        return f"usage: {name} (no required args)"
    except Exception:
        return ""


def _hint(name: str, problem: str) -> str:
    """Steering observation: what failed plus how to call again.

    ..
    """
    use = _usage(name)
    tail = f" {use}." if use else ""
    return (f"tool {name or '?'} failed: {problem}.{tail} Do NOT repeat "
            f"this call identically — fix the args and call once, or "
            f"answer from evidence already in hand.")


class ErrorHintMiddleware(AgentMiddleware):
    """Turn tool failures into guided observations. One class, one job."""

    def _failed(self, name: str, exc: Exception, call_id: str) -> ToolMessage:
        """Observation for a raised tool error. Never raises."""
        try:
            problem = str(exc)[:300] or type(exc).__name__
        except Exception:
            problem = "unknown error"
        return ToolMessage(content=_hint(name, problem), tool_call_id=call_id)

    def _apply(self, name: str, result):
        """Append the usage line to ``error:`` results. Never raises."""
        try:
            text = _text_of(result)
        except Exception:
            return result
        if not text.startswith("error:"):
            return result
        use = _usage(name)
        if not use:
            return result
        cid = str(getattr(result, "tool_call_id", "") or "")
        note = (f"\nerror-hint: {use} — fix the args and call once, "
                f"or answer from evidence already in hand.")
        content = getattr(result, "content", "")
        if isinstance(content, str):
            try:
                return result.model_copy(update={"content": text + note})
            except Exception:
                return result
        return result

    def wrap_tool_call(self, request, handler):
        name, _, cid = _call_of(request)
        try:
            result = handler(request)
        except Exception as exc:
            return self._failed(name, exc, cid)
        return self._apply(name, result)

    async def awrap_tool_call(self, request, handler):
        name, _, cid = _call_of(request)
        try:
            result = await handler(request)
        except Exception as exc:
            return self._failed(name, exc, cid)
        return self._apply(name, result)
