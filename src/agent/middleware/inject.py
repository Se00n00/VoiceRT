"""Inject the router's tool shortlist into one model call.

Pluggable hook for the deepagents middleware list: hands the narrowed
op set to :class:`src.agent.chat_model.LocalChatModel` for the current
model call only (a ContextVar owned by :mod:`src.agent.tool_router`,
always reset). Anything unresolved hands through untouched, so a turn
can never break here.
"""
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage

from src.agent.tool_router import SemanticRouter, _current_ops

__all__ = ["InjectToolMiddleware"]


def _request_text(request: Any) -> tuple:
    text, obs = "", ""
    try:
        for m in request.messages or []:
            if isinstance(m, HumanMessage) and isinstance(m.content, str):
                text = m.content
            elif isinstance(m, ToolMessage) and isinstance(m.content, str):
                obs += "\n" + m.content
    except Exception:
        pass
    return text, obs


class InjectToolMiddleware(AgentMiddleware):
    def __init__(self, **kwargs):
        self.router = SemanticRouter(**kwargs)

    def _select(self, request: Any) -> list | None:
        try:
            text, obs = _request_text(request)
            return self.router.route(text, obs)
        except Exception:
            return None

    def wrap_model_call(self, request, handler):
        ops = self._select(request)
        if not ops:
            return handler(request)
        tok = _current_ops.set(tuple(ops))
        try:
            return handler(request)
        finally:
            _current_ops.reset(tok)

    async def awrap_model_call(self, request, handler):
        ops = self._select(request)
        if not ops:
            return await handler(request)
        tok = _current_ops.set(tuple(ops))
        try:
            return await handler(request)
        finally:
            _current_ops.reset(tok)
