"""Policy gate: deny short-circuits, confirms ask, allows run.

Pluggable hook for the deepagents middleware list (both agent paths).
:class:`PolicyMiddleware` asks :class:`src.agent.policy.PolicyGate`
once per tool call: ``deny`` returns a steering observation WITHOUT
running the tool; ``confirm`` awaits the harness confirmer (WS gate on
server, ``lambda: True`` on autonomous routes); with no confirmer the
call auto-allows and the ``announce`` hook logs it (console). ``allow``
runs untouched. Sync and async paths both implemented (the harness
invokes async; sync stays for unit calls). Never raises.
"""
import inspect as _inspect

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from src.agent.policy import PolicyGate

__all__ = ["PolicyMiddleware"]


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
