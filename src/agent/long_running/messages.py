"""Message conversion and final-reply extraction for the agent turn.

Pure helpers shared by the middleware, the turn runner and the voice
facade: session dicts <-> LangChain messages, stream chunk pieces,
thinking extraction, and the one finalization rule (last assistant text,
``Done.`` after tools, degenerate fallback). Never touches the graph.
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.prompts.voice import DEGENERATE_REPLY
from src.models.chat_model import is_degenerate
from src.models.llm import LlmModel

__all__ = [
    "TurnResult",
    "chunk_pieces",
    "finalize_reply",
    "finalize_turn",
    "lc_messages",
    "norm_msg",
    "resolve_reply",
    "text_of",
    "think_of",
]


@dataclass(frozen=True)
class TurnResult:
    """Final text, thinking trace and work flags for one agent turn."""

    reply: str
    thinking: str
    did_work: bool
    degenerate: bool


def lc_messages(history: list) -> list:
    """Session dicts -> LangChain user/assistant messages."""
    out = []
    for m in history or []:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), str(m.get("content", "") or "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
    return out


def norm_msg(m) -> tuple:
    """Reduce a LangChain message or dict to type, content, calls, kwargs."""
    if isinstance(m, dict):
        return (
            m.get("type") or m.get("role") or "",
            m.get("content", ""),
            m.get("tool_calls"),
            m.get("additional_kwargs") or {},
        )
    return (
        getattr(m, "type", None) or "",
        getattr(m, "content", ""),
        getattr(m, "tool_calls", None),
        getattr(m, "additional_kwargs", None) or {},
    )


def text_of(content) -> str:
    """Message content (str or block list) -> plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(c.get("text", c)) for c in content if isinstance(c, dict)
        )
    return str(content or "")


def chunk_pieces(chunk) -> list[str]:
    """Extract non-empty text pieces from a streamed model chunk."""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return [p for p in parts if p]
    return []


def think_of(ak) -> str:
    """Pull the thinking trace out of message kwargs, else empty."""
    return str(ak.get("thinking", "") or "") if isinstance(ak, dict) else ""


def resolve_reply(content) -> str:
    """Text-only AI content -> stripped reply."""
    return str(content or "").strip()


def finalize_reply(content: str, did_work: bool = False) -> TurnResult:
    """Finalize one assistant text with the shared reply rules.

    ..
    """
    split_think, answer = LlmModel.split_thinking(str(content or ""))
    reply = str(answer or content or "").strip()
    if not reply and did_work:
        reply = "Done."
    degenerate = bool(reply and is_degenerate(reply))
    if degenerate:
        reply = DEGENERATE_REPLY
    return TurnResult(
        reply=reply,
        thinking=split_think,
        did_work=bool(did_work),
        degenerate=degenerate,
    )


def finalize_turn(messages: list) -> TurnResult:
    """Extract the final reply and thinking from one turn's messages.

    Uses the last assistant message without tool calls, marks tool work,
    and applies the shared ``Done.`` and degenerate-output rules.
    ..
    """
    reply, thinking, last_think = "", "", ""
    did_work = False
    for m in messages or []:
        role, content, tool_calls, ak = norm_msg(m)
        text = text_of(content)
        if role in ("ai", "assistant"):
            think = think_of(ak)
            if think:
                last_think = think
                thinking = think
            if tool_calls:
                did_work = True
            elif text.strip():
                reply = resolve_reply(text)
        elif role == "tool":
            did_work = True
    split_think, answer = LlmModel.split_thinking(reply)
    if split_think and split_think != last_think:
        thinking = split_think
    reply = str(answer or reply or "").strip()
    if not reply and did_work:
        reply = "Done."
    degenerate = bool(reply and is_degenerate(reply))
    if degenerate:
        reply = DEGENERATE_REPLY
    return TurnResult(
        reply=reply,
        thinking=thinking,
        did_work=did_work,
        degenerate=degenerate,
    )
