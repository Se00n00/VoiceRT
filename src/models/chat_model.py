"""Chat model wrapper for the local legs behind the chat template.

Adapts :class:`src.models.llm.LlmModel` to the LangChain
``BaseChatModel`` interface expected by ``deepagents.create_deep_agent``.

Tools travel in the template's native grammar (``tools=`` specs in,
template calls out) and pass through verbatim as ``tool_calls`` —
extraction lives in :meth:`src.models.llm.LlmModel.extract_tool_calls`,
op mapping to the execution layer outside this file. This module holds
no grammar and no parser. Thinking traces (``<think>…</think>``) ride
in ``additional_kwargs`` for the TUI.
"""
import asyncio
import concurrent.futures
import json
import re
import uuid
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import create_tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_function

from src.agent.budget import for_backend
from src.agent.tool_router import (
    ALL_TOOL_DEFS,
    current_ops,
    tools_for_request,
)
from src.models.llm import LlmModel
from src.agent.prompts.chat import (
    ECHO_RETRY,
    GARBLED_RETRY,
    REPEAT_NOTE_TEMPLATE,
)

def is_degenerate(raw: str) -> bool:
    """Small-model babble guard: one word dominating a long reply."""
    words = str(raw or "").split()
    if len(words) < 10:
        return False
    from collections import Counter

    top = Counter(w.lower() for w in words).most_common(1)[0][1]
    return top / len(words) > 0.4


def is_echo(thinking: str, answer: str) -> bool:
    """Detect the 1B dither: model repeats its thinking as the reply.

    E.g. think ``I should use bash…`` + answer identical —
    narration instead of a call (or instead of a real chat sentence).
    Whitespace-normalized; 40-char floor so short coincidences
    (``hi`` / ``hi``) don't trigger a wasted retry.
    """
    t = " ".join(str(thinking or "").split())
    a = " ".join(str(answer or "").split())
    if not t or not a:
        return False
    if a == t:
        return True
    if len(t) >= 40 and t in a:
        return True
    if len(a) >= 40 and a in t:
        return True
    return False


# Read-only ops: repeating one with identical args can never add
# information (the observation is already in context). Side-effecting
# ops (bash/python/write/edit) and job polls legitimately repeat.
REPEAT_GUARD_OPS = frozenset({
    "read", "list", "glob", "grep", "websearch", "webfetch",
    "get_active_window", "list_windows", "skill",
})

EXTRA_PROMPT_TOOLS = ("task", "write_todos")
"""Graph-bound tools with NO house def in ``ALL_TOOL_DEFS``.

The router's ``task`` def describes the MCP background-job tool, but the
bound graph tool is the subagent transport (and ``write_todos`` comes
from TodoListMiddleware). For these, the real schema is rendered into
the prompt instead of the router def — always, when the graph has them.
"""


class LocalChatModel(BaseChatModel):
    """Thin LangChain wrapper around :class:`LlmModel`."""

    llm: Any
    bound_tools: Optional[List[dict]] = None
    # tool_names: execution tools bound by the current graph. The prompt
    # half (router/ops) is intersected with these so the template can
    # never advertise an op the graph cannot execute (subagent tag sets,
    # planner's Main subset).
    tool_names: Optional[frozenset] = None
    # extra_defs: house-shaped defs converted from graph-bound tools that
    # have no router def of their own (see EXTRA_PROMPT_TOOLS).
    extra_defs: Optional[List[dict]] = None
    # no_tools: prompt WITHOUT tool specs (summarizer calls). A model
    # handed tools alongside a "respond ONLY with ..." instruction may
    # emit a tool call instead of the requested text.
    no_tools: bool = False

    def __init__(self, llm: Any, **kwargs):
        super().__init__(llm=llm, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "local-qwen"

    def bind_tools(self, tools, **kwargs):
        # Only dict specs pass through untouched. StructuredTools bound by
        # the agent stay execution-side: the prompt half is decided by
        # backend in _prompt_tools — rendering foreign schemas into
        # the chat template would silently change model behavior.
        defs = [t for t in tools or []
                if isinstance(t, dict) and "name" in t]
        names, extras = set(), []
        for t in tools or []:
            if isinstance(t, dict):
                n = str(t.get("name", "") or "")
            else:
                n = str(getattr(t, "name", "") or "")
            if not n:
                continue
            names.add(n)
            if n in EXTRA_PROMPT_TOOLS and not isinstance(t, dict):
                try:
                    extras.append(convert_to_openai_function(t))
                except Exception:
                    continue
        return LocalChatModel(llm=self.llm, bound_tools=defs or None,
                              tool_names=frozenset(names) or None,
                              extra_defs=extras or None)

    def _messages_to_llm_input(self, messages: List[BaseMessage]):
        # Convert LangChain messages to the (text, history, observation)
        # triple expected by LlmModel.messages_for_terminal, preserving tool
        # results as observation text (and separately, for the tool router).
        # Bodies are capped so one huge turn can't blow the small local
        # context window.
        history = []
        text = ""
        observations: list[str] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                # Skip: LlmModel.messages_for_terminal already prepends the
                # system prompt, and strict templates (Qwen3.8/Bonsai raise
                # "System message must be at the beginning") reject a
                # second system entry surfacing mid-history via deepagents
                # re-sends. Measured 500s on every Bonsai agent turn.
                continue
            elif isinstance(m, HumanMessage):
                text = str(m.content)[:2000]
                # Don't duplicate: if history already has this user turn, skip
                # (deepagents may re-send). We just keep last human as text.
            elif isinstance(m, AIMessage):
                # Prior assistant turn (for history)
                if m.content:
                    history.append({"role": "assistant", "content": str(m.content)})
                # If it had tool calls, we represent them as assistant + tool results
                # will follow as ToolMessages
            elif isinstance(m, ToolMessage):
                # Tool result -> observation for next turn
                # We append as user-like observation; the terminal harness
                # does this via `observation` kwarg, but for chat we inline.
                observations.append(str(m.content)[:800])
                text += f"\nLast result: {str(m.content)[:800]}"
        return text, history, "\n".join(observations)

    def _prompt_tools(self, text: str = "", observation: str = ""):
        if self.no_tools:
            return None
        tools = self.bound_tools
        backend = str(getattr(getattr(self.llm, "config", None), "backend", ""))
        if tools is None and backend in ("bonsai", "qwen"):
            # Native-spec legs get the narrowed defs as REAL function specs
            # (qwen: chat-template tools=; bonsai: OpenAI tools[]). Small
            # models drown in 11 — the semantic shortlist wins when
            # InjectToolMiddleware set one, else the keyword router decides.
            ops = current_ops()
            if ops:
                by_name = {t["name"]: t for t in ALL_TOOL_DEFS}
                picked = [by_name[o] for o in ops if o in by_name]
                if picked:
                    tools = picked
            if tools is None:
                # Route: narrow the 9 native tool definitions to what this
                # step plausibly needs (small models drown in 9). The harness
                # appends a "CWD: ... SHELL: ..." trailer to every turn — strip
                # it for routing (else "shell" forces exec into every set).
                # None (or another backend) keeps the full set / preamble path.
                routable = re.sub(r"\nCWD: .*?(\s+SHELL: \w+)?\s*$", "", text)
                tools = tools_for_request(routable, observation)
        if tools and self.tool_names:
            # Never advertise an op the graph has no execution tool for
            # (planner Main subset, specialist tag sets).
            tools = [t for t in tools
                     if str((t or {}).get("name", "")) in self.tool_names]
        if tools is not None and self.extra_defs:
            # Replace/append the real schema for graph tools the router
            # cannot describe correctly (subagent task, write_todos).
            have = {str((t or {}).get("name", "")) for t in tools}
            for d in self.extra_defs:
                name = str(d.get("name", "") or "")
                if not name or name not in self.tool_names:
                    continue
                if name in have:
                    tools = [d if str((t or {}).get("name", "")) == name
                             else t for t in tools]
                else:
                    tools = [*tools, d]
        return tools

    def _needs_catalogue(self) -> bool:
        """Prose op catalogue wanted only without native tool specs.

        Template-native legs get real function specs beside the
        messages — pure English suffices. Unknown backends keep the
        catalogue (fail open).
        ..
        """
        backend = str(getattr(getattr(self.llm, "config", None), "backend", ""))
        return backend not in ("bonsai", "qwen")

    def _step_budget(self) -> int:
        # Think + tool call must fit in ONE step (the old harness learned
        # this the hard way: 48/160 strangled the model mid-think, killing
        # the think→toolcall loop). Floors come from the budget table
        # (src/agent/budget.py:BUDGETS) so pack() and the loop agree.
        base_max = int(getattr(getattr(self.llm, "config", None), "max_tokens", 48) or 48)
        backend = str(getattr(getattr(self.llm, "config", None), "backend", ""))
        return max(base_max, for_backend(backend).step_floor)

    def _parse_action(self, raw: str):
        """Split thinking; model-owned template extraction; else chat."""
        thinking, answer = LlmModel.split_thinking(raw)
        ext = getattr(self.llm, "extract_tool_calls", None)
        calls: list = []
        if callable(ext):
            if answer.strip():
                calls = list(ext(answer) or [])
            if not calls and raw.strip():
                calls = list(ext(raw) or [])
        return thinking, answer, calls

    def _needs_retry(self, raw: str, thinking: str, answer: str) -> str | None:
        """One-retry nudge for garbled or echoed output."""
        if is_degenerate(raw):
            return GARBLED_RETRY
        if is_echo(thinking, answer):
            return ECHO_RETRY
        return None

    async def _stream_raw(self, msgs, max_tokens: int, tools, acc: dict):
        """Yield (kind, piece) live; stash the full text in ``acc["raw"]``.

        kind is "text" or "think" (sidecar reasoning deltas). Think pieces
        accumulate into ``acc["raw"]`` wrapped as ``<think>`` blocks so the
        existing ``LlmModel.split_thinking`` parse recovers the full trace
        (multiple blocks join). Full text prefers whole-id re-decode
        (per-piece decodes can split BPE pairs), else joined pieces, else
        one blocking generate.
        """
        stop = None
        acc["raw"] = ""
        acc["live"] = False
        stream = getattr(self.llm, "stream", None)
        if callable(stream):
            try:
                try:
                    gen = stream(msgs, max_tokens=max_tokens, tools=tools, stop=stop)
                except TypeError:
                    try:
                        gen = stream(msgs, max_tokens=max_tokens, tools=tools)
                    except TypeError:
                        gen = stream(msgs, max_tokens=max_tokens)
                acc["live"] = True
                pieces: list[tuple] = []
                ids: list[int] = []
                live = False
                try:
                    async for tok in gen:
                        kind = str(getattr(tok, "kind", "text") or "text")
                        piece = str(getattr(tok, "piece", "") or "")
                        try:
                            ids.append(int(getattr(tok, "token_id", -1)))
                        except Exception:
                            pass
                        if piece:
                            pieces.append((kind, piece))
                            live = True
                            yield kind, piece
                except TypeError:
                    pass  # not an async generator (odd fake) — fall through
                if live or [i for i in ids if i >= 0]:
                    # Re-decode whole ids: per-piece decodes can split BPE.
                    good_ids = [i for i in ids if i >= 0]
                    if good_ids and hasattr(self.llm, "decode"):
                        try:
                            decoded = await self.llm.decode(good_ids)
                            if decoded:
                                acc["raw"] = decoded
                                return
                        except Exception:
                            pass
                    if pieces:
                        # Group consecutive think pieces into ONE block:
                        # per-piece wrapping re-parses as newline-joined
                        # word salad ("The\nuser\nsaid…", measured 2026-09-28).
                        buf: list[str] = []
                        think_buf: list[str] = []
                        for k, p in pieces:
                            if k == "think":
                                think_buf.append(p)
                            else:
                                if think_buf:
                                    buf.append("<think>" + "".join(think_buf) +
                                               "</think>")
                                    think_buf.clear()
                                buf.append(p)
                        if think_buf:
                            buf.append("<think>" + "".join(think_buf) +
                                       "</think>")
                        acc["raw"] = "".join(buf)
                        return
            except Exception:
                pass
        # Non-streaming legs (and total stream failure): one blocking call.
        try:
            res = await self.llm.generate(msgs, max_tokens=max_tokens, tools=tools, stop=stop)
        except TypeError:
            try:
                res = await self.llm.generate(msgs, max_tokens=max_tokens, tools=tools)
            except TypeError:
                res = await self.llm.generate(msgs, max_tokens=max_tokens)
        raw = str(getattr(res, "text", "") or "")
        acc["raw"] = raw
        if raw:
            yield "text", raw

    async def _collect_raw(self, msgs, max_tokens: int, tools) -> str:
        """Full raw text for one step, with one retry on garble/echo."""
        acc: dict = {}
        async for _ in self._stream_raw(msgs, max_tokens, tools, acc):
            pass
        raw = acc.get("raw", "")
        if not raw:
            return raw
        thinking, answer, calls = self._parse_action(raw)
        nudge = self._needs_retry(raw, thinking, answer)
        if nudge is None:
            return raw
        acc2: dict = {}
        async for _ in self._stream_raw(msgs + [{"role": "user", "content": nudge}],
                                        max_tokens, tools, acc2):
            pass
        return acc2.get("raw", "") or raw
    def _final_message(self, raw: str):
        """Build the result AIMessage: template tool calls or plain chat."""
        thinking, answer, calls = self._parse_action(raw)
        msg = AIMessage(
            content=answer or raw,
            tool_calls=[{"name": name, "args": args,
                         "id": f"call_{uuid.uuid4().hex[:8]}",
                         "type": "tool_call"} for name, args in calls],
        )
        if thinking:
            msg.additional_kwargs["thinking"] = thinking
        return msg

    @staticmethod
    def _repeat_note(messages: List[BaseMessage]) -> str | None:
        """Self-note short-circuit when the last two tool calls match.

        Measured 2026-09-28: Bonsai issued the same websearch 7× in one
        turn ("let me fetch X" in thinking, websearch again in acting)
        and never answered. When the two most recent tool calls are the
        same read-only op with identical args, the next call would be the
        third copy — return a note that rides along as extra user context
        for the model call (the model still decides, and its words stay
        the reply). Bounded by the existing recursion limit; returns None
        (normal path) otherwise.
        """
        recent: list[tuple] = []
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in getattr(m, "tool_calls", None) or []:
                    if isinstance(tc, dict):
                        name, args = tc.get("name", ""), tc.get("args", {})
                    else:
                        name, args = getattr(tc, "name", ""), getattr(tc, "args", {})
                    try:
                        key = (str(name), json.dumps(args, sort_keys=True,
                                                     default=str))
                    except Exception:
                        continue
                    recent.append(key)
        if len(recent) < 2 or recent[-1] != recent[-2]:
            return None
        name, _ = recent[-1]
        if name not in REPEAT_GUARD_OPS:
            return None
        return REPEAT_NOTE_TEMPLATE.format(name=name)

    async def _agenerate_async(
        self, messages: List[BaseMessage], **kwargs
    ) -> ChatResult:
        text, history, observation = self._messages_to_llm_input(messages)
        note = self._repeat_note(messages)
        if note is not None:
            # Same read-only call twice with evidence in hand: the next
            # call would be the third copy. Don't block it — steer it:
            # the note rides along as context and the model still decides
            # (and its words stay the reply).
            text = f"{text}\n{note}"
        tools = self._prompt_tools(text, observation)
        msgs = self.llm.messages_for_terminal(text, history, cwd="",
                                              observation="",
                                              catalogue=self._needs_catalogue())
        raw = await self._collect_raw(msgs, self._step_budget(), tools)
        return ChatResult(generations=[ChatGeneration(message=self._final_message(raw))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """True per-token stream for the agent loop.

        Yields live content chunks, then ONE final chunk carrying the
        parsed ``tool_calls`` (aggregated by langchain into the AI
        message) and ``thinking`` in ``additional_kwargs``. Garble/echo
        retries stream transparently inside the same step.
        """
        def _cgc(msg):
            return ChatGenerationChunk(message=msg)

        text, history, observation = self._messages_to_llm_input(messages)
        note = self._repeat_note(messages)
        if note is not None:
            text = f"{text}\n{note}"
        tools = self._prompt_tools(text, observation)
        max_tokens = self._step_budget()
        msgs = self.llm.messages_for_terminal(text, history, cwd="",
                                              observation="",
                                              catalogue=self._needs_catalogue())

        acc0: dict = {}
        # NOTE: nested async-generator drain (no return values allowed).
        live: bool | None = None
        agen = self._stream_raw(msgs, max_tokens, tools, acc0)
        async for kind, piece in agen:
            # Generate-fallback legs yield one whole-text piece: keep their
            # "no token events" semantics, content goes on the final chunk.
            if live is None:
                live = bool(acc0.get("live"))
            if not live:
                continue
            if kind == "think":
                # Live think lane: the TUI appends these to the think
                # bubble (main.py emits thinking/append events for them).
                # The full trace still lands via the final parse as well.
                yield _cgc(AIMessageChunk(
                    content="",
                    additional_kwargs={"thinking_delta": piece}))
            else:
                yield _cgc(AIMessageChunk(content=piece))
        raw = acc0.get("raw", "")
        if raw:
            thinking, answer, calls0 = self._parse_action(raw)
            nudge = self._needs_retry(raw, thinking, answer)
            if nudge is not None:
                acc1: dict = {}
                live2: bool | None = None
                agen2 = self._stream_raw(msgs + [{"role": "user", "content": nudge}],
                                          max_tokens, tools, acc1)
                async for kind2, piece2 in agen2:
                    if live2 is None:
                        live2 = bool(acc1.get("live"))
                    if not live2:
                        continue
                    if kind2 == "think":
                        yield _cgc(AIMessageChunk(
                            content="",
                            additional_kwargs={"thinking_delta": piece2}))
                    else:
                        yield _cgc(AIMessageChunk(content=piece2))
                raw = acc1.get("raw", "") or raw
                live = live2 if live2 is not None else live
        thinking, answer, calls = self._parse_action(raw)
        extra: dict = {}
        if thinking:
            extra["thinking"] = thinking
        chunks = []
        for i, (name, args) in enumerate(calls):
            chunks.append(create_tool_call_chunk(
                name=name,
                args=json.dumps(args),
                id=f"call_{uuid.uuid4().hex[:8]}",
                index=i,
            ))
        final_content = "" if live else answer
        if chunks:
            yield _cgc(AIMessageChunk(content=final_content, additional_kwargs=extra,
                                      tool_call_chunks=chunks))
        elif extra:
            yield _cgc(AIMessageChunk(content=final_content, additional_kwargs=extra))
        else:
            yield _cgc(AIMessageChunk(content=final_content))

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._agenerate_async(messages, **kwargs))
        else:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(asyncio.run, self._agenerate_async(messages, **kwargs))
                return fut.result()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await self._agenerate_async(messages, **kwargs)
