"""Chat model wrapper for MiniCPM / Qwen local LLMs.

Adapts :class:`src.models.llm.LlmModel` (which speaks via
``messages_for_terminal`` + JSON/XML tool calling) to the LangChain
``BaseChatModel`` interface expected by ``deepagents.create_deep_agent``.

It supports ``bind_tools`` / ``with_structured_output`` by storing the
bound tools and injecting them into the prompt as the native tool
definitions (for MiniCPM, via ``tools=`` to the chat template; for Qwen,
via the JSON preamble). Tool calls are parsed from the model's text
output and returned as ``tool_calls`` on the ``AIMessage``.

Thinking tokens (``<think>…</think>``) are preserved via
``split_thinking`` and emitted as a separate ``thinking`` field on the
message's ``additional_kwargs`` so the TUI can render them dimmed.
"""
import asyncio
import json
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from src.models.llm import LlmModel, split_thinking
from src.tools.terminal import TERMINAL_TOOLS, is_degenerate, is_echo, parse_bare_tail, parse_functiongemma_action, parse_gemma_action, parse_terminal_action, parse_xml_action

# Read-only ops: repeating one with identical args can never add
# information (the observation is already in context). Side-effecting
# ops (exec/python/write/edit) and poll loops are excluded — retries
# and job polls legitimately repeat.
REPEAT_GUARD_OPS = frozenset({
    "read", "list", "grep", "searxng", "fetch",
    "read_file", "ls", "web_search",
})


class LocalChatModel(BaseChatModel):
    """Thin LangChain wrapper around :class:`LlmModel`."""

    llm: Any
    bound_tools: Optional[List[dict]] = None
    # no_tools: prompt WITHOUT tool specs (summarizer calls). A model
    # handed tools alongside a "respond ONLY with ..." instruction may
    # emit a tool call instead of the requested text.
    no_tools: bool = False

    def __init__(self, llm: Any, **kwargs):
        super().__init__(llm=llm, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "local-minicpm"

    def bind_tools(self, tools, **kwargs):
        # Only dict specs pass through untouched. StructuredTools bound by
        # the agent stay execution-side: the prompt half is decided by
        # backend in _prompt_tools (native TERMINAL_TOOLS for minicpm,
        # JSON-preamble path otherwise) — rendering foreign schemas into
        # the chat template would silently change model behavior.
        defs = [t for t in tools or []
                if isinstance(t, dict) and "name" in t]
        return LocalChatModel(llm=self.llm, bound_tools=defs or None)

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
        if tools is None and (backend.startswith("minicpm") or backend in ("bonsai", "qwen17")):
            # Native-spec legs get the narrowed defs as REAL function specs
            # (minicpm: chat-template tools=; bonsai: OpenAI tools[]). Small
            # models drown in 11 — the semantic shortlist wins when
            # InjectToolMiddleware set one, else the keyword router decides.
            from src.agent.tool_router import current_ops
            from src.tools.terminal import TERMINAL_TOOLS as _NATIVE

            ops = current_ops()
            if ops:
                by_name = {t["name"]: t for t in _NATIVE}
                picked = [by_name[o] for o in ops if o in by_name]
                if picked:
                    return picked
            # Route: narrow the 9 native tool definitions to what this
            # step plausibly needs (small models drown in 9). The harness
            # appends a "CWD: ... SHELL: ..." trailer to every turn — strip
            # it for routing (else "shell" forces exec into every set).
            # None (or non-minicpm) keeps the full set / preamble path.
            import re

            from src.tools.terminal import tools_for_request

            routable = re.sub(r"\nCWD: .*?(\s+SHELL: \w+)?\s*$", "", text)
            tools = tools_for_request(routable, observation)
        return tools

    def _step_budget(self) -> int:
        # Think + tool call must fit in ONE step (the old harness learned
        # this the hard way: 48/160 strangled the model mid-think, killing
        # the think→toolcall loop). Floors come from the budget table
        # (src/agent/budget.py:BUDGETS) so pack() and the loop agree.
        from src.agent.budget import for_backend

        base_max = int(getattr(getattr(self.llm, "config", None), "max_tokens", 48) or 48)
        backend = str(getattr(getattr(self.llm, "config", None), "backend", ""))
        return max(base_max, for_backend(backend).step_floor)

    def _parse_action(self, raw: str):
        """Gemma native envelope first, JSON, fuzzy aliases, bare tail last."""
        thinking, answer = split_thinking(raw)
        action = parse_gemma_action(answer) if answer.strip() else None
        if action is None and answer.strip():
            # FunctionGemma-270M native envelope (270M context-lab leg).
            action = parse_functiongemma_action(answer)
        if action is None and answer.strip():
            action = parse_terminal_action(answer)
        if action is None and answer.strip():
            action = parse_xml_action(answer)
        if action is None:
            action = parse_gemma_action(raw)
        if action is None:
            action = parse_functiongemma_action(raw)
        if action is None:
            action = parse_terminal_action(raw)
        if action is None:
            action = parse_xml_action(raw)
        if action is None and answer.strip():
            action = parse_bare_tail(answer)
        if action is None:
            action = parse_bare_tail(raw)
        return thinking, answer, action

    @staticmethod
    def _builtin_call(action) -> tuple:
        """Old terminal op -> deepagents built-in (name, args).

        The prompt still shows the legacy schema (exec/read/write/... with
        path/text args); the graph executes the built-ins (execute/
        read_file/... with file_path/content args). Anything without a
        built-in passes through unchanged (the tool node reports it and
        the model recovers).
        """
        d = action.as_dict()
        op = action.op

        def _abs(p: str) -> str:
            p = str(p or "")
            return p if p.startswith("/") else "/" + p

        if op == "exec":
            return "execute", {"command": d.get("command", "")}
        if op == "read":
            return "read_file", {"file_path": _abs(d.get("path", ""))}
        if op == "write":
            return "write_file", {"file_path": _abs(d.get("path", "")),
                                  "content": d.get("text", "")}
        if op == "edit":
            return "edit_file", {"file_path": _abs(d.get("path", "")),
                                 "old_string": d.get("anchor", ""),
                                 "new_string": d.get("text", "")}
        if op == "list":
            return "ls", {"path": _abs(d.get("path", "") or ".")}
        if op == "grep":
            args: dict = {"pattern": d.get("pattern", "")}
            if d.get("path", ""):
                args["path"] = _abs(d.get("path", ""))
            return "grep", args
        if op == "searxng":
            # advertised name; the attached tool is the direct web_search.
            return "web_search", {"pattern": d.get("pattern", "")}
        return op, {k: v for k, v in d.items() if k != "action"}

    @staticmethod
    def _looks_like_call(raw: str) -> bool:
        """Unparsed text that still smells like a tool attempt."""
        t = str(raw or "")
        return ("<|tool_call>" in t or "<function" in t
                or "<start_function_call>" in t
                or '"action"' in t or "'action'" in t)

    def _needs_retry(self, raw: str, thinking: str, answer: str,
                     action=None) -> str | None:
        """One-retry nudge for garbled, echoed, or unparsable-call output."""
        if is_degenerate(raw):
            return ("That reply was garbled. Answer again: ONLY one JSON "
                    "action or one short sentence.")
        if is_echo(thinking, answer):
            return ("Do not repeat your thinking as the reply. Reply with "
                    "EITHER one short chat sentence OR exactly one function "
                    "call, nothing else.")
        if action is None and self._looks_like_call(raw):
            # Measured 2026-09-28: a 27B model emitted a call-shaped reply
            # the parsers rejected, and the turn silently ended as chat.
            # One nudge recovers it instead of dropping the user's task.
            return ("Your tool call didn't parse (invalid JSON?). Reply "
                    "with EXACTLY one valid function call, nothing else.")
        return None

    async def _stream_raw(self, msgs, max_tokens: int, tools, acc: dict):
        """Yield (kind, piece) live; stash the full text in ``acc["raw"]``.

        kind is "text" or "think" (sidecar reasoning deltas). Think pieces
        accumulate into ``acc["raw"]`` wrapped as ``<think>`` blocks so the
        existing :func:`split_thinking` parse recovers the full trace
        (multiple blocks join). Full text prefers whole-id re-decode
        (per-piece decodes can split BPE pairs), else joined pieces, else
        one blocking generate.
        """
        backend = str(getattr(getattr(self.llm, "config", None), "backend", ""))
        if backend == "gemma270":
            # FunctionGemma rambles call/response/call chains without a
            # stop: end generation at the first call close so exactly one
            # action materializes per step.
            stop = ["<end_function_call>"] if tools else None
        else:
            stop = ["</function>"] if tools else None
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
        thinking, answer, action = self._parse_action(raw)
        nudge = self._needs_retry(raw, thinking, answer, action)
        if nudge is None:
            return raw
        acc2: dict = {}
        async for _ in self._stream_raw(msgs + [{"role": "user", "content": nudge}],
                                        max_tokens, tools, acc2):
            pass
        return acc2.get("raw", "") or raw
    def _final_message(self, raw: str):
        """Build the result AIMessage: parsed tool call or plain chat."""
        import uuid

        thinking, answer, action = self._parse_action(raw)
        if action and action.op != "done":
            name, args = self._builtin_call(action)
            tc = {
                "name": name,
                "args": args,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
            msg = AIMessage(content=answer or raw, tool_calls=[tc])
            if thinking:
                msg.additional_kwargs["thinking"] = thinking
            return msg
        if action is not None:
            # Legacy done envelope: the reply IS the content (never the JSON).
            msg = AIMessage(content=action.reply or answer or raw)
            if thinking:
                msg.additional_kwargs["thinking"] = thinking
            return msg
        msg = AIMessage(content=answer or raw)
        if thinking:
            msg.additional_kwargs["thinking"] = thinking
        return msg

    @staticmethod
    def _repeat_note(messages: List[BaseMessage]) -> str | None:
        """Self-note short-circuit when the last two tool calls match.

        Measured 2026-09-28: Bonsai issued the same web_search 7× in one
        turn ("let me fetch X" in thinking, web_search again in acting)
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
        return (f"I already called {name} with these exact arguments and "
                f"have the results above. Do NOT call it again — answer "
                f"the user now from those results, in one short chat "
                f"sentence, no tool call.")

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
        msgs = self.llm.messages_for_terminal(text, history, cwd="", observation="")
        raw = await self._collect_raw(msgs, self._step_budget(), tools)
        return ChatResult(generations=[ChatGeneration(message=self._final_message(raw))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        """True per-token stream for the agent loop.

        Yields live content chunks, then ONE final chunk carrying the
        parsed ``tool_calls`` (aggregated by langchain into the AI
        message) and ``thinking`` in ``additional_kwargs``. Garble/echo
        retries stream transparently inside the same step.
        """
        from langchain_core.messages import AIMessageChunk
        from langchain_core.messages.ai import create_tool_call_chunk
        from langchain_core.outputs import ChatGenerationChunk

        def _cgc(msg):
            return ChatGenerationChunk(message=msg)

        text, history, observation = self._messages_to_llm_input(messages)
        note = self._repeat_note(messages)
        if note is not None:
            text = f"{text}\n{note}"
        tools = self._prompt_tools(text, observation)
        max_tokens = self._step_budget()
        msgs = self.llm.messages_for_terminal(text, history, cwd="", observation="")

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
            thinking, answer, action0 = self._parse_action(raw)
            nudge = self._needs_retry(raw, thinking, answer, action0)
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
        thinking, answer, action = self._parse_action(raw)
        extra: dict = {}
        if thinking:
            extra["thinking"] = thinking
        chunks = []
        if action and action.op != "done":
            import uuid

            name, args = self._builtin_call(action)
            chunks.append(create_tool_call_chunk(
                name=name,
                args=json.dumps(args),
                id=f"call_{uuid.uuid4().hex[:8]}",
                index=0,
            ))
        final_content = "" if live else (action.reply if action is not None and action.op == "done" else answer)
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
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(asyncio.run, self._agenerate_async(messages, **kwargs))
                return fut.result()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await self._agenerate_async(messages, **kwargs)
