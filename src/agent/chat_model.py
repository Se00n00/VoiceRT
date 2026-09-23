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
from src.tools.terminal import TERMINAL_TOOLS, is_degenerate, is_echo, parse_bare_tail, parse_terminal_action, parse_xml_action


class LocalChatModel(BaseChatModel):
    """Thin LangChain wrapper around :class:`LlmModel`."""

    llm: Any
    bound_tools: Optional[List[dict]] = None

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
                # System is handled via LlmModel's system_prompt; keep as history
                history.append({"role": "system", "content": m.content})
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
        tools = self.bound_tools
        if tools is None and str(getattr(getattr(self.llm, "config", None), "backend", "")).startswith("minicpm"):
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
        # the think→toolcall loop). Floor at 256, 320 for thinking MiniCPM.
        base_max = int(getattr(getattr(self.llm, "config", None), "max_tokens", 48) or 48)
        is_minicpm = str(getattr(getattr(self.llm, "config", None), "backend", "")).startswith("minicpm")
        return max(base_max, 320) if is_minicpm else max(base_max, 256)

    def _parse_action(self, raw: str):
        """Native envelope first, fuzzy aliases second, bare tail last."""
        thinking, answer = split_thinking(raw)
        action = parse_terminal_action(answer) if answer.strip() else None
        if action is None and answer.strip():
            action = parse_xml_action(answer)
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

    def _needs_retry(self, raw: str, thinking: str, answer: str) -> str | None:
        """One-retry nudge for garbled or echoed output, else None."""
        if is_degenerate(raw):
            return ("That reply was garbled. Answer again: ONLY one JSON "
                    "action or one short sentence.")
        if is_echo(thinking, answer):
            return ("Do not repeat your thinking as the reply. Reply with "
                    "EITHER one short chat sentence OR exactly one function "
                    "call, nothing else.")
        return None

    async def _stream_raw(self, msgs, max_tokens: int, tools, acc: dict):
        """Yield raw text pieces live; stash the full text in ``acc["raw"]``.

        Full text prefers whole-id re-decode (per-piece decodes can split
        BPE pairs), else joined pieces, else one blocking generate.
        """
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
                pieces: list[str] = []
                ids: list[int] = []
                live = False
                try:
                    async for tok in gen:
                        piece = str(getattr(tok, "piece", "") or "")
                        try:
                            ids.append(int(getattr(tok, "token_id", -1)))
                        except Exception:
                            pass
                        if piece:
                            pieces.append(piece)
                            live = True
                            yield piece
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
                        acc["raw"] = "".join(pieces)
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
            yield raw

    async def _collect_raw(self, msgs, max_tokens: int, tools) -> str:
        """Full raw text for one step, with one retry on garble/echo."""
        acc: dict = {}
        async for _ in self._stream_raw(msgs, max_tokens, tools, acc):
            pass
        raw = acc.get("raw", "")
        if not raw:
            return raw
        thinking, answer, _ = self._parse_action(raw)
        nudge = self._needs_retry(raw, thinking, answer)
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

    async def _agenerate_async(
        self, messages: List[BaseMessage], **kwargs
    ) -> ChatResult:
        text, history, observation = self._messages_to_llm_input(messages)
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
        tools = self._prompt_tools(text, observation)
        max_tokens = self._step_budget()
        msgs = self.llm.messages_for_terminal(text, history, cwd="", observation="")

        acc0: dict = {}
        # NOTE: nested async-generator drain (no return values allowed).
        live: bool | None = None
        agen = self._stream_raw(msgs, max_tokens, tools, acc0)
        async for piece in agen:
            # Generate-fallback legs yield one whole-text piece: keep their
            # "no token events" semantics, content goes on the final chunk.
            if live is None:
                live = bool(acc0.get("live"))
            if live:
                yield _cgc(AIMessageChunk(content=piece))
        raw = acc0.get("raw", "")
        if raw:
            thinking, answer, _ = self._parse_action(raw)
            nudge = self._needs_retry(raw, thinking, answer)
            if nudge is not None:
                acc1: dict = {}
                live2: bool | None = None
                agen2 = self._stream_raw(msgs + [{"role": "user", "content": nudge}],
                                         max_tokens, tools, acc1)
                async for piece in agen2:
                    if live2 is None:
                        live2 = bool(acc1.get("live"))
                    if live2:
                        yield _cgc(AIMessageChunk(content=piece))
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
