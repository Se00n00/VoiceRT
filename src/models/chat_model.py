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
from pydantic import BaseModel, ConfigDict, Field

from src.agent.budget import for_backend
from src.agent.mcp.specs import MCP_TOOL_DEFS, MCP_TOOL_NAMES
from src.agent.tool_router import (
    ALLOWED_OPS,
    ALL_TOOL_DEFS,
    current_ops,
    tools_for_request,
)
from src.models.llm import LlmModel, split_thinking
from src.prompts.chat import (
    ECHO_RETRY,
    GARBLED_RETRY,
    REPEAT_NOTE_TEMPLATE,
    UNPARSEABLE_RETRY,
)

class TerminalAction(BaseModel):
    """Validated single model step. MCP ops carry args; done carries reply."""

    model_config = ConfigDict(frozen=True)

    op: str = "done"
    command: str = ""
    path: str = ""
    text: str = ""
    reply: str = ""
    anchor: str = ""
    pattern: str = ""
    code: str = ""  # python_exec payload (own key: shell deny patterns must not see it)
    args: dict = Field(default_factory=dict)  # MCP-tool args (op in MCP_TOOL_NAMES)

    def as_dict(self) -> dict:
        out: dict = {"action": self.op}
        if self.command:
            out["command"] = self.command
        if self.path:
            out["path"] = self.path
        if self.text:
            out["text"] = self.text
        if self.reply:
            out["reply"] = self.reply
        if self.anchor:
            out["anchor"] = self.anchor
        if self.pattern:
            out["pattern"] = self.pattern
        if self.code:
            out["code"] = self.code
        for k, v in (self.args or {}).items():
            out.setdefault(str(k), v)
        return out


# Old/alias op names -> canonical MCP ops, checked BEFORE the
# terminal table. Multi-word keys first: several are prefixes of
# terminal keys in reverse ("readfile" contains "read"), so the
# terminal table would shadow them anywhere later.
_MCP_NORM_OPS = (
    ("typetext", "type_text"), ("type_text", "type_text"),
    ("presskey", "press_key"), ("press_key", "press_key"),
    ("hotkey", "press_key"),
    ("getactivewindow", "get_active_window"),
    ("get_active_window", "get_active_window"),
    ("activewindow", "get_active_window"),
    ("listwindows", "list_windows"), ("list_windows", "list_windows"),
    ("windows", "list_windows"),
    ("listdirectory", "list"), ("list_directory", "list"),
    ("listdir", "list"), ("ls", "list"), ("dir", "list"),
    ("listfiles", "list"),
    ("searchfiles", "glob"), ("search_files", "glob"),
    ("glob", "glob"),
    ("readfile", "read"), ("read_file", "read"),
    ("writefile", "write"), ("write_file", "write"),
    ("editfile", "edit"), ("edit_file", "edit"),
    ("movefile", "move"), ("move_file", "move"),
    ("move", "move"), ("rename", "move"),
    ("deletefile", "delete"), ("delete_file", "delete"),
    ("delete", "delete"), ("remove", "delete"),
    ("searchweb", "websearch"), ("search_web", "websearch"),
    ("websearch", "websearch"), ("searxng", "websearch"),
    ("web_search", "websearch"),
    ("openurl", "open_url"), ("open_url", "open_url"),
    ("openapp", "open_app"), ("open_app", "open_app"),
    ("launch", "open_app"),
    ("extractpage", "webfetch"), ("extract_page", "webfetch"),
    ("webfetch", "webfetch"), ("fetch", "webfetch"),
    ("downloadfile", "download_file"), ("download_file", "download_file"),
    ("download", "download_file"),
    ("browserclick", "browser_click"), ("browser_click", "browser_click"),
    ("browsertype", "browser_type"), ("browser_type", "browser_type"),
    ("browserscroll", "browser_scroll"),
    ("browser_scroll", "browser_scroll"),
    ("screenshot", "screenshot"), ("screencapture", "screenshot"),
    ("click", "click"),
    ("scroll", "scroll"),
    ("writetodos", "write_todos"), ("write_todos", "write_todos"),
    ("todos", "write_todos"), ("todo", "write_todos"),
    ("subagent", "task"), ("task", "task"),
    ("ask", "question"), ("question", "question"),
    ("skill", "skill"),
)


def _norm_op(op: str) -> str:
    """Fuzzy op match for models that paraphrase the schema.

    MiniCPM emits near-misses (``run_command``, ``list_files``) instead of
    the exact ops. Substring match on the normalized name, most-specific
    first. Returns "" when nothing matches.
    """
    o = re.sub(r"[^a-z]+", "", str(op or "").lower())
    if not o:
        return ""
    for key, mapped in _MCP_NORM_OPS:
        if key in o:
            return mapped
    for key, mapped in (
        # python_exec first: "exec" is a substring of "pythonexec" and
        # would shadow it anywhere later in this table.
        ("pythonexec", "python_exec"), ("python", "python_exec"),
        # bash unifies exec/exec_bg/poll: background + poll are flags
        # on one tool now (execbg/poll handled with extra args below).
        ("runcommand", "bash"), ("execute", "bash"), ("shell", "bash"),
        ("bash", "bash"), ("command", "bash"), ("run", "bash"),
        ("execbg", "bash"), ("background", "bash"),
        ("exec", "bash"),
        ("poll", "bash"), ("wait", "bash"), ("check", "bash"),
        ("job", "bash"),
        ("listfiles", "list"), ("list", "list"), ("ls", "list"),
        ("dir", "list"),
        ("webfetch", "webfetch"), ("fetch", "webfetch"),
        ("extract", "webfetch"),
        ("websearch", "websearch"), ("searxng", "websearch"),
        ("glob", "glob"),
        ("read", "read"), ("cat", "read"), ("open", "read"),
        ("show", "read"),
        ("write", "write"), ("save", "write"), ("create", "write"),
        ("edit", "edit"), ("patch", "edit"), ("replace", "edit"),
        ("move", "move"), ("rename", "move"),
        ("delete", "delete"), ("remove", "delete"),
        ("grep", "grep"), ("search", "grep"), ("find", "grep"),
        ("download", "download_file"),
        ("done", "done"), ("reply", "done"), ("answer", "done"),
        ("chat", "done"), ("speak", "done"), ("say", "done"),
        ("respond", "done"),
    ):
        if key in o:
            return mapped
    return ""


# Legacy payload keys -> canonical spec keys (models emit near-miss
# shapes: {"path": ...} for url, {"text": ...} for content, ...).
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "command": ("command", "cmd", "script"),
    "job_id": ("job_id", "job", "job-id"),
    "path": ("path", "file", "dir", "root", "directory", "folder"),
    "content": ("content", "text", "value", "body"),
    "pattern": ("pattern", "regex", "query", "q"),
    "code": ("code", "text", "script"),
    "url": ("url", "link", "path", "href"),
    "query": ("query", "pattern", "q", "text"),
    "root": ("root", "path", "dir"),
    "src": ("src", "source", "from", "path"),
    "dst": ("dst", "dest", "destination", "to"),
    "selector": ("selector", "css", "target"),
    "text": ("text", "content", "value", "body"),
    "old_string": ("old_string", "anchor", "old", "before", "search"),
    "new_string": ("new_string", "text", "content", "after"),
}

_BG_HINTS = ("execbg", "background", "nohup", "daemon")
_POLL_HINTS = ("poll", "wait", "check", "job")


def _payload(obj: dict, *keys: str, limit: int = 4000) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:limit]
        if v is not None and not isinstance(v, (dict, list)) and str(v).strip():
            return str(v).strip()[:limit]
    return ""


def _from_obj(obj: dict, strict: bool) -> TerminalAction | None:
    """Build an action from a parsed JSON dict, or None."""
    if not isinstance(obj, dict):
        return None
    raw_op = str(obj.get("action", "")).strip().lower()
    if strict:
        if raw_op not in ALLOWED_OPS:
            return None
        op = raw_op
    else:
        op = _norm_op(raw_op)
        if not op:
            return None
    if op == "done":
        reply = _payload(obj, "reply", "message", "answer", limit=2000)
        if not reply:
            reply = _payload(obj, "text", limit=2000)
        return TerminalAction(op="done", reply=reply)
    if op == "write_todos":
        # Middleware-owned planner (TodoListMiddleware): args pass
        # through verbatim so the tool node executes it by name.
        args = {str(k): v for k, v in obj.items() if k != "action"}
        return TerminalAction(op="write_todos", args=args)
    if op in MCP_TOOL_NAMES:
        # MCP tool: args from the declared spec keys (+ legacy aliases).
        spec = next((t for t in MCP_TOOL_DEFS if t["name"] == op), {})
        params = (spec.get("parameters", {}) or {})
        props = params.get("properties", {}) or {}
        required = params.get("required", []) or []
        args: dict = {}
        for k in props:
            keys = (k,) + tuple(a for a in _KEY_ALIASES.get(k, ())
                                if a != k)
            v = _payload(obj, *keys, limit=10000 if k in (
                "content", "text", "code") else 4000)
            if v:
                args[k] = v
            else:
                raw = obj.get(k)
                if isinstance(raw, (int, float, bool)):
                    args[k] = raw
        norm = re.sub(r"[^a-z]+", "", raw_op)
        if op == "bash":
            if not strict and any(h in norm for h in _BG_HINTS):
                args.setdefault("background", True)
            if not strict and norm in _POLL_HINTS and args.get("command") \
                    and not args.get("job_id"):
                args["job_id"] = args.pop("command")
            if not strict and not args.get("command") and any(
                    h in norm for h in _POLL_HINTS):
                jid = _payload(obj, "command", "cmd", "job",
                               "job_id", "job-id")
                if jid:
                    args["job_id"] = jid
            if not args.get("command") and args.get("job_id"):
                pass
            elif not args.get("command", "").strip():
                cmd = _payload(obj, "command", "cmd", "script")
                if cmd:
                    args["command"] = cmd
                else:
                    return None
            bg = args.get("background")
            if isinstance(bg, str):
                args["background"] = bg.strip().lower() in (
                    "1", "true", "yes")
        for k in required:
            if k not in args or not str(args[k]).strip():
                return None
        return TerminalAction(op=op, args=args)
    return None


def parse_terminal_action(text: str) -> TerminalAction | None:
    """Parse ONE JSON action from model text. None = plain chat. Never raises.

    Exact schema first (Qwen path, unchanged); fuzzy op/payload aliases
    second (MiniCPM paraphrases like ``run_command`` / ``list_files``).
    """
    if not text or not text.strip():
        return None
    raw = text.strip()
    candidates = []
    if raw.startswith("{"):
        candidates.append(raw)
    candidates.extend(m.group(0) for m in _JSON_RE.finditer(raw))
    parsed = []
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    for obj in parsed:
        action = _from_obj(obj, strict=True)
        if action is not None:
            return action
    for obj in parsed:
        action = _from_obj(obj, strict=False)
        if action is not None:
            return action
    return None


_FUNC_RE = re.compile(
    r'<function\s+name="([^"]+)"\s*>(.*?)</function\s*>',
    re.DOTALL | re.IGNORECASE)
_PARAM_RE = re.compile(
    r'<param\s+name="([^"]+)"\s*>(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</param\s*>',
    re.DOTALL | re.IGNORECASE)

# MiniCPM5 native XML names -> our ops (it invents near-misses like run/speak)
_XML_OPS = {
    "bash": "bash", "exec": "bash", "run": "bash", "shell": "bash",
    "command": "bash", "execute": "bash", "exec_bg": "bash",
    "poll": "bash",
    "read": "read", "cat": "read", "open": "read", "show": "read",
    "write": "write", "save": "write", "create": "write",
    "edit": "edit", "patch": "edit",
    "grep": "grep", "search": "grep",
    "glob": "glob",
    "move": "move", "rename": "move",
    "delete": "delete", "remove": "delete",
    "webfetch": "webfetch", "fetch": "webfetch",
    "websearch": "websearch", "searxng": "websearch",
    "list": "list", "ls": "list", "dir": "list",
    "python_exec": "python_exec", "python": "python_exec",
    "open_url": "open_url", "open_app": "open_app",
    "browser_click": "browser_click", "browser_type": "browser_type",
    "browser_scroll": "browser_scroll", "download_file": "download_file",
    "screenshot": "screenshot", "click": "click",
    "type_text": "type_text", "press_key": "press_key",
    "scroll": "scroll", "get_active_window": "get_active_window",
    "list_windows": "list_windows",
    "task": "task", "write_todos": "write_todos",
    "question": "question", "skill": "skill",
    "done": "done", "reply": "done", "answer": "done", "chat": "done",
    "speak": "done", "say": "done", "respond": "done",
}


def parse_xml_action(text: str) -> TerminalAction | None:
    """Parse MiniCPM5-native ``<function><param>`` tool calls. Never raises."""
    if not text or "<function" not in text.lower():
        return None
    for fm in _FUNC_RE.finditer(text):
        name = fm.group(1).strip().lower()
        op = _XML_OPS.get(name)
        if op is None:
            continue
        params: dict = {}
        for pm in _PARAM_RE.finditer(fm.group(2)):
            key = pm.group(1).strip().lower()
            val = pm.group(2) if pm.group(2) is not None else (pm.group(3) or "")
            params[key] = val.strip()[:4000]
        if op == "bash" and name in ("exec_bg", "background"):
            params.setdefault("background", "true")
        if op == "bash" and name in ("poll", "job"):
            jid = (params.get("command") or params.get("job")
                   or params.get("job_id") or "")
            if jid.strip():
                params = {"action": "bash", "job_id": jid[:64],
                          **{k: v for k, v in params.items()
                             if k not in ("command", "job", "job_id")}}
                action = _from_obj(params, strict=False)
                if action is not None:
                    return action
                continue
        obj = {"action": op}
        obj.update(params)
        action = _from_obj(obj, strict=False)
        if action is not None:
            return action
        continue
    return None


def _repair_json(raw: str) -> str:
    """Append missing closing braces/brackets, or return "" when hopeless.

    Models (even 27B ones, measured 2026-09-28) sometimes drop the final
    ``}`` of an otherwise valid args object — the whole turn then dies
    as chat because one brace is missing. Only strings starting with
    ``{`` qualify; braces inside string literals don't count; at most
    two closers are appended (deeper damage stays unparsed). Returns
    parseable JSON or "" (the repaired candidate must itself decode —
    garbage in stays out). Never raises.
    """
    s = (raw or "").strip()
    if not s.startswith("{"):
        return ""
    stack: list[str] = []
    in_str = False
    esc = False
    pairs = {"{": "}", "[": "]"}
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                return ""  # stray closer — not a truncation, don't guess
    if in_str or len(stack) > 2:
        return ""
    cand = s + "".join(reversed(stack))
    try:
        if isinstance(json.loads(cand), dict):
            return cand
    except Exception:
        pass
    return ""


_BARE_RE = re.compile(
    r'name="([^"]+)"\s*>\s*([^<>\n]*?)(?=\s*name="[^"]+"\s*>|\s*$)',
    re.DOTALL | re.IGNORECASE | re.MULTILINE)


_GEMMA_RE = re.compile(
    r"<\|tool_call>call:([A-Za-z0-9_.\-]+)(.*?)<tool_call\|>",
    re.DOTALL)


def parse_gemma_action(text: str) -> TerminalAction | None:
    """Parse Gemma-4 native ``<|tool_call>call:name{args}<tool_call|>`` blocks.

    Server-side template rendering (llama.cpp) emits these when OpenAI
    tools are passed; the leg re-serializes structured calls into the
    same grammar (see :mod:`src.models.bonsai_llamacpp`). The arg span
    runs to the block marker so nested objects survive; it must parse
    as one JSON object. First valid block wins (single-action
    contract; multi-call grading lives in the eval extractor).
    Unknown op names return None. Never raises.
    """
    if not text or "<|tool_call>" not in text:
        return None
    for m in _GEMMA_RE.finditer(text):
        name = (m.group(1) or "").strip()
        op = _norm_op(name)
        if not op:
            continue
        args = {}
        raw_args = (m.group(2) or "").strip()
        if raw_args:
            try:
                parsed = json.loads(raw_args)
            except Exception:
                fixed = _repair_json(raw_args)
                try:
                    parsed = json.loads(fixed) if fixed else None
                except Exception:
                    continue
                if parsed is None:
                    continue
            if not isinstance(parsed, dict):
                continue
            args = parsed
        obj = {"action": op}
        obj.update({str(k): v for k, v in args.items()})
        action = _from_obj(obj, strict=False)
        if action is not None:
            return action
    return None


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

_FUNCALL_RE = re.compile(
    r"<start_function_call>call:([A-Za-z0-9_.\-]+)(.*?)<end_function_call>",
    re.DOTALL)


_SINGLE_ARG_OPS = {
    # op -> arg key when the model emits a bare (braceless) value span
    "bash": "command",
    "read": "path", "list": "path", "webfetch": "url",
    "grep": "pattern", "websearch": "query", "glob": "pattern",
    "move": "src", "delete": "path",
    "skill": "name", "task": "command", "question": "question",
}


def _funcgemma_args(op: str, span: str) -> dict:
    """Normalize a FunctionGemma arg span to a JSON-ish dict.

    Observed shapes (measured 2026-09-28, 270M):
    - ``{path:`.`.}`` — braces, backtick quotes, trailing dots
    - ``{path:<escape>server.py<escape>}`` — <escape> entities as quotes
    - ``https://example.com<escape>`` — bare value, no braces at all
    Returns {} when nothing salvageable. Never raises.
    """
    t = (span or "").strip()
    if not t:
        return {}
    t = t.replace("<escape>", '"').replace("`", '"').strip()
    t = re.sub(r"\.+([\"'}}\]]?)\s*$", r"\1", t)
    if t.startswith("{"):
        # bare JS-ish keys are not JSON: {path:...} -> {"path":...}
        t = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', t)
        for cand in (t, _repair_json(t)):
            if not cand:
                continue
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        # last resort: {key: anything} captured as a plain string
        m = re.match(r"^\{\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?\s*:\s*(.+?)\s*\}?$", t, re.DOTALL)
        if m:
            return {m.group(1): m.group(2).strip('\'" ')}
        return {}
    key = _SINGLE_ARG_OPS.get(op, "")
    if key:
        return {key: t.strip('" {}')}
    return {}


def parse_toolcall_dict(text: str) -> TerminalAction | None:
    """Parse LangChain/OpenAI tool-call dicts emitted as plain text.

    Measured 2026-09-28 (Qwen3-1.7B subtasks): the model sometimes emits
    the call it just made (or wants to make) as bare JSON
    ``{"name": "list", "arguments": {"path": "."}}`` instead of any
    envelope. Without this the text falls through to chat and the turn
    ends with call-shaped garbage as its reply. Never raises.
    """
    if not text or '"name"' not in text:
        return None
    try:
        start = text.index('{"name"')
    except ValueError:
        try:
            start = text.index("{\"name\"")
        except ValueError:
            return None
    # balanced-brace scan from the first {"name" occurrence
    depth, instr, esc = 0, False, False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end <= start:
        return None
    try:
        obj = json.loads(text[start:end])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name", "") or "")
    args = obj.get("arguments", {})
    if not isinstance(args, dict):
        return None
    op = _norm_op(name)
    if not op:
        return None
    merged = {"action": op}
    merged.update({str(k): v for k, v in args.items()})
    return _from_obj(merged, strict=False)


def parse_functiongemma_action(text: str) -> TerminalAction | None:
    """Parse FunctionGemma-270M native call envelopes.

    The model emits its own grammar (never our JSON)::

        <start_function_call>call:list{path:`.`}<end_function_call>

    then rambles on with imagined ``<start_function_response>`` blocks —
    only the FIRST call block counts (callers stop generation at
    ``<end_function_call>`` anyway, so longer chains never materialize).
    Unknown op names return None. Never raises.
    """
    if not text or "<start_function_call>" not in text:
        return None
    for m in _FUNCALL_RE.finditer(text):
        name = (m.group(1) or "").strip()
        op = _norm_op(name)
        if not op:
            continue
        args = _funcgemma_args(op, m.group(2) or "")
        obj = {"action": op}
        obj.update({str(k): v for k, v in args.items()})
        action = _from_obj(obj, strict=False)
        if action is not None:
            return action
    # Unclosed trailing block: generation often ends (EOS/stop) without
    # the close marker — the call runs to end of string.
    m = re.search(r"<start_function_call>call:([A-Za-z0-9_.\-]+)(.*)$",
                  text, re.DOTALL)
    if m:
        op = _norm_op((m.group(1) or "").strip())
        if op:
            args = _funcgemma_args(op, m.group(2) or "")
            obj = {"action": op}
            obj.update({str(k): v for k, v in args.items()})
            return _from_obj(obj, strict=False)
    return None


def parse_bare_tail(text: str) -> TerminalAction | None:
    """Parse the 1B tail quirk: envelope openers dropped, bare pairs left.

    MiniCPM5-1B sometimes emits ``name="list"> name="path">.`` — the
    ``<function``/``<param`` openers missing, then EOS. The pairs are
    recovered into an obj and validated by :func:`_from_obj`, so only
    well-formed ops/params survive. Refuses text containing ``<`` or
    ``{`` (well-formed XML/JSON belong to the other parsers). Never raises.
    """
    if not text or not text.strip():
        return None
    if "<" in text or "{" in text:
        return None
    ms = _BARE_RE.findall(text)
    if not ms:
        return None
    first = ms[0][0].strip().lower()
    op = _norm_op(first) or _XML_OPS.get(first)
    if not op:
        return None
    rest = ms[1:]
    obj: dict = {"action": op}
    for k, v in rest:
        obj.setdefault(k.strip().lower(), v.strip()[:4000])
    return _from_obj(obj, strict=False)


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


class LocalChatModel(BaseChatModel):
    """Thin LangChain wrapper around :class:`LlmModel`."""

    llm: Any
    bound_tools: Optional[List[dict]] = None
    # no_tools: prompt WITHOUT tool specs (summarizer calls). A model
    # handed tools alongside a "respond ONLY with ..." instruction may
    # emit a tool call instead of the requested text.
    no_tools: bool = False
    # use_mcp_fs: the MCP tools (read/write/edit with {path, content,
    # ...} args) are attached and silently WIN any name collision over
    # the deepagents built-ins. Set by the harness builders that load
    # the MCP extras; _builtin_call then emits the MCP arg schema so
    # terminal read/write/edit actually execute instead of dying.
    use_mcp_fs: bool = False

    def __init__(self, llm: Any, **kwargs):
        super().__init__(llm=llm, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "local-qwen"

    def bind_tools(self, tools, **kwargs):
        # Only dict specs pass through untouched. StructuredTools bound by
        # the agent stay execution-side: the prompt half is decided by
        # backend in _prompt_tools (native TERMINAL_TOOLS for bonsai,
        # JSON-preamble path otherwise) — rendering foreign schemas into
        # the chat template would silently change model behavior.
        names = {getattr(t, "name", "") for t in tools or []}
        mcp = bool(names & set(MCP_TOOL_NAMES))
        defs = [t for t in tools or []
                if isinstance(t, dict) and "name" in t]
        return LocalChatModel(llm=self.llm, bound_tools=defs or None,
                              use_mcp_fs=self.use_mcp_fs or mcp)

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
                    return picked
            # Route: narrow the 9 native tool definitions to what this
            # step plausibly needs (small models drown in 9). The harness
            # appends a "CWD: ... SHELL: ..." trailer to every turn — strip
            # it for routing (else "shell" forces exec into every set).
            # None (or another backend) keeps the full set / preamble path.
            routable = re.sub(r"\nCWD: .*?(\s+SHELL: \w+)?\s*$", "", text)
            tools = tools_for_request(routable, observation)
        return tools

    def _needs_catalogue(self) -> bool:
        """Prose op catalogue wanted only without native tool specs.

        bonsai receives real function specs beside the messages — pure
        English suffices. Qwen's template may ignore the specs kwarg,
        so it keeps the catalogue. Unknown backends keep it (fail open).
        ..
        """
        backend = str(getattr(getattr(self.llm, "config", None), "backend", ""))
        return backend not in ("bonsai",)

    def _step_budget(self) -> int:
        # Think + tool call must fit in ONE step (the old harness learned
        # this the hard way: 48/160 strangled the model mid-think, killing
        # the think→toolcall loop). Floors come from the budget table
        # (src/agent/budget.py:BUDGETS) so pack() and the loop agree.
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
            # Bare OpenAI/LangChain tool-call dicts as text (Qwen subtasks).
            action = parse_toolcall_dict(answer)
        if action is None and answer.strip():
            action = parse_terminal_action(answer)
        if action is None and answer.strip():
            action = parse_xml_action(answer)
        if action is None:
            action = parse_gemma_action(raw)
        if action is None:
            action = parse_functiongemma_action(raw)
        if action is None:
            action = parse_toolcall_dict(raw)
        if action is None:
            action = parse_terminal_action(raw)
        if action is None:
            action = parse_xml_action(raw)
        if action is None and answer.strip():
            action = parse_bare_tail(answer)
        if action is None:
            action = parse_bare_tail(raw)
        return thinking, answer, action

    def _builtin_call(self, action) -> tuple:
        """Prompt op -> execution tool (name, args).

        All tools live in MCP now, so the prompt op IS the execution
        name; this only normalizes legacy payload keys (exec/read/write
        with path/text args, deepagents built-ins with file_path args)
        into the canonical MCP arg schema. Anything unknown passes
        through unchanged (the tool node reports it and the model
        recovers).
        """
        d = action.as_dict()
        op = action.op
        args = dict(getattr(action, "args", None) or {})

        def _pick(*keys: str, default: str = "") -> str:
            for k in keys:
                v = args.get(k, "") if isinstance(args, dict) else ""
                if isinstance(v, str) and v.strip():
                    return v
                v = d.get(k, "")
                if isinstance(v, str) and v.strip():
                    return v
            return default

        if op == "bash":
            out: dict = {}
            cmd = _pick("command", "cmd", "script")
            jid = _pick("job_id", "job", "job-id")
            if jid and not cmd:
                out["job_id"] = jid
            elif cmd:
                out["command"] = cmd
            else:
                return op, dict(args)
            bg = args.get("background", d.get("background", False))
            if isinstance(bg, str):
                bg = bg.strip().lower() in ("1", "true", "yes")
            if bg:
                out["background"] = True
            return "bash", out
        if op in ("exec", "execute", "exec_bg", "poll"):
            # legacy deepagents/terminal names -> unified bash.
            cmd = _pick("command", "cmd", "script", "job", "job_id")
            if op in ("exec_bg",) or "background" in d:
                return "bash", {"command": cmd, "background": True}
            if op == "poll" or (cmd.startswith("job-") and op != "exec"):
                return "bash", {"job_id": cmd}
            return "bash", {"command": cmd}
        if op in ("read", "read_file"):
            return "read", {"path": _pick("path", "file", "file_path", "dir")}
        if op in ("write", "write_file"):
            return "write", {"path": _pick("path", "file", "file_path"),
                             "content": _pick("content", "text")}
        if op in ("edit", "edit_file"):
            return "edit", {"path": _pick("path", "file", "file_path"),
                            "old_string": _pick("old_string", "anchor", "old"),
                            "new_string": _pick("new_string", "text", "content")}
        if op in ("list", "ls", "list_directory"):
            return "list", {"path": _pick("path", "file", "dir") or "."}
        if op in ("glob", "search_files"):
            return "glob", {"pattern": _pick("pattern", "query"),
                            "path": _pick("path", "root", "dir") or "."}
        if op == "grep":
            return "grep", {"pattern": _pick("pattern", "query", "regex"),
                            "path": _pick("path", "dir", "root") or "."}
        if op in ("move", "move_file"):
            return "move", {"src": _pick("src", "source", "from_", "from", "path"),
                            "dst": _pick("dst", "dest", "to")}
        if op in ("delete", "delete_file"):
            return "delete", {"path": _pick("path", "file", "dir")}
        if op in ("webfetch", "fetch", "extract_page"):
            return "webfetch", {"url": _pick("url", "path", "link")}
        if op in ("websearch", "searxng", "web_search", "search_web"):
            return "websearch", {"query": _pick("query", "pattern", "q", "text")}
        if op == "python_exec":
            return "python_exec", {"code": _pick("code", "text", "script")}
        if op == "write_todos":
            # Middleware-owned planner: verbatim args, exact name.
            return "write_todos", dict(args) if isinstance(args, dict) else {}
        return op, {k: v for k, v in d.items() if k != "action"}

    @staticmethod
    def _looks_like_call(raw: str) -> bool:
        """Unparsed text that still smells like a tool attempt."""
        t = str(raw or "")
        if ("<|tool_call>" in t or "<function" in t
                or "<start_function_call>" in t
                or '"action"' in t or "'action'" in t):
            return True
        # Bare OpenAI/LangChain dicts (incl. invented tools like "count"):
        # retry as a parse nudge rather than ending the turn as chat.
        return '"name"' in t and '"arguments"' in t

    def _needs_retry(self, raw: str, thinking: str, answer: str,
                     action=None) -> str | None:
        """One-retry nudge for garbled, echoed, or unparsable-call output."""
        if is_degenerate(raw):
            return GARBLED_RETRY
        if is_echo(thinking, answer):
            return ECHO_RETRY
        if action is None and self._looks_like_call(raw):
            # Measured 2026-09-28: a 27B model emitted a call-shaped reply
            # the parsers rejected, and the turn silently ended as chat.
            # One nudge recovers it instead of dropping the user's task.
            return UNPARSEABLE_RETRY
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
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(asyncio.run, self._agenerate_async(messages, **kwargs))
                return fut.result()

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await self._agenerate_async(messages, **kwargs)
