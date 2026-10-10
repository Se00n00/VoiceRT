"""Single-model terminal action schema + safety policy.

Same contract as the browser tools: the ONE Qwen model either chats
(plain text, spoken back) or emits exactly one JSON action per step::

    {"action": "exec", "command": "ls -la"}
    {"action": "read", "path": "server.py"}
    {"action": "write", "path": "notes.txt", "text": "hello"}
    {"action": "list", "path": "."}
    {"action": "done", "reply": "Listed the repo."}

This module is pure (no weights, no yaml). File/shell execution lives in
deepagents' backend tools plus the MCP extras
(:mod:`src.agent.mcp.tools`); the policy here decides deny / confirm /
allow. Your shell, your risk: denies cover system-breakers, everything
mutating needs an explicit yes.
"""
import json
import re

from pydantic import BaseModel, ConfigDict, Field

from src.agent.mcp.defs import MCP_TOOL_DEFS, MCP_TOOL_NAMES

__all__ = [
    "ALLOWED_OPS",
    "TERMINAL_TOOLS",
    "TerminalAction",
    "parse_terminal_action",
    "parse_xml_action",
    "parse_bare_tail",
    "parse_gemma_action",
    "parse_functiongemma_action",
    "parse_toolcall_dict",
    "check_policy",
    "is_denied",
    "needs_confirm",
    "is_degenerate",
    "is_echo",
    "route_tool_ops",
    "tools_for_request",
]

ALLOWED_OPS = ("exec", "exec_bg", "poll", "read", "write", "edit", "grep", "list", "python_exec", "fetch", "searxng", "done") + MCP_TOOL_NAMES


# Native tool definitions for models with template-level tool support
# (MiniCPM5: its chat template renders these into <tools> XML when passed
# as `tools=` to apply_chat_template). Same eight ops as the JSON schema.
TERMINAL_TOOLS = [
    {"name": "exec",
     "description": "Run a bash shell command.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]}},
    {"name": "exec_bg",
     "description": "Start a long-running shell command in the background.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]}},
    {"name": "poll",
     "description": "Poll a background job started by exec_bg.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string",
                                               "description": "job id like job-1"}},
                    "required": ["command"]}},
    {"name": "read",
     "description": "Read a text file.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}},
    {"name": "write",
     "description": "Write text to a file.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "text": {"type": "string"}},
                    "required": ["path", "text"]}},
    {"name": "edit",
     "description": "Replace anchor text verbatim in a file.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "anchor": {"type": "string"},
                                   "text": {"type": "string"}},
                    "required": ["path", "anchor", "text"]}},
    {"name": "grep",
     "description": "Grep a pattern across files.",
     "parameters": {"type": "object",
                    "properties": {"pattern": {"type": "string"},
                                   "path": {"type": "string"}},
                    "required": ["pattern"]}},
    {"name": "list",
     "description": "List directory entries.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}}}},
    {"name": "python_exec",
     "description": "Run Python code in an isolated container.",
     "parameters": {"type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]}},
    {"name": "fetch",
     "description": "Fetch a web page as text.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string",
                                            "description": "page URL"}},
                    "required": ["path"]}},
    {"name": "searxng",
     "description": "Web search, returns titles/URLs/snippets.",
     "parameters": {"type": "object",
                    "properties": {"pattern": {"type": "string",
                                               "description": "search query"}},
                    "required": ["pattern"]}},
]

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

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
_DENY_RE = re.compile("|".join(f"(?:{p})" for p in _DENY_RES))

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
_CONFIRM_CMD_RE = re.compile("|".join(f"(?:{p})" for p in _CONFIRM_RES))


class TerminalAction(BaseModel):
    """Validated single terminal step. edit carries anchor, grep carries pattern."""

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


# MCP-tool aliases, checked BEFORE the terminal table: several are
# prefixes of terminal keys in reverse ("readfile" contains "read"),
# so the terminal table would shadow them anywhere later.
_MCP_NORM_OPS = (
    ("typetext", "type_text"), ("type_text", "type_text"),
    ("presskey", "press_key"), ("press_key", "press_key"),
    ("hotkey", "press_key"),
    ("getactivewindow", "get_active_window"),
    ("get_active_window", "get_active_window"),
    ("activewindow", "get_active_window"),
    ("listwindows", "list_windows"), ("list_windows", "list_windows"),
    ("windows", "list_windows"),
    ("listdirectory", "list_directory"), ("list_directory", "list_directory"),
    ("listdir", "list_directory"),
    ("searchfiles", "search_files"), ("search_files", "search_files"),
    ("readfile", "read_file"), ("read_file", "read_file"),
    ("writefile", "write_file"), ("write_file", "write_file"),
    ("editfile", "edit_file"), ("edit_file", "edit_file"),
    ("movefile", "move_file"), ("move_file", "move_file"),
    ("move", "move_file"),
    ("deletefile", "delete_file"), ("delete_file", "delete_file"),
    ("delete", "delete_file"), ("remove", "delete_file"),
    ("searchweb", "search_web"), ("search_web", "search_web"),
    ("openurl", "open_url"), ("open_url", "open_url"),
    ("openapp", "open_app"), ("open_app", "open_app"),
    ("launch", "open_app"),
    ("extractpage", "extract_page"), ("extract_page", "extract_page"),
    ("downloadfile", "download_file"), ("download_file", "download_file"),
    ("download", "download_file"),
    ("browserclick", "browser_click"), ("browser_click", "browser_click"),
    ("browsertype", "browser_type"), ("browser_type", "browser_type"),
    ("browserscroll", "browser_scroll"),
    ("browser_scroll", "browser_scroll"),
    ("screenshot", "screenshot"), ("screencapture", "screenshot"),
    ("click", "click"),
    ("scroll", "scroll"),
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
        ("runcommand", "exec"), ("execute", "exec"), ("shell", "exec"),
        ("bash", "exec"), ("command", "exec"), ("run", "exec"),
        # exec_bg before exec: "exec" is a substring of "execbg" and
        # would shadow it (same bug class as above; caught by eval).
        ("execbg", "exec_bg"), ("background", "exec_bg"),
        ("exec", "exec"),
        ("poll", "poll"), ("wait", "poll"), ("check", "poll"),
        ("listfiles", "list"), ("list", "list"), ("ls", "list"),
        ("dir", "list"),
        ("fetch", "fetch"), ("webfetch", "fetch"),
        ("searxng", "searxng"), ("websearch", "searxng"),
        ("read", "read"), ("cat", "read"), ("open", "read"),
        ("show", "read"),
        ("write", "write"), ("save", "write"), ("create", "write"),
        ("edit", "edit"), ("patch", "edit"), ("replace", "edit"),
        ("grep", "grep"), ("search", "grep"), ("find", "grep"),
        ("done", "done"), ("reply", "done"), ("answer", "done"),
        ("chat", "done"), ("speak", "done"), ("say", "done"),
        ("respond", "done"),
    ):
        if key in o:
            return mapped
    return ""


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
    if op in MCP_TOOL_NAMES:
        # MCP-category tool: args straight from the declared spec keys.
        spec = next((t for t in MCP_TOOL_DEFS if t["name"] == op), {})
        params = (spec.get("parameters", {}) or {})
        props = params.get("properties", {}) or {}
        required = params.get("required", []) or []
        args: dict = {}
        for k in props:
            v = _payload(obj, k, limit=10000 if k in (
                "content", "text", "code") else 4000)
            if v:
                args[k] = v
            else:
                raw = obj.get(k)
                if isinstance(raw, (int, float, bool)):
                    args[k] = raw
        for k in required:
            if k not in args or not str(args[k]).strip():
                return None
        return TerminalAction(op=op, args=args)
    # per-op payload extraction to avoid cross-contamination (e.g. edit text
    # becoming command)
    command = ""
    if op in ("exec", "exec_bg", "poll"):
        command = _payload(obj, "command", "cmd", "script")
        if op == "poll" and not command:
            command = _payload(obj, "job", "job_id")
    path = ""
    if op in ("read", "write", "edit", "grep", "list", "fetch"):
        path = _payload(obj, "path", "file", "dir", "url", "link", limit=1024)
    body = ""
    if op in ("write", "edit"):
        body = _payload(obj, "text", "content", "value", "body", limit=20000)
    if op == "python_exec":
        body = _payload(obj, "code", "text", "script", limit=20000)
    reply = _payload(obj, "reply", "message", "answer", limit=2000)
    if op == "done" and not reply:
        # Model-invented answer vehicle: {"name": "answer",
        # "arguments": {"text": "..."}} — the payload key is "text".
        reply = _payload(obj, "text", limit=2000)
    anchor = _payload(obj, "anchor", "old", "before", "search", limit=10000) if op == "edit" else ""
    pattern = _payload(obj, "pattern", "regex", "query", limit=500) if op in ("grep", "searxng") else ""
    if op == "list" and command and not path:
        # "list the files" via an explicit shell command: run it.
        op = "exec"
        command = _payload(obj, "command", "cmd", "script")
    if op in ("exec", "exec_bg", "poll") and not command.strip():
        return None
    if op in ("read", "write", "edit") and not path.strip():
        return None
    if op == "write" and not body:
        return None
    if op == "edit" and (not body or not anchor):
        return None
    if op == "grep" and not pattern.strip():
        return None
    if op == "fetch" and not path.strip():
        return None
    if op == "searxng" and not pattern.strip():
        return None
    if op == "python_exec" and not body.strip():
        return None
    return TerminalAction(op=op, command=command, path=path,
                          text="" if op == "python_exec" else body,
                          reply=reply, anchor=anchor, pattern=pattern,
                          code=body if op == "python_exec" else "")


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
    "exec": "exec", "run": "exec", "shell": "exec", "bash": "exec",
    "command": "exec", "execute": "exec", "exec_bg": "exec_bg",
    "poll": "poll",
    "read": "read", "cat": "read", "open": "read", "show": "read",
    "write": "write", "save": "write", "create": "write",
    "edit": "edit", "patch": "edit",
    "grep": "grep", "search": "grep",
    "fetch": "fetch", "webfetch": "fetch",
    "searxng": "searxng", "websearch": "searxng",
    "list": "list", "ls": "list", "dir": "list",
    "python_exec": "python_exec", "python": "python_exec",
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
        if op == "exec":
            cmd = (params.get("command") or params.get("cmd")
                   or params.get("text") or "")
            if cmd.strip():
                return TerminalAction(op="exec", command=cmd[:4000])
        elif op == "exec_bg":
            cmd = (params.get("command") or params.get("cmd") or "")
            if cmd.strip():
                return TerminalAction(op="exec_bg", command=cmd[:4000])
        elif op == "poll":
            jid = (params.get("command") or params.get("job") or "")
            if jid.strip():
                return TerminalAction(op="poll", command=jid[:64])
        elif op == "read":
            path = params.get("path") or params.get("file") or ""
            if path.strip():
                return TerminalAction(op="read", path=path[:1024])
        elif op == "edit":
            path = params.get("path") or params.get("file") or ""
            anchor = params.get("anchor") or params.get("old") or ""
            body = params.get("text") or params.get("content") or ""
            if path.strip() and anchor and body:
                return TerminalAction(op="edit", path=path[:1024],
                                      anchor=anchor[:10000], text=body[:20000])
        elif op == "grep":
            pat = params.get("pattern") or params.get("query") or ""
            path = params.get("path") or ""
            if pat.strip():
                return TerminalAction(op="grep", pattern=pat[:500],
                                      path=path[:1024])
        elif op == "write":
            path = params.get("path") or params.get("file") or ""
            body = params.get("text") or params.get("content") or ""
            if path.strip() and body:
                return TerminalAction(op="write", path=path[:1024],
                                      text=body[:20000])
        elif op == "list":
            return TerminalAction(
                op="list", path=(params.get("path") or ".")[:1024])
        elif op == "fetch":
            url = params.get("path") or params.get("url") or params.get("link") or ""
            if url.strip():
                return TerminalAction(op="fetch", path=url[:1024])
        elif op == "searxng":
            query = params.get("pattern") or params.get("query") or ""
            if query.strip():
                return TerminalAction(op="searxng", pattern=query[:500])
        elif op == "python_exec":
            code = params.get("code") or params.get("text") or params.get("script") or ""
            if code.strip():
                return TerminalAction(op="python_exec", code=code[:20000])
        elif op == "done":
            reply = (params.get("reply") or params.get("text")
                     or params.get("message") or "")
            return TerminalAction(op="done", reply=reply[:2000])
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


_FUNCALL_RE = re.compile(
    r"<start_function_call>call:([A-Za-z0-9_.\-]+)(.*?)<end_function_call>",
    re.DOTALL)


_SINGLE_ARG_OPS = {
    # op -> arg key when the model emits a bare (braceless) value span
    "exec": "command", "exec_bg": "command", "poll": "command",
    "read": "path", "list": "path", "fetch": "path",
    "grep": "pattern", "searxng": "pattern",
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


def _unwrap(cmd: str) -> str:
    """One-layer unwrap for deny checks: bash -c 'RM -rf /' etc."""
    s = str(cmd or "").strip()
    m = re.search(r"bash\s+-c\s+['\"](.*)['\"]", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1)
    # $(rm -rf /) — extract inner
    m = re.search(r"\$\(\s*(.+?)\s*\)", s, flags=re.DOTALL)
    if m:
        return m.group(1)
    return s


def is_denied(action: TerminalAction) -> str:
    """Return denial reason, or '' when not denied."""
    if action is None:
        return ""
    cmd = getattr(action, "command", "") or ""
    # catch bash -c wrappers / $(...) / eval: unwrap one layer for deny check
    inner = _unwrap(cmd)
    if action.op in ("exec", "exec_bg") and _DENY_RE.search(inner):
        return "destructive command blocked by policy"
    if action.op == "exec_bg" and _DENY_RE.search(cmd):
        return "destructive command blocked by policy"
    if action.op == "write":
        p = action.path.strip()
        if p in ("", "/", "~") or ".." in p.split("/"):
            return "refusing to write outside a named file below cwd"
    return ""


def needs_confirm(action: TerminalAction) -> bool:
    """True when the TUI must ask before executing."""
    if action is None:
        return False
    if action.op in ("write", "edit"):
        return True
    if action.op in ("exec", "exec_bg"):
        return bool(_CONFIRM_CMD_RE.search(_unwrap(getattr(action, "command", "") or "")))
    return False


def check_policy(action: TerminalAction) -> tuple[str, str]:
    """('deny'|'confirm'|'allow', reason). Pure, unit-testable."""
    reason = is_denied(action)
    if reason:
        return "deny", reason
    if needs_confirm(action):
        return "confirm", "mutating/outward-facing command"
    return "allow", ""


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

    E.g. think ``I should use exec…`` + answer identical —
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


ROUTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    # word-boundary matched (short words especially); keep shell verbs here —
    # routing is not permission, the policy gate still decides deny/confirm.
    "exec": ("run", "execute", "command", "shell", "script", "delete",
             "remove", "move", "copy", "install", "test", "tests", "git",
             "python", "pytest", "build", "compile", "sudo", "apt", "rm",
             "mv", "dd", "chmod", "chown", "curl", "wget", "ssh", "docker",
             "systemctl", "eval", "pip", "npm"),
    "exec_bg": ("background", "long-running", "long running",
                "in the background", "nohup", "daemon", "watch"),
    "poll": ("poll", "job-", "job status", "still running", "check on",
             "background job"),
    "read": ("read", "open", "show", "cat", "display", "contents", "file"),
    "write": ("write", "create", "save", "new file", "make a file"),
    "edit": ("edit", "fix", "patch", "change", "modify", "update",
             "replace", "anchor"),
    "grep": ("grep", "search", "find", "look for", "where", "defined"),
    "list": ("list", "files", "directory", "folder", "ls", "contents"),
    "python_exec": ("python", "code", "script", "pandas", "numpy", "plot"),
    "fetch": ("fetch", "webfetch", "url", "link", "website", "webpage",
              "http"),
    "searxng": ("searxng", "web search", "websearch", "web", "internet",
                "online", "google"),
    "screenshot": ("screenshot", "screen capture", "capture the screen",
                   "capture screen"),
    "click": ("click", "left-click", "right-click", "click on"),
    "type_text": ("type", "typing", "keystrokes", "enter text"),
    "press_key": ("press key", "hotkey", "keyboard shortcut", "shortcut",
                  "ctrl+", "alt+"),
    "scroll": ("scroll",),
    "get_active_window": ("active window", "focused window",
                          "current window", "foreground window"),
    "list_windows": ("windows", "open windows", "window list", "window"),
    "open_app": ("open app", "launch", "start app", "application", "app"),
    "list_directory": ("list_directory",),
    "search_files": ("search files", "find file", "locate file",
                     "search_files"),
    "read_file": ("read_file",),
    "write_file": ("write_file",),
    "edit_file": ("edit_file",),
    "move_file": ("move", "move_file", "rename"),
    "delete_file": ("delete", "remove", "delete_file"),
    "search_web": ("search_web",),
    "open_url": ("open url", "open link", "open in browser", "open_url"),
    "extract_page": ("extract", "page text", "page content",
                     "extract_page"),
    "browser_click": ("browser click", "click on page", "browser_click"),
    "browser_type": ("fill", "fill in", "type into", "input field",
                     "browser_type"),
    "browser_scroll": ("scroll page", "browser_scroll"),
    "download_file": ("download", "download_file"),
}
"""Intent keywords per op. Matched case-insensitively on word boundaries."""

_ROUTE_DEPS: dict[str, tuple[str, ...]] = {
    # read-before-write is the core workflow: never strand edit/write
    # on a first step with no way to see the file. Poll needs its job.
    "edit": ("read",),
    "write": ("read",),
    "exec_bg": ("poll",),
    "poll": ("exec_bg",),
    "edit_file": ("read_file",),
    "write_file": ("read_file",),
    "delete_file": ("list_directory",),
    "move_file": ("list_directory",),
}

_ROUTE_FOLLOWUP: tuple[str, ...] = ("read", "list", "read_file",
                                    "list_directory")
"""Follow-up steps (observation present) can always navigate results."""


def _route_hits(blob: str) -> set[str]:
    import re

    hits: set[str] = set()
    for op, kws in ROUTE_KEYWORDS.items():
        for kw in kws:
            trail = r"\b" if kw[-1].isalnum() or kw[-1] == "_" else ""
            if re.search(r"\b" + re.escape(kw) + trail, blob):
                hits.add(op)
                break
    return hits


def route_tool_ops(text: str, observation: str = "") -> list[str] | None:
    """Narrow the ops to what this step plausibly needs.

    Returns sorted op names, or None (= show all tools) when nothing
    matches — recall over precision: a wrongly dropped tool fails the
    turn, a spare one only costs prompt tokens.
    """
    blob = f"{text or ''}\n{observation or ''}".lower()
    hits = _route_hits(blob)
    if not hits:
        return None
    expanded = set(hits)
    for op in hits:
        expanded.update(_ROUTE_DEPS.get(op, ()))
    if (observation or "").strip():
        expanded.update(_ROUTE_FOLLOWUP)
    # Backbone: the hands (read/write/list) plus the universal fallback
    # (exec) are always visible. A task needing a second step (fetch THEN
    # write) must see write in step one, or the model loops the first
    # tool and gives up (measured). Exec stays because the model reaches
    # for shell constantly — and Groq 400s calls to undeclared tools.
    expanded.update(("read", "write", "list", "exec"))
    ops = sorted(expanded & set(ALLOWED_OPS))
    return ops or None


def tools_for_request(text: str, observation: str = "") -> list:
    """Terminal + MCP defs filtered by :func:`route_tool_ops`.

    Order kept (terminal first, then MCP). No hits -> terminal tools
    only: the MCP set stays hidden unless routed, so small models never
    drown in 33 specs.
    """
    ops = route_tool_ops(text, observation)
    if ops is None:
        return TERMINAL_TOOLS
    wanted = set(ops)
    out = [t for t in TERMINAL_TOOLS if t["name"] in wanted]
    out += [t for t in MCP_TOOL_DEFS if t["name"] in wanted]
    return out
