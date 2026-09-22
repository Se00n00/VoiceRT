"""Single-model terminal action schema + safety policy.

Same contract as the browser tools: the ONE Qwen model either chats
(plain text, spoken back) or emits exactly one JSON action per step::

    {"action": "exec", "command": "ls -la"}
    {"action": "read", "path": "server.py"}
    {"action": "write", "path": "notes.txt", "text": "hello"}
    {"action": "list", "path": "."}
    {"action": "done", "reply": "Listed the repo."}

This module is pure (no weights, no yaml). Execution lives in
:mod:`src.agent.terminal` (LangGraph harness); the policy here decides
deny / confirm / allow. Your shell, your risk: denies cover
system-breakers, everything mutating needs an explicit yes.
"""
import json
import re
from dataclasses import dataclass

__all__ = [
    "ALLOWED_OPS",
    "TERMINAL_TOOLS",
    "TerminalAction",
    "parse_terminal_action",
    "parse_xml_action",
    "parse_bare_tail",
    "check_policy",
    "is_denied",
    "needs_confirm",
    "is_degenerate",
    "is_echo",
    "route_tool_ops",
    "tools_for_request",
]

ALLOWED_OPS = ("exec", "exec_bg", "poll", "read", "write", "edit", "grep", "list", "python_exec", "fetch", "searxng", "done")


# Native tool definitions for models with template-level tool support
# (MiniCPM5: its chat template renders these into <tools> XML when passed
# as `tools=` to apply_chat_template). Same eight ops as the JSON schema.
TERMINAL_TOOLS = [
    {"name": "exec",
     "description": "Run a bash shell command (persistent CWD/env). Prefer read-only commands.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]}},
    {"name": "exec_bg",
     "description": "Start a long-running shell command in the background. Poll it later.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]}},
    {"name": "poll",
     "description": "Poll a background job started by exec_bg. Poll until running is false.",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string",
                                               "description": "job id like job-1"}},
                    "required": ["command"]}},
    {"name": "read",
     "description": "Read a text file below the working directory.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}},
    {"name": "write",
     "description": "Write text to a file below the working directory.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "text": {"type": "string"}},
                    "required": ["path", "text"]}},
    {"name": "edit",
     "description": "Anchored patch: replace `anchor` verbatim in the file with `text`. Anchor must copy from the file; fails if missing.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "anchor": {"type": "string"},
                                   "text": {"type": "string"}},
                    "required": ["path", "anchor", "text"]}},
    {"name": "grep",
     "description": "Grep a pattern across files below the working directory.",
     "parameters": {"type": "object",
                    "properties": {"pattern": {"type": "string"},
                                   "path": {"type": "string"}},
                    "required": ["pattern"]}},
    {"name": "list",
     "description": "List directory entries below the working directory.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}}}},
    {"name": "python_exec",
     "description": "Run Python code inside an isolated container (no shell, no host files). Use for data work, plotting, quick scripts.",
     "parameters": {"type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"]}},
    {"name": "fetch",
     "description": "Fetch a web page (http/https) and return its text, truncated. Offline-friendly: for local files use read instead.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string",
                                            "description": "page URL"}},
                    "required": ["path"]}},
    {"name": "searxng",
     "description": "Web search via the local SearXNG instance. Returns titles, URLs and snippets.",
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


@dataclass(frozen=True)
class TerminalAction:
    """Validated single terminal step. edit carries anchor, grep carries pattern."""

    op: str = "done"
    command: str = ""
    path: str = ""
    text: str = ""
    reply: str = ""
    anchor: str = ""
    pattern: str = ""
    code: str = ""  # python_exec payload (own key: shell deny patterns must not see it)

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
        return out


def _norm_op(op: str) -> str:
    """Fuzzy op match for models that paraphrase the schema.

    MiniCPM emits near-misses (``run_command``, ``list_files``) instead of
    the exact ops. Substring match on the normalized name, most-specific
    first. Returns "" when nothing matches.
    """
    o = re.sub(r"[^a-z]+", "", str(op or "").lower())
    if not o:
        return ""
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


_BARE_RE = re.compile(
    r'name="([^"]+)"\s*>\s*([^<>\n]*?)(?=\s*name="[^"]+"\s*>|\s*$)',
    re.DOTALL | re.IGNORECASE | re.MULTILINE)


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
}
"""Intent keywords per op. Matched case-insensitively on word boundaries."""

_ROUTE_DEPS: dict[str, tuple[str, ...]] = {
    # read-before-write is the core workflow: never strand edit/write
    # on a first step with no way to see the file. Poll needs its job.
    "edit": ("read",),
    "write": ("read",),
    "exec_bg": ("poll",),
    "poll": ("exec_bg",),
}

_ROUTE_FOLLOWUP: tuple[str, ...] = ("read", "list")
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
    ops = sorted(expanded & set(ALLOWED_OPS))
    return ops or None


def tools_for_request(text: str, observation: str = "") -> list:
    """TERMINAL_TOOLS filtered by :func:`route_tool_ops` (order kept)."""
    ops = route_tool_ops(text, observation)
    if ops is None:
        return TERMINAL_TOOLS
    wanted = set(ops)
    return [t for t in TERMINAL_TOOLS if t["name"] in wanted]
