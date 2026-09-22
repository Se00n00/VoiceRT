"""Terminal harness: single-model voice agent for the shell, via the agent lib.

Same contract as before: the ONE model either chats (plain text, spoken
back) or emits tool calls per step — but there is NO hand-rolled workflow
here. The turn runs on :func:`langchain.agents.create_agent` (model +
9 local tools, no extra middleware), and this module only translates:

- model/tool events -> :class:`AgentEvent` (action/observation/thinking/
  token/chat/summary/confirm/deny/error/audio — the TUI/bridge/CLI shape)
- policy (:func:`check_policy`) + ``confirm_fn`` (``y/n``, default denies)
  enforced inside the tool functions, so every entry point is protected
- TTS the final reply head, session memory, per-turn persistent shell

No checkpointer: memory lives in ``LangChainSessionMemory`` (RAM-only),
same as the voice loop.
"""
import asyncio
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from src.agent.events import AgentEvent
from src.agent.memory import LangChainSessionMemory
from src.tools.terminal import (
    TerminalAction,
    check_policy,
    is_degenerate,
    parse_bare_tail,
    parse_terminal_action,
    parse_xml_action,
)

__all__ = [
    "TerminalConfig",
    "TerminalHarness",
    "TurnCtx",
    "build_terminal_tools",
    "run_command",
]


@dataclass(frozen=True)
class TerminalConfig:
    """Per-harness knobs. Not YAML — construct in code."""

    max_steps: int = 6
    # Think + tool call needs room in ONE step: a MiniCPM/Qwen thinking
    # trace alone is often 100-150 tokens, plus 30-80 for the call.
    # 256 fits the context comfortably; MiniCPM thinking models get 320
    # (see LocalChatModel._step_budget).
    max_tokens_per_step: int = 256
    cwd: str = "."
    timeout_s: float = 30.0
    out_cap: int = 6000
    # Docker sandbox for tool execution (None = host, as before).
    # Set to SandboxConfig(...) to run exec/exec_bg/poll inside a
    # per-turn container (host cwd bind-mounted at /work). File ops stay
    # host-side on the same tree; spawn_terminal always stays on host.
    sandbox: Any = None


def _edit_diff(src: str, anchor: str, new: str, ctx: int = 3) -> str:
    """Unified diff preview for an anchored edit. Never raises."""
    try:
        import difflib
        a = (anchor.splitlines() or [""])[0][:80]
        b = (new.splitlines() or [""])[0][:80]
        diff = difflib.unified_diff(
            src.splitlines(), src.replace(anchor, new, 1).splitlines(),
            fromfile="before", tofile="after", n=ctx, lineterm="")
        out = "\n".join(list(diff)[:40])
        return out or f"- {a}\n+ {b}"
    except Exception:
        return f"- {anchor[:200]}\n+ {new[:200]}"


def _write_diff(path: str, old: str, new: str) -> str:
    try:
        import difflib
        diff = difflib.unified_diff(
            old.splitlines(), new.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", n=3, lineterm="")
        out = "\n".join(list(diff)[:50])
        return out or "(new file)"
    except Exception:
        return "(diff unavailable)"


def run_command(action: TerminalAction, cwd: str = ".",
                 timeout_s: float = 30.0, out_cap: int = 6000,
                 shell=None, jobs=None) -> str:
    """Execute one validated action. No weights. Never raises.

    If a :class:`PersistentShell` is supplied, exec/exec_bg/poll use it
    (persistent CWD/env; bg jobs share the turn). Otherwise a one-shot
    subprocess is used (tests).
    """
    jm = jobs if jobs is not None else getattr(shell, "jobs", None) if shell else None
    try:
        if action.op == "exec":
            if shell is not None:
                rc, out = shell.exec(action.command, timeout_s=timeout_s)
                if len(out) > out_cap:
                    out = out[:out_cap] + f"\n…[truncated {len(out)} chars]"
                return f"rc={rc}\n{out.strip() or '(no output)'}"
            p = subprocess.run(
                action.command, shell=True, cwd=cwd or ".",
                capture_output=True, text=True, timeout=timeout_s)
            out = (p.stdout or "") + (p.stderr or "")
            if len(out) > out_cap:
                out = out[:out_cap] + f"\n…[truncated {len(out)} chars]"
            return f"rc={p.returncode}\n{out.strip() or '(no output)'}"
        if action.op == "spawn_terminal":
            from src.mcp.terminal_spawn import spawn_terminal_window

            res = spawn_terminal_window(command=action.command, cwd=cwd or None, title=action.title)
            if res.get("ok"):
                return f"spawned terminal {res['terminal_id']} cwd={res['cwd']} cmd={res['command']}"
            return f"error: {res.get('error')}"
        if action.op == "exec_bg":
            if jm is None:
                return "error: background jobs unavailable (no shell context)"
            jid = jm.start(action.command)
            return f"started {jid}: {action.command[:200]}"
        if action.op == "poll":
            if jm is None:
                return "error: background jobs unavailable (no shell context)"
            r = jm.poll(action.command)
            prefix = "running" if r.get("running") else f"done rc={r.get('rc')}"
            return f"{prefix} {r.get('job','')}\n{(r.get('output','') or '').strip() or '(no output)'}"
        if action.op == "edit":
            base = os.path.abspath(cwd or ".")
            full = os.path.abspath(os.path.join(base, action.path.strip()))
            if not full.startswith(base):
                return "denied: path escapes cwd"
            try:
                with open(full, "r", errors="replace") as f:
                    src = f.read()
            except FileNotFoundError:
                return f"error: no such file {action.path.strip()}"
            except Exception as exc:
                return f"error: {exc}"
            if action.anchor not in src:
                return "error: anchor not found (must copy verbatim from the file)"
            if src.count(action.anchor) > 1:
                return "error: anchor not unique (appears %d times)" % src.count(action.anchor)
            diff = _edit_diff(src, action.anchor, action.text)
            try:
                with open(full, "w") as f:
                    f.write(src.replace(action.anchor, action.text, 1))
            except Exception as exc:
                return f"error: {exc}"
            return "patched %s with:\n%s" % (action.path.strip(), diff)
        if action.op == "grep":
            base = os.path.abspath(cwd or ".")
            path = action.path.strip() or "."
            full = os.path.abspath(os.path.join(base, path))
            if not full.startswith(base):
                return "denied: path escapes cwd"
            try:
                import re as _re
                pat = _re.compile(action.pattern)
            except Exception as exc:
                return f"error: bad pattern: {exc}"
            hits = []
            for root, _, files in os.walk(full if os.path.isdir(full) else os.path.dirname(full)):
                rel_root = os.path.relpath(root, base)
                if rel_root.startswith(".git") or "__pycache__" in rel_root:
                    continue
                for name in files:
                    f = os.path.join(root, name)
                    try:
                        txt = open(f, "r", errors="replace").read()
                    except Exception:
                        continue
                    for lno, line in enumerate(txt.splitlines(), 1):
                        if pat.search(line):
                            hits.append(f"{os.path.relpath(f, base)}:{lno}: {line.strip()[:200]}")
                            if len(hits) >= 80:
                                break
                    if len(hits) >= 80:
                        break
                if len(hits) >= 80:
                    break
            return "\n".join(hits) or "(no matches)"
        if action.op == "list":
            target = action.path.strip() or "."
            base = os.path.abspath(cwd or ".")
            full = os.path.abspath(os.path.join(base, target))
            if not full.startswith(base):
                return "denied: path escapes cwd"
            try:
                names = sorted(os.listdir(full))
            except Exception as exc:
                return f"error: {exc}"
            return "\n".join(names[:200]) or "(empty)"
        if action.op == "read":
            base = os.path.abspath(cwd or ".")
            full = os.path.abspath(os.path.join(base, action.path.strip()))
            if not full.startswith(base):
                return "denied: path escapes cwd"
            try:
                with open(full, "r", errors="replace") as f:
                    data = f.read(out_cap + 1)
            except Exception as exc:
                return f"error: {exc}"
            if len(data) > out_cap:
                data = data[:out_cap] + "\n…[truncated]"
            return data or "(empty file)"
        if action.op == "write":
            base = os.path.abspath(cwd or ".")
            full = os.path.abspath(os.path.join(base, action.path.strip()))
            if not full.startswith(base):
                return "denied: path escapes cwd"
            try:
                os.makedirs(os.path.dirname(full) or base, exist_ok=True)
                with open(full, "w") as f:
                    f.write(action.text)
            except Exception as exc:
                return f"error: {exc}"
            return f"wrote {len(action.text)} chars to {action.path.strip()}"
        return f"unknown op: {action.op}"
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout_s:.0f}s"
    except Exception as exc:  # never break the agent loop
        return f"error: {exc}"


def _speak_head(reply: str, limit: int = 280) -> str:
    """Head of a reply for voicing; full text stays on screen/in memory."""
    t = " ".join(str(reply or "").split())
    if len(t) <= limit:
        return t
    cut = t[:limit]
    dot = cut.rfind(". ")
    head = (cut[:dot + 1] if dot > 120 else cut).strip()
    return f"{head} … full output is on screen."


@dataclass
class TurnCtx:
    """Per-turn execution context shared by the 9 local tools."""

    shell: Any = None
    cwd: str = "."
    approvals: set = field(default_factory=set)
    confirm_fn: Any = None
    loop: Any = None
    cfg: Any = None
    sink: list = field(default_factory=list)  # (kind, data) for confirm/deny UI events
    sandbox: Any = None  # DockerSandbox for this turn, if enabled


def _approval_key(action: TerminalAction) -> str:
    cmd = getattr(action, "command", "") or ""
    if cmd.strip():
        return cmd.strip().split()[0][:40]
    return getattr(action, "op", "")[:40]


def _resolve_confirm(ctx: TurnCtx, action: TerminalAction):
    """Ask confirm_fn (sync or async). Returns 'always' | True | False."""
    fn = ctx.confirm_fn
    if fn is None:
        return False
    try:
        if asyncio.iscoroutinefunction(fn):
            fut = asyncio.run_coroutine_threadsafe(fn(action), ctx.loop)
            res = fut.result(timeout=300)
        else:
            res = fn(action)
    except Exception:
        return False
    if isinstance(res, str) and res.lower() in ("always", "a", "allowlist"):
        return "always"
    return bool(res)


def _file_preview(action: TerminalAction, cwd: str) -> str:
    """Diff preview for mutating file ops shown in the confirm event."""
    try:
        base = os.path.abspath(cwd or ".")
        full = os.path.abspath(os.path.join(base, action.path.strip())) if action.path.strip() else ""
        if not full or not full.startswith(base) or not os.path.isfile(full):
            return "(new file)" if action.op == "write" else ""
        with open(full, "r", errors="replace") as f:
            old = f.read()
        if action.op == "edit" and action.anchor in old:
            return _edit_diff(old, action.anchor, action.text)[:1200]
        if action.op == "write":
            return _write_diff(action.path.strip(), old, action.text)[:1200]
    except Exception:
        pass
    return ""


def _guarded(ctx: TurnCtx, action: TerminalAction) -> str | None:
    """Policy gate: None = allowed, else the tool-result string to return."""
    cfg = ctx.cfg
    timeout_s = getattr(cfg, "timeout_s", 30.0) if cfg else 30.0
    out_cap = getattr(cfg, "out_cap", 6000) if cfg else 6000
    # approvals override: pre-approved prefixes skip confirm
    verdict, reason = check_policy(action)
    if verdict == "confirm" and ctx.approvals:
        if _approval_key(action) in ctx.approvals or action.op in ctx.approvals:
            verdict, reason = "allow", "pre-approved"
    if verdict == "deny":
        ctx.sink.append(("deny", {"action": action.as_dict(), "reason": reason}))
        return f"Blocked: {reason}."
    if verdict == "confirm":
        preview = _file_preview(action, ctx.cwd) if action.op in ("edit", "write") else ""
        ctx.sink.append(("confirm", {"action": action.as_dict(),
                                     "reason": reason, "preview": preview}))
        ok = _resolve_confirm(ctx, action)
        if ok == "always":
            ctx.approvals.add(_approval_key(action))
        if not ok:
            ctx.sink.append(("deny", {"action": action.as_dict(),
                                      "reason": "denied by user"}))
            return "Cancelled — I did not run that."
    shell = ctx.shell
    cwd = getattr(shell, "cwd", ctx.cwd) if shell is not None else ctx.cwd
    out = run_command(action, cwd=cwd, timeout_s=timeout_s,
                      out_cap=out_cap, shell=shell)
    # track cwd if the persistent shell moved
    try:
        if shell is not None and hasattr(shell, "cwd"):
            ctx.cwd = shell.cwd
    except Exception:
        pass
    # cap context burn; full detail already went to the observation event
    if len(out) > 2000:
        lines = out.splitlines()
        if len(lines) > 60:
            out = "\n".join(lines[:20]) + f"\n…[{len(lines)-40} lines omitted]…\n" + "\n".join(lines[-20:])
            out = out[:2000]
        else:
            out = out[:2000] + "\n…[truncated]"
    return out


def build_terminal_tools(ctx: TurnCtx) -> list:
    """9 local LangChain tools closing over this turn's shell + policy."""
    from langchain_core.tools import StructuredTool

    def _exec(command: str) -> str:
        """Run a bash shell command (persistent CWD/env). Prefer read-only commands."""
        return _guarded(ctx, TerminalAction(op="exec", command=command)) or ""

    def _exec_bg(command: str) -> str:
        """Start a long-running shell command in the background. Poll it later."""
        return _guarded(ctx, TerminalAction(op="exec_bg", command=command)) or ""

    def _poll(command: str) -> str:
        """Poll a background job started by exec_bg. Job id like job-1."""
        return _guarded(ctx, TerminalAction(op="poll", command=command)) or ""

    def _read(path: str) -> str:
        """Read a text file below the working directory."""
        return _guarded(ctx, TerminalAction(op="read", path=path)) or ""

    def _write(path: str, text: str) -> str:
        """Write text to a file below the working directory."""
        return _guarded(ctx, TerminalAction(op="write", path=path, text=text)) or ""

    def _edit(path: str, anchor: str, text: str) -> str:
        """Anchored patch: replace `anchor` verbatim in the file with `text`."""
        return _guarded(ctx, TerminalAction(op="edit", path=path, anchor=anchor, text=text)) or ""

    def _grep(pattern: str, path: str = ".") -> str:
        """Grep a pattern across files below the working directory."""
        return _guarded(ctx, TerminalAction(op="grep", pattern=pattern, path=path)) or ""

    def _list(path: str = ".") -> str:
        """List directory entries below the working directory."""
        return _guarded(ctx, TerminalAction(op="list", path=path)) or ""

    def _spawn_terminal(command: str = "", cwd: str = "", title: str = "") -> str:
        """Spawn a new OS terminal window. Use for opencode/htop/interactive TUIs."""
        return _guarded(ctx, TerminalAction(op="spawn_terminal", command=command,
                                            cwd=cwd, title=title)) or ""

    return [
        StructuredTool.from_function(_exec, name="exec"),
        StructuredTool.from_function(_exec_bg, name="exec_bg"),
        StructuredTool.from_function(_poll, name="poll"),
        StructuredTool.from_function(_read, name="read"),
        StructuredTool.from_function(_write, name="write"),
        StructuredTool.from_function(_edit, name="edit"),
        StructuredTool.from_function(_grep, name="grep"),
        StructuredTool.from_function(_list, name="list"),
        StructuredTool.from_function(_spawn_terminal, name="spawn_terminal"),
    ]


def _lc_messages(history: list) -> list:
    """Session dicts -> LangChain messages (user/assistant only)."""
    from langchain_core.messages import AIMessage, HumanMessage

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


def _chunk_pieces(chunk) -> list[str]:
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


class TerminalHarness:
    """One shared-model terminal agent: text in, actions + speech out.

    No custom graph: each turn builds a stock
    :func:`langchain.agents.create_agent` (model + 9 local tools) and
    streams its events as :class:`AgentEvent`.
    """

    def __init__(self, llm, tts=None, sessions=None,
                 config: TerminalConfig | None = None, confirm_fn=None):
        self.llm = llm
        self.tts = tts
        self.sessions = sessions or LangChainSessionMemory()
        self.config = config or TerminalConfig()
        self.confirm_fn = confirm_fn
        self.approvals: dict[str, set[str]] = {}  # sid -> set of approved command prefixes

    def remember_approval(self, session_id: str, command: str):
        """Remember that this session approved a command prefix."""
        if not session_id or not command:
            return
        self.approvals.setdefault(session_id, set()).add(command.strip().split()[0][:40])

    def is_approved(self, session_id: str | None, action: TerminalAction) -> bool:
        if not session_id:
            return False
        allow = self.approvals.get(session_id, set())
        if not allow:
            return False
        cmd = getattr(action, "command", "") or ""
        first = cmd.strip().split()[0] if cmd.strip() else ""
        return first in allow or action.op in allow

    def _build_agent(self, ctx: TurnCtx):
        from langchain.agents import create_agent

        from src.agent.chat_model import LocalChatModel
        from src.tools.terminal import TERMINAL_PREAMBLE

        system = TERMINAL_PREAMBLE + (
            "\n\nWork until done: use tools step by step, then give one "
            "short final reply.")
        return create_agent(
            LocalChatModel(llm=self.llm),
            build_terminal_tools(ctx),
            system_prompt=system,
        )

    async def run_turn(self, text: str, session_id: str | None = None,
                       cwd: str | None = None):
        """Run one turn, yielding :class:`AgentEvent` per step."""
        from langchain_core.messages import AIMessageChunk

        from src.models.llm import split_thinking

        if not str(text or "").strip():
            raise ValueError("empty text")
        sid = session_id
        # execution backend: host shell, or a per-turn docker container
        # (exec inside, file ops on the bind-mounted tree, spawn on host).
        from src.agent.shell import PersistentShell
        sandbox = None
        if getattr(self.config, "sandbox", None) is not None:
            from src.sandbox.docker import DockerShell

            sandbox = self.config.sandbox.create(host_cwd=cwd or self.config.cwd)
            try:
                sandbox.ensure_running()
            except Exception as exc:
                yield AgentEvent(node="term", kind="error",
                                 data={"message": f"sandbox: {exc}"[:300]})
                yield AgentEvent(node="term", kind="summary", data={
                    "text": str(text), "reply": "", "thinking": "",
                    "session_id": sid})
                return
            shell = DockerShell(sandbox, timeout_s=self.config.timeout_s)
        else:
            shell = PersistentShell(cwd=cwd or self.config.cwd, timeout_s=self.config.timeout_s)
        ctx = TurnCtx(shell=shell, cwd=cwd or self.config.cwd,
                      approvals=set(self.approvals.get(sid, set())) if sid else set(),
                      confirm_fn=self.confirm_fn,
                      loop=asyncio.get_running_loop(), cfg=self.config,
                      sandbox=sandbox)
        # wrap confirm to support "always" -> remember approval
        orig_confirm = self.confirm_fn

        async def _confirm_wrapper(action):
            res = None
            if orig_confirm is not None:
                res = (await orig_confirm(action)
                       if asyncio.iscoroutinefunction(orig_confirm)
                       else orig_confirm(action))
            if isinstance(res, str) and res.lower() in ("always", "a", "allowlist"):
                if sid:
                    self.remember_approval(sid, getattr(action, "command", "") or getattr(action, "op", ""))
                    ctx.approvals.add(_approval_key(action))
                return "always"
            return bool(res) if res is not None else False

        ctx.confirm_fn = _confirm_wrapper
        agent = self._build_agent(ctx)
        user_text = str(text) + f"\nCWD: {ctx.cwd} SHELL: bash"
        lc_history = _lc_messages(self.sessions.history(sid) if sid else [])
        from langchain_core.messages import HumanMessage
        lc_msgs = lc_history + [HumanMessage(content=user_text)]

        final_reply = ""
        did_work = False
        last_think = ""
        drained = 0

        def _drain():
            """Policy sink (confirm/deny) recorded by the tools, in order."""
            nonlocal drained
            while drained < len(ctx.sink):
                kind, data = ctx.sink[drained]
                drained += 1
                yield AgentEvent(node="term", kind=kind, data=data)

        def _norm_msg(m):
            if isinstance(m, dict):
                return (m.get("type") or m.get("role") or "",
                        m.get("content", ""), m.get("tool_calls"),
                        m.get("additional_kwargs") or {})
            return (getattr(m, "type", None) or "",
                    getattr(m, "content", ""),
                    getattr(m, "tool_calls", None),
                    getattr(m, "additional_kwargs", None) or {})

        def _think_of(ak) -> str:
            return str(ak.get("thinking", "") or "") if isinstance(ak, dict) else ""

        def _resolve_reply(content: str) -> str:
            """Text-only AI content -> reply; legacy done envelopes resolve."""
            txt = str(content or "").strip()
            if not txt:
                return ""
            for cand in (parse_terminal_action(txt), parse_xml_action(txt),
                         parse_bare_tail(txt)):
                if cand is not None and cand.op == "done":
                    return cand.reply or ""
            return txt

        rl = 10 + int(getattr(self.config, "max_steps", 6) or 6) * 5
        try:
            stream = agent.astream({"messages": lc_msgs},
                                   config={"recursion_limit": rl},
                                   stream_mode=["messages", "updates"])
            async for chunk in stream:
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    mode, payload = chunk
                else:
                    mode, payload = "updates", chunk
                if mode == "messages":
                    msg, _meta = payload if isinstance(payload, (tuple, list)) else (payload, {})
                    if isinstance(msg, AIMessageChunk):
                        for piece in _chunk_pieces(msg):
                            yield AgentEvent(node="term", kind="token",
                                             data={"piece": piece[:500]})
                    continue
                # updates mode: node -> {messages: [...]}
                if not isinstance(payload, dict):
                    continue
                for ev in _drain():
                    yield ev
                for _node, data in payload.items():
                    msgs = data.get("messages", []) if isinstance(data, dict) else []
                    for m in msgs if isinstance(msgs, list) else []:
                        role, content, tool_calls, ak = _norm_msg(m)
                        if isinstance(content, list):
                            content = " ".join(
                                str(c.get("text", c)) for c in content
                                if isinstance(c, dict))
                        if role in ("ai", "assistant"):
                            think = _think_of(ak)
                            if think and think != last_think:
                                last_think = think
                                yield AgentEvent(node="term", kind="thinking",
                                                 data={"text": think[:2000]})
                            if tool_calls:
                                for tc in tool_calls:
                                    if isinstance(tc, dict):
                                        name = tc.get("name", "?")
                                        args = tc.get("args") or {}
                                    else:
                                        name = getattr(tc, "name", "?")
                                        args = getattr(tc, "args", {}) or {}
                                    if not isinstance(args, dict):
                                        args = {}
                                    yield AgentEvent(
                                        node="term", kind="action",
                                        data={"action": {"action": name, **args}})
                                    did_work = True
                                # tool-call message bodies (often the raw
                                # envelope) are not the reply; the model's
                                # next text turn decides it.
                            elif str(content or "").strip():
                                final_reply = _resolve_reply(content)
                        elif role == "tool":
                            did_work = True
                            obs = str(content or "")
                            if obs.startswith("Blocked:"):
                                yield AgentEvent(
                                    node="term", kind="deny",
                                    data={"reason": obs[len("Blocked:"):].strip()[:300]})
                            elif obs.startswith("Cancelled"):
                                yield AgentEvent(
                                    node="term", kind="deny",
                                    data={"reason": "denied by user"})
                            else:
                                yield AgentEvent(
                                    node="term", kind="observation",
                                    data={"observation": obs[:1200]})
                        # human/system/tool-echoes: not UI events
            for ev in _drain():
                yield ev
        except Exception as exc:  # never kill the turn on engine errors
            yield AgentEvent(node="term", kind="error",
                             data={"message": str(exc)[:300]})
        finally:
            try:
                shell.close()
            except Exception:
                pass
            if sandbox is not None:
                try:
                    sandbox.close()
                except Exception:
                    pass

        # thinking never voiced/memorized — only the answer is
        thinking, reply = split_thinking(final_reply)
        if thinking and thinking != last_think:
            yield AgentEvent(node="term", kind="thinking",
                             data={"text": thinking[:2000]})
        if not reply and did_work:
            reply = "Done."
        if reply and is_degenerate(reply):
            yield AgentEvent(node="term", kind="error",
                             data={"message": "model output degenerate"})
            reply = "Sorry — I garbled that. Try rephrasing."
        if reply or did_work:
            if not reply:
                reply = "Done."
            yield AgentEvent(node="term", kind="chat", data={"reply": reply})
        if self.tts is not None and reply:
            try:
                out = await self.tts.speak(_speak_head(reply))
                import numpy as _np

                yield AgentEvent(node="term", kind="audio", data={
                    "wav": _np.asarray(out.wav, dtype="float32"),
                    "sr": int(out.sample_rate), "sentence": reply[:500]})
            except Exception as exc:
                yield AgentEvent(node="term", kind="error",
                                 data={"message": f"tts failed: {exc}"[:200]})
        if sid and (text or reply):
            try:
                self.sessions.remember_turn(sid, str(text), reply)
            except Exception:
                pass
        yield AgentEvent(node="term", kind="summary", data={
            "text": str(text), "reply": reply,
            "thinking": thinking[:2000], "session_id": sid})
