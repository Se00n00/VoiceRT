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
    "TERMINAL_PREAMBLE",
    "TerminalAction",
    "parse_terminal_action",
    "check_policy",
    "is_denied",
    "needs_confirm",
]

ALLOWED_OPS = ("exec", "read", "write", "list", "done")

TERMINAL_PREAMBLE = (
    "You control a local terminal (bash). Reply with EITHER one short chat "
    "sentence OR exactly one JSON action. No other text when acting. "
    'Ops: exec {action,command} | read {action,path} | '
    'write {action,path,text} | list {action,path?} | done {action,reply}. '
    "Prefer read-only commands. Never emit destructive commands; they are blocked. "
    'Example user "list the files" -> {"action": "list", "path": "."}. '
    'Example user "hello" -> Hello! How can I help?'
)

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
    """Validated single terminal step."""

    op: str = "done"
    command: str = ""
    path: str = ""
    text: str = ""
    reply: str = ""

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
        return out


def parse_terminal_action(text: str) -> TerminalAction | None:
    """Parse ONE JSON action from model text. None = plain chat. Never raises."""
    if not text or not text.strip():
        return None
    raw = text.strip()
    candidates = []
    if raw.startswith("{"):
        candidates.append(raw)
    candidates.extend(m.group(0) for m in _JSON_RE.finditer(raw))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        op = str(obj.get("action", "")).strip().lower()
        if op not in ALLOWED_OPS:
            continue
        command = str(obj.get("command", "") or "")[:4000]
        path = str(obj.get("path", "") or "")[:1024]
        body = str(obj.get("text", "") or "")[:20000]
        reply = str(obj.get("reply", "") or "")[:2000]
        if op == "exec" and not command.strip():
            continue
        if op in ("read", "write", "list") and not path.strip() and op != "list":
            continue
        if op == "write" and not body:
            continue
        return TerminalAction(op=op, command=command, path=path,
                              text=body, reply=reply)
    return None


def is_denied(action: TerminalAction) -> str:
    """Return denial reason, or '' when not denied."""
    if action is None:
        return ""
    if action.op == "exec" and _DENY_RE.search(action.command):
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
    if action.op in ("write",):
        return True
    if action.op == "exec" and _CONFIRM_CMD_RE.search(action.command):
        return True
    return False


def check_policy(action: TerminalAction) -> tuple[str, str]:
    """('deny'|'confirm'|'allow', reason). Pure, unit-testable."""
    reason = is_denied(action)
    if reason:
        return "deny", reason
    if needs_confirm(action):
        return "confirm", "mutating/outward-facing command"
    return "allow", ""
