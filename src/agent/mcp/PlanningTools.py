"""
Stateful planning tools: task, question, skill.

These are THE PlanningTools tool implementations —
:mod:`src.agent.mcp.mcp` registers them over MCP via ``app.add_tool``.
All tool functions are sync and total (never raise — errors become
strings); state lives in module singletons (one class, one job).

Todo tracking is NOT here: plan state belongs to the agent loop via
``TodoListMiddleware`` (see :mod:`src.prompts.middleware` for its
custom prompts), not to a stateless tool call.

contract, dependencies, who calls it: background task launch/poll,
pending-question queue, SKILL.md lookup; depends on stdlib plus
:mod:`src.agent.shell` (task jobs); used by the MCP registry. State
is per server process (stdio servers persist for the session; a
server restart clears it).
"""
import os as _os
import threading as _threading

__all__ = [
    "task",
    "question",
    "skill",
]


class TaskRunner:
    """Background shell tasks (launch/poll/list) for agent sub-work."""

    def __init__(self) -> None:
        self._mgr = None

    def _jobs(self):
        """JobManager backend, created on first use."""
        if self._mgr is None:
            from src.agent.shell import JobManager
            base = (_os.environ.get("VOICE_WORKDIR", "") or "").strip()
            self._mgr = JobManager(
                cwd=base if base and _os.path.isdir(base) else ".")
        return self._mgr

    def launch(self, command: str) -> str:
        """Start a command in the background; returns its id."""
        cmd = str(command or "").strip()
        if not cmd:
            return "error: empty command"
        try:
            jid = self._jobs().start(cmd)
        except Exception as exc:
            return f"error: launch failed ({exc})"[:300]
        return f"started {jid}: {cmd[:200]}"

    def poll(self, tid: str) -> str:
        """New output since the last poll, plus a running flag."""
        tid = str(tid or "").strip()
        if not tid:
            return "error: empty id"
        try:
            out = self._jobs().poll(tid)
        except Exception as exc:
            return f"error: poll failed ({exc})"[:300]
        state = "running" if out.get("running") else "done"
        text = str(out.get("output", "") or "").strip() or "(no output)"
        return f"{tid} [{state} rc={out.get('rc')}]\n{text[:6000]}"


class QuestionQueue:
    """Pending questions for the user (ask/answer/pending)."""

    def __init__(self) -> None:
        self._lock = _threading.Lock()
        self._seq = 0
        self._items: dict[str, dict] = {}

    def ask(self, text: str, options: str) -> str:
        """Queue one question; returns its id plus the formatted prompt."""
        q = str(text or "").strip()
        if not q:
            return "error: empty question"
        opts = str(options or "").strip()
        with self._lock:
            self._seq += 1
            qid = f"q-{self._seq}"
            self._items[qid] = {"question": q[:2000],
                                "options": opts[:2000], "answer": ""}
        tail = f"\noptions: {opts}" if opts else ""
        return f"{qid}: {q}{tail}\n(ask the user in chat, then record with question answer)"

    def answer(self, qid: str, reply: str) -> str:
        """Record the user's answer to a queued question."""
        qid = str(qid or "").strip()
        reply = str(reply or "").strip()
        if not qid or not reply:
            return "error: need id and answer"
        with self._lock:
            item = self._items.get(qid)
            if item is None:
                return f"error: no question {qid!r}"
            item["answer"] = reply[:2000]
            return f"{qid} answered: {reply[:200]}"

    def pending(self) -> str:
        """Unanswered questions, oldest first."""
        with self._lock:
            open_items = [(k, v) for k, v in self._items.items()
                          if not v.get("answer")]
            if not open_items:
                return "(no pending questions)"
            return "\n".join(f"{k}: {v['question']}" for k, v in open_items)


_TASKS = TaskRunner()
_QUESTIONS = QuestionQueue()


def task(action: str = "list", id: str = "", command: str = "") -> str:
    """Background work units: launch a command, poll it, or list."""
    try:
        op = str(action or "list").strip().lower() or "list"
        if op == "launch":
            return _TASKS.launch(command)
        if op in ("poll", "result"):
            return _TASKS.poll(id)
        if op == "list":
            return "(task ids come from launch; poll by id)"
        return f"error: bad action {action!r} (launch/poll/list)"
    except Exception as exc:
        return f"error: task failed ({exc})"[:300]


def question(action: str = "pending", id: str = "", question: str = "",
             options: str = "", answer: str = "") -> str:
    """User clarifications: ask (queue), answer (record), pending."""
    try:
        op = str(action or "pending").strip().lower() or "pending"
        if op == "ask":
            return _QUESTIONS.ask(question, options)
        if op == "answer":
            return _QUESTIONS.answer(id, answer)
        if op == "pending":
            return _QUESTIONS.pending()
        return f"error: bad action {action!r} (ask/answer/pending)"
    except Exception as exc:
        return f"error: question failed ({exc})"[:300]


def _skill_bases() -> list:
    """Candidate skill roots (repo, workdir, user config)."""
    roots = []
    try:
        here = _os.path.abspath(__file__)
        repo = _os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.dirname(here))))
        roots.append(_os.path.join(repo, "skills"))
    except Exception:
        pass
    try:
        wd = (_os.environ.get("VOICE_WORKDIR", "") or "").strip()
        if wd:
            roots.append(_os.path.join(wd, "skills"))
    except Exception:
        pass
    roots.append(_os.path.expanduser("~/.config/opencode/skills"))
    seen, out = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def skill(name: str = "") -> str:
    """Load a skill (SKILL.md) by name; blank name lists available."""
    try:
        want = str(name or "").strip()
        found: dict[str, str] = {}
        for base in _skill_bases():
            try:
                if not _os.path.isdir(base):
                    continue
                for entry in sorted(_os.listdir(base)):
                    path = _os.path.join(base, entry, "SKILL.md")
                    if _os.path.isfile(path):
                        found.setdefault(entry, path)
            except Exception:
                continue
        if not want:
            if not found:
                return "(no skills installed)"
            return "skills:\n" + "\n".join(f"- {k}" for k in sorted(found))
        for base in _skill_bases():
            for cand in (_os.path.join(base, want, "SKILL.md"),
                         _os.path.join(base, want + ".md")):
                try:
                    if _os.path.isfile(cand):
                        with open(cand, "r", encoding="utf-8",
                                  errors="replace") as f:
                            return f.read(12000) or "(empty skill)"
                except Exception:
                    continue
        if want in found:
            try:
                with open(found[want], "r", encoding="utf-8",
                          errors="replace") as f:
                    return f.read(12000) or "(empty skill)"
            except Exception as exc:
                return f"error: cannot read skill ({exc})"[:300]
        avail = ", ".join(sorted(found)) if found else "none installed"
        return f"error: no skill {want!r} (available: {avail})"[:500]
    except Exception as exc:
        return f"error: skill failed ({exc})"[:300]
