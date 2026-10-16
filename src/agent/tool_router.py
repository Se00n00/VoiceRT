"""Semantic tool router: give the model only the tools it needs.

:class:`SemanticRouter` embeds the request once with BAAI/bge-small-en-v1.5
(plain transformers, CLS pooling, CPU by default — milliseconds per turn,
zero VRAM impact) and returns the top-k ops by cosine similarity.
:class:`src.agent.middleware.InjectToolMiddleware` is the pluggable hook:
add it to the deepagents middleware list and it hands that shortlist to
:class:`src.models.chat_model.LocalChatModel` for the current model call
only (a ContextVar, always reset). Anything unresolved — model missing,
offline, empty scores — returns None and the keyword router stays in
charge, so a turn can never break here.

New tools plug in by adding one entry to ``ROUTING_BLURBS`` (kept
separate from the template descriptions on purpose: tuning retrieval
text must not perturb what the model reads).
"""
import threading
from contextvars import ContextVar

import torch
import torch.nn.functional as _F
from transformers import AutoModel, AutoTokenizer

from src.agent.mcp.specs import MCP_TOOL_DEFS, MCP_TOOL_NAMES, TOOL_TAGS

__all__ = [
    "ROUTING_BLURBS",
    "ROUTE_KEYWORDS",
    "ROUTER_MODEL_ID",
    "ALL_TOOL_DEFS",
    "TERMINAL_TOOLS",
    "ALLOWED_OPS",
    "SemanticRouter",
    "current_ops",
    "route_tool_ops",
    "tools_for_request",
    "collect_tools",
    "filter_by_tags",
    "top_k",
    "build_execution_tools",
]

ROUTER_MODEL_ID = "BAAI/bge-small-en-v1.5"

ROUTING_BLURBS: dict[str, str] = {
    "bash": "Run a bash shell command: ls, pwd, cat, echo, git status, "
            "run tests, install packages; background=true for servers and "
            "watchers, job_id to poll a background job.",
    "read": "Read a text file from disk to see its contents: open a file, "
            "show code, display configuration.",
    "write": "Create a new file or overwrite one with given text: save "
             "output to a file.",
    "edit": "Change part of an existing file: fix code, patch a function, "
            "update text, modify configuration.",
    "grep": "Search inside local files for a pattern: find where something "
            "is defined, look for a string in the codebase.",
    "glob": "Find files by name under a folder: locate a file, "
            "glob search.",
    "list": "List files and directories on disk: show folder contents.",
    "move": "Move or rename a file or directory to a new path.",
    "delete": "Delete a file or directory from disk: remove it "
              "permanently.",
    "python_exec": "Run Python code to compute or analyze: arithmetic, "
                   "math, totals, tax, dataframes, plots, calculations.",
    "webfetch": "Fetch and read one specific web page when you already "
                "have its full http or https URL. Use for: open this link, "
                "read this page, quote a heading from a URL.",
    "websearch": "Search the web for information you do not have. Use for: "
                 "what is, find, endpoint, API reference, latest news, "
                 "prices, documentation, discover URLs. Never guess a URL — "
                 "search for it first.",
    "screenshot": "Capture the computer screen to a PNG image file: take "
                  "a screenshot, see what is on the display.",
    "click": "Click the mouse at screen coordinates: press a button, "
             "click on something visible on screen.",
    "type_text": "Type text with the keyboard into the focused window: "
                 "enter text, fill what is open on screen.",
    "press_key": "Press a keyboard key or shortcut combo: Enter, Escape, "
                 "ctrl+c, alt+tab, function keys.",
    "scroll": "Scroll the screen or window up and down with the mouse "
              "wheel.",
    "get_active_window": "Find which window is active and focused: current "
                         "window title and id.",
    "list_windows": "List the open windows on the desktop: see what apps "
                    "are running.",
    "open_app": "Launch a desktop application by name: open a program, "
                "start an app.",
    "open_url": "Open a web page in the default browser for the user to "
                "see.",
    "browser_click": "Click an element on a web page by CSS selector.",
    "browser_type": "Type text into a web page form field by CSS "
                    "selector, optionally submit.",
    "browser_scroll": "Scroll a web page up or down and read what "
                      "appears.",
    "download_file": "Download a file from a URL and save it to disk.",
    "task": "Run background work: launch a command, poll its output.",
    "question": "Clarify with the user: ask, answer, or list pending.",
    "skill": "Load a skill by name to follow its workflow.",
}

ALL_TOOL_DEFS = TERMINAL_TOOLS = list(MCP_TOOL_DEFS)

ALLOWED_OPS = tuple(list(MCP_TOOL_NAMES) + ["done", "write_todos"])

ROUTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    # word-boundary matched (short words especially); keep shell verbs here —
    # routing is not permission, the policy gate still decides deny/confirm.
    "bash": ("run", "execute", "command", "shell", "script",
             "background", "long-running", "long running",
             "in the background", "nohup", "daemon", "watch",
             "poll", "job-", "job status", "still running", "check on",
             "background job", "delete", "remove", "move", "copy",
             "install", "test", "tests", "git",
             "python", "pytest", "build", "compile", "sudo", "apt", "rm",
             "mv", "dd", "chmod", "chown", "curl", "wget", "ssh", "docker",
             "systemctl", "eval", "pip", "npm"),
    "read": ("read", "open", "show", "cat", "display", "contents", "file"),
    "write": ("write", "create", "save", "new file", "make a file"),
    "edit": ("edit", "fix", "patch", "change", "modify", "update",
             "replace", "anchor"),
    "grep": ("grep", "search", "find", "look for", "where", "defined"),
    "glob": ("glob", "search files", "find file", "locate file"),
    "list": ("list", "files", "directory", "folder", "ls", "contents"),
    "move": ("move", "rename"),
    "delete": ("delete", "remove"),
    "python_exec": ("python", "code", "script", "pandas", "numpy", "plot"),
    "webfetch": ("fetch", "webfetch", "url", "link", "website", "webpage",
                 "http", "extract"),
    "websearch": ("searxng", "web search", "websearch", "web", "internet",
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
    "open_url": ("open url", "open link", "open in browser", "open_url"),
    "browser_click": ("browser click", "click on page", "browser_click"),
    "browser_type": ("fill", "fill in", "type into", "input field",
                     "browser_type"),
    "browser_scroll": ("scroll page", "browser_scroll"),
    "download_file": ("download", "download_file"),
    "task": ("task", "subtask", "subagent", "background task"),
    "write_todos": ("todo", "todos", "plan", "checklist", "task list"),
    "question": ("question", "ask", "clarify", "confirmation"),
    "skill": ("skill", "SKILL.md"),
}
"""Intent keywords per op. Matched case-insensitively on word boundaries."""

_ROUTE_DEPS: dict[str, tuple[str, ...]] = {
    # read-before-write is the core workflow: never strand edit/write
    # on a first step with no way to see the file.
    "edit": ("read",),
    "write": ("read",),
    "delete": ("list",),
    "move": ("list",),
}

_ROUTE_FOLLOWUP: tuple[str, ...] = ("read", "list", "glob")
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
    # (bash) are always visible. A task needing a second step (webfetch
    # THEN write) must see write in step one, or the model loops the
    # first tool and gives up (measured). Bash stays because the model
    # reaches for shell constantly — and Groq 400s undeclared tools.
    expanded.update(("read", "write", "list", "bash"))
    ops = sorted(expanded & set(ALLOWED_OPS))
    return ops or None


def tools_for_request(text: str, observation: str = "") -> list:
    """MCP defs filtered by :func:`route_tool_ops`.

    Order kept (defs order). No hits -> full Main-tagged set: small
    models never drown in the ComputerUse/Browser DOM specs.
    """
    ops = route_tool_ops(text, observation)
    if ops is None:
        return [t for t in TERMINAL_TOOLS
                if "Main" in TOOL_TAGS.get(t["name"], ())]
    wanted = set(ops)
    return [t for t in TERMINAL_TOOLS if t["name"] in wanted]


_current_ops: ContextVar[tuple | None] = ContextVar("tool_router_ops", default=None)


def current_ops() -> list | None:
    ops = _current_ops.get()
    return list(ops) if ops else None


def collect_tools(*pools) -> list:
    """Flatten tool pools into one list, deduped by name.

    Later pools win (MCP extras override same-named backend tools);
    first-seen order kept.
    """
    seen: dict[str, object] = {}
    for pool in pools or ():
        for t in pool or ():
            name = str(getattr(t, "name", "") or "")
            if not name:
                continue
            seen[name] = t
    return list(seen.values())


def filter_by_tags(tools: list, tags=None) -> list:
    """Keep tools whose tags intersect ``tags``; None/empty keeps all.

    Tag allow-list hook (refined later); today it passes everything.
    """
    wanted = {str(t) for t in (tags or ()) if str(t).strip()}
    if not wanted:
        return list(tools or [])
    out = []
    for t in tools or []:
        name = str(getattr(t, "name", "") or "")
        if set(TOOL_TAGS.get(name, ())) & wanted:
            out.append(t)
    return out


def top_k(text: str = "", observation: str = "", tools: list | None = None,
          router=None, k: int = 5) -> list | None:
    """Top-K tools for this turn (prompt shortlist), or None for all.

    Semantic router when one is handed in, keyword fallback otherwise.
    None means show everything (recall over precision).
    """
    tools = list(tools or [])
    if not tools:
        return None
    names: list | None = None
    if router is not None:
        try:
            names = router.route(text, observation)
        except Exception:
            names = None
    if names is None:
        try:
            names = route_tool_ops(text, observation)
        except Exception:
            return None
    if names is None:
        return None
    wanted = set(names[:max(1, int(k or 5))])
    picked = [t for t in tools
              if str(getattr(t, "name", "") or "") in wanted]
    return picked or None


def build_execution_tools(*pools, tags=None) -> list:
    """Thin listing layer: collect pools, then tag-filter.

    Semantic top-K stays prompt-side (InjectToolMiddleware); execution
    always sees the full tag-filtered set so calls never 400.
    """
    return filter_by_tags(collect_tools(*pools), tags)


class SemanticRouter:
    def __init__(self, k: int = 3, model_id: str | None = None):
        self.k = max(1, int(k))
        self.model_id = model_id or ROUTER_MODEL_ID
        self._lock = threading.Lock()
        self._tok = None
        self._model = None
        self._tool_names: list = []
        self._tool_matrix = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        with self._lock:
            if self._model is not None:
                return True
            try:
                self._tok = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModel.from_pretrained(self.model_id)
                self._model.eval()
                names, vecs = [], []
                for t in ALL_TOOL_DEFS:
                    name = t.get("name", "")
                    text = ROUTING_BLURBS.get(name, t.get("description", ""))
                    v = self._embed(text)
                    if v is None:
                        return False
                    names.append(name)
                    vecs.append(v)
                with torch.no_grad():
                    self._tool_matrix = _F.normalize(
                        torch.stack(vecs), p=2, dim=1)
                self._tool_names = names
                return True
            except Exception:
                self._tok, self._model = None, None
                return False

    def _embed(self, text: str):
        try:
            ids = self._tok(str(text or "")[:1000], return_tensors="pt",
                            truncation=True, max_length=512)
            with torch.no_grad():
                out = self._model(**ids).last_hidden_state[:, 0]
            return _F.normalize(out, p=2, dim=1)[0]
        except Exception:
            return None

    def route(self, text: str = "", observation: str = "") -> list | None:
        blob = f"{text or ''}\n{observation or ''}".strip()
        if not blob or not self._ensure_loaded():
            return None
        q = self._embed(blob)
        if q is None or self._tool_matrix is None:
            return None
        try:
            with torch.no_grad():
                scores = (self._tool_matrix @ q).tolist()
        except Exception:
            return None
        ranked = sorted(zip(scores, self._tool_names), reverse=True)
        ops = [name for _, name in ranked[:self.k]]
        if not ops:
            return None
        try:
            expanded = set(ops)
            for op in ops:
                expanded.update(_ROUTE_DEPS.get(op, ()))
            if (observation or "").strip():
                expanded.update(_ROUTE_FOLLOWUP)
            # Backbone: the hands (read/write/list) plus the universal
            # fallback (bash) are always visible (see route_tool_ops:
            # a missing write strands webfetch-then-write tasks in a
            # webfetch loop, and Groq 400s calls to undeclared tools).
            expanded.update(("read", "write", "list", "bash"))
            ordered = [n for n in self._tool_names if n in expanded & set(ALLOWED_OPS)]
            return ordered or None
        except Exception:
            return list(ops)
