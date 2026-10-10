"""Semantic tool router: give the model only the tools it needs.

:class:`SemanticRouter` embeds the request once with BAAI/bge-small-en-v1.5
(plain transformers, CLS pooling, CPU by default — milliseconds per turn,
zero VRAM impact) and returns the top-k ops by cosine similarity.
:class:`src.agent.middleware.InjectToolMiddleware` is the pluggable hook:
add it to the deepagents middleware list and it hands that shortlist to
:class:`src.agent.chat_model.LocalChatModel` for the current model call
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

from src.agent.mcp.defs import MCP_TOOL_DEFS
from src.tools.terminal import (
    _ROUTE_DEPS,
    _ROUTE_FOLLOWUP,
    ALLOWED_OPS,
    TERMINAL_TOOLS,
)

__all__ = [
    "ROUTING_BLURBS",
    "ROUTER_MODEL_ID",
    "ALL_TOOL_DEFS",
    "SemanticRouter",
    "current_ops",
]

ROUTER_MODEL_ID = "BAAI/bge-small-en-v1.5"

ROUTING_BLURBS: dict[str, str] = {
    "exec": "Run a bash shell command in the terminal: ls, pwd, cat, echo, "
            "git status, run tests, install packages, move or copy files.",
    "exec_bg": "Start a long-running shell command in the background: "
               "servers, watchers, sleep, training runs.",
    "poll": "Check on a background job started earlier: read its new "
            "output, see if it finished.",
    "read": "Read a text file from disk to see its contents: open a file, "
            "show code, display configuration.",
    "write": "Create a new file or overwrite one with given text: save "
             "output to a file.",
    "edit": "Change part of an existing file: fix code, patch a function, "
            "update text, modify configuration.",
    "grep": "Search inside local files for a pattern: find where something "
            "is defined, look for a string in the codebase.",
    "list": "List files and directories on disk: show folder contents.",
    "python_exec": "Run Python code to compute or analyze: arithmetic, "
                   "math, totals, tax, dataframes, plots, calculations.",
    "fetch": "Fetch and read one specific web page when you already have "
             "its full http or https URL. Use for: open this link, read "
             "this page, quote a heading from a URL.",
    "searxng": "Search the web for information you do not have. Use for: "
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
    "list_directory": "List a directory on disk: show folder contents, "
                      "browse files.",
    "search_files": "Find files by name under a folder: locate a file, "
                    "glob search.",
    "read_file": "Read lines from a text file: show file contents with "
                 "line numbers.",
    "write_file": "Create a new file or overwrite one with given text: "
                  "save output to a file.",
    "edit_file": "Replace exact text inside an existing file: patch code, "
                 "fix a line.",
    "move_file": "Move or rename a file or directory to a new path.",
    "delete_file": "Delete a file or directory from disk: remove it "
                   "permanently.",
    "search_web": "Search the web for information: titles, links and "
                  "snippets on any topic.",
    "open_url": "Open a web page in the default browser for the user to "
                "see.",
    "extract_page": "Read one specific web page as text when you already "
                    "have its full URL.",
    "browser_click": "Click an element on a web page by CSS selector.",
    "browser_type": "Type text into a web page form field by CSS "
                    "selector, optionally submit.",
    "browser_scroll": "Scroll a web page up or down and read what "
                      "appears.",
    "download_file": "Download a file from a URL and save it to disk.",
}

ALL_TOOL_DEFS = TERMINAL_TOOLS + MCP_TOOL_DEFS

_current_ops: ContextVar[tuple | None] = ContextVar("tool_router_ops", default=None)


def current_ops() -> list | None:
    ops = _current_ops.get()
    return list(ops) if ops else None


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
            ordered = [n for n in self._tool_names if n in expanded & set(ALLOWED_OPS)]
            return ordered or None
        except Exception:
            return list(ops)
