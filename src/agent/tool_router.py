"""Semantic tool router: give the model only the tools it needs.

:class:`SemanticRouter` embeds the request once with BAAI/bge-small-en-v1.5
(plain transformers, CLS pooling, CPU by default — milliseconds per turn,
zero VRAM impact) and returns the top-k ops by cosine similarity.
:class:`InjectToolMiddleware` is the pluggable hook: add it to the
deepagents middleware list and it hands that shortlist to
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
from typing import Any

from langchain.agents.middleware import AgentMiddleware

__all__ = [
    "ROUTING_BLURBS",
    "ROUTER_MODEL_ID",
    "InjectToolMiddleware",
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
}

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
                from transformers import AutoModel, AutoTokenizer

                from src.tools.terminal import TERMINAL_TOOLS

                self._tok = AutoTokenizer.from_pretrained(self.model_id)
                self._model = AutoModel.from_pretrained(self.model_id)
                self._model.eval()
                names, vecs = [], []
                for t in TERMINAL_TOOLS:
                    name = t.get("name", "")
                    text = ROUTING_BLURBS.get(name, t.get("description", ""))
                    v = self._embed(text)
                    if v is None:
                        return False
                    names.append(name)
                    vecs.append(v)
                import torch

                with torch.no_grad():
                    import torch.nn.functional as _F

                    self._tool_matrix = _F.normalize(
                        torch.stack(vecs), p=2, dim=1)
                self._tool_names = names
                return True
            except Exception:
                self._tok, self._model = None, None
                return False

    def _embed(self, text: str):
        try:
            import torch

            ids = self._tok(str(text or "")[:1000], return_tensors="pt",
                            truncation=True, max_length=512)
            with torch.no_grad():
                out = self._model(**ids).last_hidden_state[:, 0]
            import torch.nn.functional as _F

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
            import torch

            with torch.no_grad():
                scores = (self._tool_matrix @ q).tolist()
        except Exception:
            return None
        ranked = sorted(zip(scores, self._tool_names), reverse=True)
        ops = [name for _, name in ranked[:self.k]]
        if not ops:
            return None
        try:
            from src.tools.terminal import _ROUTE_DEPS, _ROUTE_FOLLOWUP

            expanded = set(ops)
            for op in ops:
                expanded.update(_ROUTE_DEPS.get(op, ()))
            if (observation or "").strip():
                expanded.update(_ROUTE_FOLLOWUP)
            from src.tools.terminal import ALLOWED_OPS

            ordered = [n for n in self._tool_names if n in expanded & set(ALLOWED_OPS)]
            return ordered or None
        except Exception:
            return list(ops)


def _request_text(request: Any) -> tuple:
    text, obs = "", ""
    try:
        from langchain_core.messages import HumanMessage, ToolMessage

        for m in request.messages or []:
            if isinstance(m, HumanMessage) and isinstance(m.content, str):
                text = m.content
            elif isinstance(m, ToolMessage) and isinstance(m.content, str):
                obs += "\n" + m.content
    except Exception:
        pass
    return text, obs


class InjectToolMiddleware(AgentMiddleware):
    def __init__(self, router: SemanticRouter | None = None, **kwargs):
        self.router = router or SemanticRouter(**kwargs)

    def _select(self, request: Any) -> list | None:
        try:
            text, obs = _request_text(request)
            return self.router.route(text, obs)
        except Exception:
            return None

    def wrap_model_call(self, request, handler):
        ops = self._select(request)
        if not ops:
            return handler(request)
        tok = _current_ops.set(tuple(ops))
        try:
            return handler(request)
        finally:
            _current_ops.reset(tok)

    async def awrap_model_call(self, request, handler):
        ops = self._select(request)
        if not ops:
            return await handler(request)
        tok = _current_ops.set(tuple(ops))
        try:
            return await handler(request)
        finally:
            _current_ops.reset(tok)
