"""
Tool specs derived from the category functions (docstring + signature).

Single source of truth for the prompt side: each spec's description is
the function's docstring first paragraph, its parameters come from the
signature (annotations become JSON types, defaultless params become
required), its tags from ``TOOL_TAGS`` below. :mod:`src.agent.mcp.mcp`
serves the same functions over MCP, and FastMCP derives its own schema
from the same signatures — the two can never drift.

contract, dependencies, who calls it: MCP_TOOL_DEFS/NAMES plus
TAG_NAMES/TOOL_TAGS; depends on the four category modules only;
used by the terminal prompt layer, router, policy, and registry.
"""
import inspect as _inspect

from src.agent.mcp import BrowserUse as _B
from src.agent.mcp import ComputerUse as _C
from src.agent.mcp import PlanningTools as _P
from src.agent.mcp import ShellUse as _S

__all__ = [
    "MCP_TOOL_DEFS",
    "MCP_TOOL_NAMES",
    "TAG_NAMES",
    "TOOL_TAGS",
]

TAG_NAMES = ("ShellUse", "BrowserUse", "ComputerUse", "Main")

TOOL_TAGS = {
    "bash": ("ShellUse", "Main"),
    "python_exec": ("ShellUse", "Main"),
    "read": ("ShellUse", "Main"),
    "write": ("ShellUse", "Main"),
    "edit": ("ShellUse", "Main"),
    "list": ("ShellUse", "Main"),
    "glob": ("ShellUse", "Main"),
    "grep": ("ShellUse", "Main"),
    "move": ("ShellUse",),
    "delete": ("ShellUse",),
    "webfetch": ("BrowserUse", "Main"),
    "websearch": ("BrowserUse", "Main"),
    "open_url": ("BrowserUse",),
    "browser_click": ("BrowserUse",),
    "browser_type": ("BrowserUse",),
    "browser_scroll": ("BrowserUse",),
    "download_file": ("BrowserUse", "ShellUse"),
    "screenshot": ("ComputerUse",),
    "click": ("ComputerUse",),
    "type_text": ("ComputerUse",),
    "press_key": ("ComputerUse",),
    "scroll": ("ComputerUse",),
    "get_active_window": ("ComputerUse",),
    "list_windows": ("ComputerUse",),
    "open_app": ("ComputerUse",),
    "task": ("ShellUse",),
    "question": ("Main",),
    "skill": ("Main",),
}

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}

_MODULES = (_S, _B, _C, _P)


def _description(fn) -> str:
    """First docstring paragraph, one line (the model-facing blurb).

    ..
    """
    doc = _inspect.getdoc(fn) or ""
    lines = []
    for line in doc.splitlines():
        s = line.strip()
        if not s:
            break
        if s != "..":
            lines.append(s)
    return " ".join(lines) or getattr(fn, "__name__", "tool")


def _parameters(fn) -> dict:
    """JSON-schema params from the signature (defaults mean optional).

    ..
    """
    try:
        sig = _inspect.signature(fn)
    except Exception:
        return {"type": "object", "properties": {}}
    props: dict = {}
    required: list = []
    for p in sig.parameters.values():
        if p.kind in (_inspect.Parameter.VAR_POSITIONAL,
                      _inspect.Parameter.VAR_KEYWORD):
            continue
        props[p.name] = {"type": _JSON_TYPES.get(p.annotation, "string")}
        if p.default is _inspect.Parameter.empty:
            required.append(p.name)
    out: dict = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def _spec(name: str, fn) -> dict:
    """One tool spec dict from a category function.

    ..
    """
    return {"name": name,
            "description": _description(fn),
            "parameters": _parameters(fn),
            "tags": list(TOOL_TAGS.get(name, ()))}


def _collect() -> list:
    """Specs in registry order (category __all__ order).

    ..
    """
    defs = []
    for mod in _MODULES:
        for name in getattr(mod, "__all__", ()):
            fn = getattr(mod, name, None)
            if callable(fn):
                defs.append(_spec(name, fn))
    return defs


MCP_TOOL_DEFS = _collect()

MCP_TOOL_NAMES = tuple(t["name"] for t in MCP_TOOL_DEFS)
