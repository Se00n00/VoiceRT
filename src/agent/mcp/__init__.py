"""
Agent-local MCP package: category tools + registry + client.

- ``ShellUse``: bash, python_exec, read, write, edit, list, glob,
  grep, move, delete.
- ``BrowserUse``: webfetch, websearch, open_url, browser_click,
  browser_type, browser_scroll, download_file.
- ``ComputerUse``: screenshot, click, type_text, press_key, scroll,
  get_active_window, list_windows, open_app.
- ``PlanningTools``: task, question, skill (todos live in middleware).
- ``specs``: model-facing specs derived from docstrings + signatures
  (``MCP_TOOL_DEFS``/``MCP_TOOL_NAMES``/``TAG_NAMES``/``TOOL_TAGS``).
- ``mcp``: thin FastMCP registry (``app.add_tool`` per tool).
- ``client``/``session``/``config``: stdio loading for the agent.
"""
from src.agent.mcp import BrowserUse
from src.agent.mcp import ComputerUse
from src.agent.mcp import PlanningTools
from src.agent.mcp import ShellUse
from src.agent.mcp import specs
from src.agent.mcp.BrowserUse import (
    browser_click,
    browser_scroll,
    browser_type,
    download_file,
    open_url,
    webfetch,
    websearch,
)
from src.agent.mcp.ComputerUse import (
    click,
    get_active_window,
    list_windows,
    open_app,
    press_key,
    screenshot,
    scroll,
    type_text,
)
from src.agent.mcp.specs import MCP_TOOL_DEFS, MCP_TOOL_NAMES, TAG_NAMES, TOOL_TAGS
from src.agent.mcp.PlanningTools import (
    question,
    skill,
    task,
)
from src.agent.mcp.ShellUse import (
    bash,
    delete,
    edit,
    glob,
    grep,
    list,
    move,
    python_exec,
    read,
    write,
)

__all__ = [
    "ShellUse",
    "BrowserUse",
    "ComputerUse",
    "PlanningTools",
    "specs",
    "MCP_TOOL_DEFS",
    "MCP_TOOL_NAMES",
    "TAG_NAMES",
    "TOOL_TAGS",
    "bash",
    "python_exec",
    "read",
    "write",
    "edit",
    "list",
    "glob",
    "grep",
    "move",
    "delete",
    "webfetch",
    "websearch",
    "open_url",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "download_file",
    "screenshot",
    "click",
    "type_text",
    "press_key",
    "scroll",
    "get_active_window",
    "list_windows",
    "open_app",
    "task",
    "question",
    "skill",
]
