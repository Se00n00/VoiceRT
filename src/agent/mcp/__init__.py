"""Agent-local MCP package: extra tools + stdio server + client.

- ``tools``: single-definition implementations (``python_exec``,
  ``fetch``, ``web_search``).
- ``computer``: OS-level GUI + apps (``screenshot``, ``click``,
  ``type_text``, ``press_key``, ``scroll``, ``get_active_window``,
  ``list_windows``, ``open_app``).
- ``filesystem``: host paths (``list_directory``, ``search_files``,
  ``read_file``, ``write_file``, ``edit_file``, ``move_file``,
  ``delete_file``).
- ``web``: browser + download (``search_web``, ``open_url``,
  ``extract_page``, ``browser_click``, ``browser_type``,
  ``browser_scroll``, ``download_file``).
- ``defs``: model-facing specs (``MCP_TOOL_DEFS``/``MCP_TOOL_NAMES``).
- ``server``: thin FastMCP wrappers for stdio consumers.
- ``client``: stdio loader returning LangChain tools for the agent.
"""
from src.agent.mcp import computer
from src.agent.mcp import defs
from src.agent.mcp import filesystem
from src.agent.mcp import tools
from src.agent.mcp import web
from src.agent.mcp.computer import (
    click,
    get_active_window,
    list_windows,
    open_app,
    press_key,
    screenshot,
    scroll,
    type_text,
)
from src.agent.mcp.defs import MCP_TOOL_DEFS, MCP_TOOL_NAMES
from src.agent.mcp.filesystem import (
    delete_file,
    edit_file,
    list_directory,
    move_file,
    read_file,
    search_files,
    write_file,
)
from src.agent.mcp.tools import (
    fetch,
    python_exec,
    web_search,
)
from src.agent.mcp.web import (
    browser_click,
    browser_scroll,
    browser_type,
    download_file,
    extract_page,
    open_url,
    search_web,
)

__all__ = [
    "tools",
    "computer",
    "filesystem",
    "web",
    "defs",
    "MCP_TOOL_DEFS",
    "MCP_TOOL_NAMES",
    "fetch",
    "python_exec",
    "web_search",
    "screenshot",
    "click",
    "type_text",
    "press_key",
    "scroll",
    "get_active_window",
    "list_windows",
    "open_app",
    "list_directory",
    "search_files",
    "read_file",
    "write_file",
    "edit_file",
    "move_file",
    "delete_file",
    "search_web",
    "open_url",
    "extract_page",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "download_file",
]
