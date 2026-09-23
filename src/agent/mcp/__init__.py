"""Agent-local MCP package: extra tools + stdio server + client.

- ``tools``: single-definition implementations (``python_exec``,
  ``fetch``, ``web_search``).
- ``server``: thin FastMCP wrappers for stdio consumers.
- ``client``: stdio loader returning LangChain tools for the agent.
"""
from src.agent.mcp import tools
from src.agent.mcp.tools import (
    fetch,
    python_exec,
    web_search,
)

__all__ = [
    "tools",
    "fetch",
    "python_exec",
    "web_search",
]
