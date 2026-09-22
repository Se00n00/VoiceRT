"""Agent-local MCP package: canonical tools + stdio server.

- ``tools``: single-definition implementations (scoped fs, docker exec,
  python exec, fetch, searxng).
- ``server``: thin FastMCP wrappers for external (stdio) consumers.
"""
from src.agent.mcp import tools
from src.agent.mcp.tools import (
    DEFAULT_ROOTS,
    default_roots,
    docker_exec,
    fetch,
    fs_edit,
    fs_grep,
    fs_list,
    fs_read,
    fs_write,
    python_exec,
    searxng_search,
    searxng_url,
)

__all__ = [
    "tools",
    "DEFAULT_ROOTS",
    "default_roots",
    "docker_exec",
    "fetch",
    "fs_edit",
    "fs_grep",
    "fs_list",
    "fs_read",
    "fs_write",
    "python_exec",
    "searxng_search",
    "searxng_url",
]
