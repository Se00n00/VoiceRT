"""MCP stdio server: scoped filesystem, docker exec, python, web.

Thin FastMCP wrappers only — every implementation lives in
:mod:`src.agent.mcp.tools` (the single definition). Policy here is
autonomous (deny breakers, auto-approve the rest); the interactive
confirm gate lives in the turn-scoped local toolset instead.

Run standalone over stdio::

    PYTHONPATH=. python -m src.agent.mcp.server
"""
from mcp.server.fastmcp import FastMCP

from src.agent.mcp import tools as T

mcp = FastMCP("voice-tools")


@mcp.tool(name="read")
def read_tool(path: str) -> str:
    """Read a text file under ~/Documents or ~/Projects."""
    return T.fs_read(path)


@mcp.tool(name="write")
def write_tool(path: str, text: str) -> str:
    """Write text to a file under ~/Documents or ~/Projects."""
    return T.fs_write(path, text)


@mcp.tool(name="edit")
def edit_tool(path: str, anchor: str, text: str) -> str:
    """Anchored patch: replace `anchor` verbatim in the file with `text`."""
    return T.fs_edit(path, anchor, text)


@mcp.tool(name="list")
def list_tool(path: str = ".") -> str:
    """List directory entries under ~/Documents or ~/Projects."""
    return T.fs_list(path)


@mcp.tool(name="grep")
def grep_tool(pattern: str, path: str = ".") -> str:
    """Grep a pattern across files under ~/Documents or ~/Projects."""
    return T.fs_grep(pattern, path)


@mcp.tool(name="docker_exec")
def docker_exec_tool(command: str, timeout: int = 30) -> str:
    """Run one command inside an isolated container (timeout, no shell)."""
    return T.docker_exec(command, timeout=timeout)


@mcp.tool(name="python_exec")
def python_exec_tool(code: str, timeout: int = 30) -> str:
    """Run Python code inside an isolated container (no shell, no host files)."""
    return T.python_exec(code, timeout=timeout)


@mcp.tool(name="fetch")
def fetch_tool(url: str) -> str:
    """Fetch a web page (http/https) and return its text."""
    return T.fetch(url)


@mcp.tool(name="searxng")
def searxng_tool(query: str) -> str:
    """Web search via the local SearXNG instance."""
    return T.searxng_search(query)


if __name__ == "__main__":
    mcp.run()
