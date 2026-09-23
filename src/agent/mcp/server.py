"""MCP stdio server: python_exec, fetch, web_search.

Thin FastMCP wrappers only — every implementation lives in
:mod:`src.agent.mcp.tools` (the single definition). Arg names are the
agent-facing ones (``code`` / ``path`` / ``pattern``), matching the
native model-facing defs and :meth:`LocalChatModel._builtin_call`
(``searxng`` ops arrive here as ``web_search``).

Run standalone over stdio::

    PYTHONPATH=. python -m src.agent.mcp.server
"""
from mcp.server.fastmcp import FastMCP

from src.agent.mcp import tools as T

mcp = FastMCP("voice-tools")


@mcp.tool(name="python_exec")
def python_exec_tool(code: str, timeout: int = 30) -> str:
    """Run Python code on this host and return its output."""
    return T.python_exec(code, timeout=timeout)


@mcp.tool(name="fetch")
def fetch_tool(path: str) -> str:
    """Fetch a web page URL and return its text."""
    return T.fetch(path)


@mcp.tool(name="web_search")
def web_search_tool(pattern: str) -> str:
    """Web search, returns titles/URLs/snippets."""
    return T.web_search(pattern)


if __name__ == "__main__":
    mcp.run()
