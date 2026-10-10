"""MCP stdio server: voice-tools + computer + filesystem + web.

Thin FastMCP wrappers only — every implementation lives in its category
module (the single definition):
- :mod:`src.agent.mcp.tools` — ``python_exec`` / ``fetch`` / ``web_search``
  (arg names are the agent-facing ones: ``code`` / ``path`` / ``pattern``),
- :mod:`src.agent.mcp.computer` — screenshot, click, type_text,
  press_key, scroll, get_active_window, list_windows, open_app,
- :mod:`src.agent.mcp.filesystem` — list_directory, search_files,
  read_file, write_file, edit_file, move_file, delete_file,
- :mod:`src.agent.mcp.web` — search_web, open_url, extract_page,
  browser_click, browser_type, browser_scroll, download_file.

Run standalone over stdio::

    PYTHONPATH=. python -m src.agent.mcp.server
"""
from mcp.server.fastmcp import FastMCP

from src.agent.mcp import tools as T
from src.agent.mcp import computer as C
from src.agent.mcp import filesystem as F
from src.agent.mcp import web as W

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


# -- computer ------------------------------------------------------------


@mcp.tool(name="screenshot")
def screenshot_tool(path: str = "") -> str:
    """Capture the screen to a PNG file; returns the saved path."""
    return C.screenshot(path)


@mcp.tool(name="click")
def click_tool(x: int, y: int, button: str = "left") -> str:
    """Click at screen coordinates (x, y, button left/middle/right)."""
    return C.click(x, y, button)


@mcp.tool(name="type_text")
def type_text_tool(text: str, interval: float = 0.0) -> str:
    """Type text into the focused window."""
    return C.type_text(text, interval)


@mcp.tool(name="press_key")
def press_key_tool(key: str) -> str:
    """Press a key or +-separated combo (e.g. 'ctrl+alt+t')."""
    return C.press_key(key)


@mcp.tool(name="scroll")
def scroll_tool(clicks: int, x: int = -1, y: int = -1) -> str:
    """Scroll vertically (clicks > 0 up, < 0 down), optionally at x/y."""
    return C.scroll(clicks, x, y)


@mcp.tool(name="get_active_window")
def get_active_window_tool() -> str:
    """Return the active window id/pid/name."""
    return C.get_active_window()


@mcp.tool(name="list_windows")
def list_windows_tool(limit: int = 50) -> str:
    """List visible windows (id + name)."""
    return C.list_windows(limit)


@mcp.tool(name="open_app")
def open_app_tool(name: str, args: str = "") -> str:
    """Launch an application by executable name."""
    return C.open_app(name, args)


# -- filesystem ----------------------------------------------------------


@mcp.tool(name="list_directory")
def list_directory_tool(path: str = ".", show_hidden: bool = False,
                         limit: int = 200) -> str:
    """List a directory (dirs first, trailing /)."""
    return F.list_directory(path, show_hidden, limit)


@mcp.tool(name="search_files")
def search_files_tool(root: str = ".", pattern: str = "*",
                       limit: int = 50) -> str:
    """Glob-search under root (recursive); relative paths back."""
    return F.search_files(root, pattern, limit)


@mcp.tool(name="read_file")
def read_file_tool(path: str, offset: int = 1, limit: int = 100,
                    max_chars: int = 12000) -> str:
    """Read text lines (1-based offset, at most limit)."""
    return F.read_file(path, offset, limit, max_chars)


@mcp.tool(name="write_file")
def write_file_tool(path: str, content: str,
                     make_parents: bool = True) -> str:
    """Write (create or overwrite) a text file."""
    return F.write_file(path, content, make_parents)


@mcp.tool(name="edit_file")
def edit_file_tool(path: str, old_string: str, new_string: str,
                    replace_all: bool = False) -> str:
    """Exact string replacement in a text file."""
    return F.edit_file(path, old_string, new_string, replace_all)


@mcp.tool(name="move_file")
def move_file_tool(src: str, dst: str) -> str:
    """Move/rename a file or directory."""
    return F.move_file(src, dst)


@mcp.tool(name="delete_file")
def delete_file_tool(path: str, recursive: bool = False) -> str:
    """Delete a file, symlink, or directory (dirs need recursive)."""
    return F.delete_file(path, recursive)


# -- web -----------------------------------------------------------------


@mcp.tool(name="search_web")
def search_web_tool(query: str, count: int = 5) -> str:
    """Web search, returns titles/URLs/snippets."""
    return W.search_web(query, count)


@mcp.tool(name="open_url")
def open_url_tool(url: str) -> str:
    """Open a URL in the OS default browser."""
    return W.open_url(url)


@mcp.tool(name="extract_page")
def extract_page_tool(url: str, max_chars: int = 6000) -> str:
    """Extract a web page URL as text."""
    return W.extract_page(url, max_chars)


@mcp.tool(name="browser_click")
def browser_click_tool(url: str, selector: str, timeout: int = 15,
                        headless: bool = True) -> str:
    """Click a CSS selector on a page; report the page after."""
    return W.browser_click(url, selector, timeout, headless)


@mcp.tool(name="browser_type")
def browser_type_tool(url: str, selector: str, text: str,
                       submit: bool = False, timeout: int = 15,
                       headless: bool = True) -> str:
    """Fill text into a CSS selector on a page (+Enter if submit)."""
    return W.browser_type(url, selector, text, submit, timeout, headless)


@mcp.tool(name="browser_scroll")
def browser_scroll_tool(url: str, direction: str = "down",
                         pixels: int = 800, timeout: int = 15,
                         headless: bool = True) -> str:
    """Scroll a page up/down by pixels; report the page after."""
    return W.browser_scroll(url, direction, pixels, timeout, headless)


@mcp.tool(name="download_file")
def download_file_tool(url: str, dest: str, max_bytes: int = 50_000_000,
                        timeout: int = 60) -> str:
    """Download a URL to a local path."""
    return W.download_file(url, dest, max_bytes, timeout)


if __name__ == "__main__":
    mcp.run()
