"""
MCP stdio registry: voice-tools (ShellUse + BrowserUse + ComputerUse).

Thin registry only — every implementation lives in its category module:
- :mod:`src.agent.mcp.ShellUse` — bash, python_exec, read, write, edit,
  list, glob, grep, move, delete,
- :mod:`src.agent.mcp.BrowserUse` — webfetch, websearch, open_url,
  browser_click, browser_type, browser_scroll, download_file,
- :mod:`src.agent.mcp.ComputerUse` — screenshot, click, type_text,
  press_key, scroll, get_active_window, list_windows, open_app,
- :mod:`src.agent.mcp.PlanningTools` — task, question, skill.

Tags mirror :mod:`src.agent.mcp.specs` (FastMCP has no ``tags=`` param,
so they travel in ``meta``). One line per tool: ``app.add_tool`` with
name + tags.

Run standalone over stdio::

    PYTHONPATH=. python -m src.agent.mcp.mcp
"""
from mcp.server.fastmcp import FastMCP

from src.agent.mcp import BrowserUse as _B
from src.agent.mcp import ComputerUse as _C
from src.agent.mcp import PlanningTools as _P
from src.agent.mcp import ShellUse as _S
from src.agent.mcp.specs import TOOL_TAGS

app = FastMCP("voice-tools")


def _tags(name: str) -> dict:
    """Tag metadata for one tool (mirrors defs).
    
    """
    return {"tags": list(TOOL_TAGS.get(name, ()))}


# -- ShellUse ------------------------------------------------------------
app.add_tool(_S.bash, name="bash", meta=_tags("bash"))
app.add_tool(_S.python_exec, name="python_exec", meta=_tags("python_exec"))
app.add_tool(_S.read, name="read", meta=_tags("read"))
app.add_tool(_S.write, name="write", meta=_tags("write"))
app.add_tool(_S.edit, name="edit", meta=_tags("edit"))
app.add_tool(_S.list, name="list", meta=_tags("list"))
app.add_tool(_S.glob, name="glob", meta=_tags("glob"))
app.add_tool(_S.grep, name="grep", meta=_tags("grep"))
app.add_tool(_S.move, name="move", meta=_tags("move"))
app.add_tool(_S.delete, name="delete", meta=_tags("delete"))

# -- BrowserUse ----------------------------------------------------------
app.add_tool(_B.webfetch, name="webfetch", meta=_tags("webfetch"))
app.add_tool(_B.websearch, name="websearch", meta=_tags("websearch"))
app.add_tool(_B.open_url, name="open_url", meta=_tags("open_url"))
app.add_tool(_B.browser_click, name="browser_click", meta=_tags("browser_click"))
app.add_tool(_B.browser_type, name="browser_type", meta=_tags("browser_type"))
app.add_tool(_B.browser_scroll, name="browser_scroll", meta=_tags("browser_scroll"))
app.add_tool(_B.download_file, name="download_file", meta=_tags("download_file"))

# -- ComputerUse ---------------------------------------------------------
app.add_tool(_C.screenshot, name="screenshot", meta=_tags("screenshot"))
app.add_tool(_C.click, name="click", meta=_tags("click"))
app.add_tool(_C.type_text, name="type_text", meta=_tags("type_text"))
app.add_tool(_C.press_key, name="press_key", meta=_tags("press_key"))
app.add_tool(_C.scroll, name="scroll", meta=_tags("scroll"))
app.add_tool(_C.get_active_window, name="get_active_window", meta=_tags("get_active_window"))
app.add_tool(_C.list_windows, name="list_windows", meta=_tags("list_windows"))
app.add_tool(_C.open_app, name="open_app", meta=_tags("open_app"))

# -- PlanningTools -------------------------------------------------------
app.add_tool(_P.task, name="task", meta=_tags("task"))
app.add_tool(_P.question, name="question", meta=_tags("question"))
app.add_tool(_P.skill, name="skill", meta=_tags("skill"))


if __name__ == "__main__":
    app.run()
