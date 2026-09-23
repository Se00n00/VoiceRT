"""MCP client: the agent's extra tools, served by configured servers.

:func:`load_extra_tools` reads the MCP server configs (see
:mod:`src.agent.mcp.config`), spawns each enabled server, and returns
``(client, tools)`` — LangChain tools backed by those servers. Keep the
client alive as long as the tools are used (each tool call opens a
session through it); :class:`src.main.VoiceAgent` holds it on
``self._mcp_client``.

Loop-safe: loading is async under the hood, but this entry point works
from sync code with or without a running event loop (a helper thread
gets a fresh loop when the caller's thread already runs one).
"""
import asyncio
import concurrent.futures

from src.agent.mcp.config import load_connections

__all__ = ["load_extra_tools"]


def _await_sync(factory, timeout_s: float = 180.0):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(
            lambda: asyncio.run(factory())).result(timeout=timeout_s)


def load_extra_tools(config_path=None, timeout_s: float = 180.0):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections, prefix = load_connections(config_path)
    client = MultiServerMCPClient(connections, tool_name_prefix=prefix)

    async def _load():
        return await client.get_tools()

    tools = _await_sync(_load, timeout_s=timeout_s)
    return client, list(tools)
