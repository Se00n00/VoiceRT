"""
Stateful MCP sessions for the agent's extra tools, one pool per process.

contract, dependencies, who calls it: McpSessions opens one persistent
ClientSession per configured server and binds LangChain tools to those
sessions (no per-call reconnect); depends only on langchain-mcp-adapters
plus src.agent.mcp.config; used by console_cli, VoiceAgent, eval lib.
"""

import contextlib

__all__ = ["McpSessions", "open_sessions"]


class McpSessions:
    """One persistent session per MCP server, tools bound to them."""

    def __init__(self, connections=None, tool_name_prefix=False) -> None:
        self._client = None
        self._connections = dict(connections or {})
        self._prefix = bool(tool_name_prefix)
        self._stack = None
        self._sessions: dict = {}
        self._tools: list = []

    @property
    def tools(self) -> list:
        """Tools bound to the open sessions (empty until started).

        ..
        """
        return list(self._tools)

    @property
    def servers(self) -> list:
        """Configured server names in connection order.

        ..
        """
        return list(self._connections.keys())

    async def start(self) -> "McpSessions":
        """Open every session and bind its tools. Returns self.

        ..
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        if self._stack is not None:
            return self
        client = MultiServerMCPClient(
            dict(self._connections), tool_name_prefix=self._prefix)
        stack = contextlib.AsyncExitStack()
        await stack.__aenter__()
        sessions: dict = {}
        tools: list = []
        try:
            for name in self._connections:
                session = await stack.enter_async_context(
                    client.session(name))
                sessions[name] = session
                tools.extend(await load_mcp_tools(
                    session, server_name=name,
                    tool_name_prefix=self._prefix))
        except Exception:
            try:
                await stack.__aexit__(None, None, None)
            except Exception:
                pass
            raise
        self._client = client
        self._stack = stack
        self._sessions = sessions
        self._tools = tools
        return self

    async def aclose(self) -> None:
        """Close every session. Never raises.

        ..
        """
        stack, self._stack = self._stack, None
        self._sessions = {}
        self._tools = []
        if stack is not None:
            try:
                await stack.__aexit__(None, None, None)
            except Exception:
                pass

    async def __aenter__(self) -> "McpSessions":
        """Open on context entry. Returns self.

        ..
        """
        return await self.start()

    async def __aexit__(self, *exc) -> None:
        """Close on context exit. Never raises.

        ..
        """
        await self.aclose()


async def open_sessions(config_path=None) -> McpSessions:
    """Configured servers -> started stateful pool. Returns the pool.

    ..
    """
    from src.agent.mcp.config import load_connections

    connections, prefix = load_connections(config_path)
    return await McpSessions(connections, tool_name_prefix=prefix).start()
