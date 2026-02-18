"""
Generic MCP (Model Context Protocol) tool implementation.

This module provides classes for connecting to any MCP server and exposing its tools.
"""

import asyncio
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, ClassVar

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult

from open_instruct import logger_utils
from open_instruct.environments.base import EnvCall, StepResult
from open_instruct.environments.tools.utils import BaseEnvConfig, Tool, coerce_args, log_env_call

logger = logger_utils.setup_logger(__name__)


class MCPTransport(StrEnum):
    """Transport types for MCP connections."""

    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"


def _get_mcp_client(transport: MCPTransport, server_url: str, command: str, args: list[str], env: dict[str, str]):
    """Get the appropriate MCP client context manager based on transport type."""
    if transport == MCPTransport.HTTP:
        return streamablehttp_client(server_url)
    elif transport == MCPTransport.SSE:
        return sse_client(server_url)
    elif transport == MCPTransport.STDIO:
        return stdio_client(StdioServerParameters(command=command, args=args, env=env or None))
    raise ValueError(f"Unknown transport type: {transport}")


async def _call_mcp_tool(client_context, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Call a tool on an MCP server, returning the raw result."""
    async with client_context as (read_stream, write_stream, *_):  # noqa: SIM117
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)


def _extract_mcp_result(result: CallToolResult) -> str:
    """Extract text from an MCP tool result."""
    # TODO: handle other content types (image, audio, resource, resource_link)
    if not hasattr(result, "content"):
        return str(result)
    parts = []
    for c in result.content:
        if getattr(c, "type", "text") != "text":
            raise ValueError(f"Unsupported MCP content type: {c.type}")
        parts.append(c.text)
    return "\n".join(parts)


class GenericMCPTool(Tool):
    """
    A generic MCP (Model Context Protocol) tool that connects to any MCP server.

    This tool connects to an MCP server and exposes a single tool from it.
    """

    config_name = "generic_mcp"

    def __init__(
        self,
        call_name: str,
        server_url: str = "",
        transport: str = "http",
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        tool_name: str | None = None,
        tool_info: dict[str, Any] | None = None,
    ) -> None:
        self.call_name = call_name
        self.server_url = server_url
        self.transport = MCPTransport(transport)
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.tool_name = tool_name
        self._tool_info = tool_info or {}

        if self.transport in (MCPTransport.HTTP, MCPTransport.SSE) and not server_url:
            raise ValueError(f"server_url is required for {self.transport.value} transport")
        if self.transport == MCPTransport.STDIO and not command:
            raise ValueError("command is required for stdio transport")

    @property
    def description(self) -> str:
        """Return the tool's description from MCP."""
        return self._tool_info.get("description", "")

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool's parameter schema from MCP."""
        return self._tool_info.get("input_schema", {"type": "object", "properties": {}})

    async def step(self, call: EnvCall) -> StepResult:
        """Call the MCP tool with retry logic."""
        kwargs = coerce_args(self.parameters, call.args)
        if self.tool_name is None:
            raise ValueError("tool_name must be set before execute")
        start_time = time.time()
        last_error: str | None = None
        retryable_exceptions = (ConnectionError, TimeoutError, OSError)

        for attempt in range(self.max_retries):
            try:
                client = _get_mcp_client(self.transport, self.server_url, self.command, self.args, self.env)
                raw_result = await asyncio.wait_for(
                    _call_mcp_tool(client, self.tool_name, kwargs), timeout=self.timeout
                )

                output = _extract_mcp_result(raw_result)

                result = StepResult(result=output, metadata={"runtime": time.time() - start_time})
                log_env_call(self.tool_name, str(kwargs), result)
                return result

            except asyncio.TimeoutError:
                result = StepResult(
                    result="",
                    metadata={
                        "error": f"Timeout after {self.timeout} seconds",
                        "timeout": True,
                        "runtime": time.time() - start_time,
                    },
                )
                log_env_call(self.tool_name, str(kwargs), result)
                return result

            except retryable_exceptions as e:
                last_error = str(e)
                if attempt + 1 < self.max_retries:
                    sleep_time = self.retry_backoff * (2**attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{self.max_retries} for MCP tool {self.tool_name}: {e}. "
                        f"Sleeping {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time)
                    continue

            except Exception as e:
                result = StepResult(result="", metadata={"error": str(e), "runtime": time.time() - start_time})
                log_env_call(self.tool_name, str(kwargs), result)
                return result

        result = StepResult(
            result="",
            metadata={"error": last_error or "Unknown error after retries", "runtime": time.time() - start_time},
        )
        log_env_call(self.tool_name, str(kwargs), result)
        return result


@dataclass
class GenericMCPToolConfig(BaseEnvConfig):
    """Configuration for generic MCP tools.

    Connects to an MCP server and exposes its tools. Use `tool_name` to specify
    which tool to expose. Use `expand_tools()` to discover all tools from the server.
    """

    tool_class: ClassVar[type[Tool]] = GenericMCPTool

    server_url: str = ""
    transport: str = "http"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 60
    max_retries: int = 3
    retry_backoff: float = 0.5
    tool_name: str | None = None
    tool_info: dict[str, Any] = field(default_factory=dict)

    async def discover_tools(self) -> dict[str, dict[str, Any]]:
        """Discover tools from the mcp server."""
        client = _get_mcp_client(MCPTransport(self.transport), self.server_url, self.command, self.args, self.env)
        async with client as (read_stream, write_stream, *_):  # noqa: SIM117
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                resp = await session.list_tools()
                return {
                    t.name: {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                    }
                    for t in resp.tools
                }

    async def expand_tools(self) -> list["GenericMCPToolConfig"]:
        """Expand this config to create one config per discovered tool.

        If `tool_name` is already specified, returns [self] unchanged.
        Otherwise, discovers all tools and returns a config for each.

        TODO: right now, we just do tool discovery at the start, and assume its unchanged over training. This is not guaranteed to be true for MCP servers!
        """
        if self.tool_name is not None:
            return [self]

        discovered = await self.discover_tools()
        logger.info(f"MCP server discovered {len(discovered)} tools: {list(discovered.keys())}")
        return [replace(self, tool_name=name, tool_info=info) for name, info in discovered.items()]
