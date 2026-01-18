"""
Generic MCP (Model Context Protocol) tool implementation.

This module provides classes for connecting to any MCP server and exposing its tools.
"""

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from open_instruct import logger_utils
from open_instruct.tools.utils import BaseToolConfig, Tool, ToolOutput

logger = logger_utils.setup_logger(__name__)


def _truncate(text: str, max_length: int = 500) -> str:
    """Truncate text for logging, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [{len(text) - max_length} more chars]"


def _log_tool_call(tool_name: str, input_text: str, output: ToolOutput) -> None:
    """Log a tool call at DEBUG level with truncated input/output."""
    logger.debug(
        f"Tool '{tool_name}' called:\n"
        f"  Input: {_truncate(input_text)}\n"
        f"  Output: {_truncate(output.output)}\n"
        f"  Error: {output.error or 'None'}\n"
        f"  Runtime: {output.runtime:.3f}s, Timeout: {output.timeout}"
    )


class MCPTransport(str, Enum):
    """Transport types for MCP connections."""

    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"


class MCPToolFactory:
    """
    Handles MCP server connections, tool discovery, and tool calls.

    Used internally by GenericMCPTool and GenericMCPToolConfig.
    """

    def __init__(
        self,
        server_url: str | None = None,
        transport: MCPTransport | str = MCPTransport.HTTP,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        """Initialize an MCPToolFactory.

        Args:
            server_url: URL for HTTP/SSE transport (e.g., "http://localhost:8000/mcp").
            transport: Transport type (http, sse, or stdio).
            command: Command to run for stdio transport.
            args: Arguments for the stdio command.
            env: Environment variables for stdio transport.
            timeout: Timeout in seconds for tool calls.
            max_retries: Maximum number of retries for transient errors.
            retry_backoff: Backoff factor for retries (uses exponential backoff).
        """
        self.server_url = server_url
        self.transport = MCPTransport(transport) if isinstance(transport, str) else transport
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Validate configuration
        if self.transport in (MCPTransport.HTTP, MCPTransport.SSE) and not server_url:
            raise ValueError(f"server_url is required for {self.transport.value} transport")
        if self.transport == MCPTransport.STDIO and not command:
            raise ValueError("command is required for stdio transport")

        # Cache for discovered tools
        self._discovered_tools: dict[str, dict[str, Any]] | None = None

    def _get_client_context(self):
        """Get the appropriate client context manager based on transport type."""
        if self.transport == MCPTransport.HTTP:
            return streamablehttp_client(self.server_url)
        elif self.transport == MCPTransport.SSE:
            return sse_client(self.server_url)
        elif self.transport == MCPTransport.STDIO:
            server_params = StdioServerParameters(
                command=self.command, args=self.args, env=self.env if self.env else None
            )
            return stdio_client(server_params)
        else:
            raise ValueError(f"Unknown transport type: {self.transport}")

    async def _discover_tools_async(self) -> dict[str, dict[str, Any]]:
        """Discover available tools from the MCP server."""
        async with self._get_client_context() as (read_stream, write_stream, *_):  # noqa: SIM117
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_response = await session.list_tools()

                tools = {}
                for tool in tools_response.tools:
                    input_schema = tool.inputSchema or {"type": "object", "properties": {}}
                    logger.info(
                        f"Discovered MCP tool: {tool.name}, "
                        f"schema keys: {list(input_schema.get('properties', {}).keys())}"
                    )
                    tools[tool.name] = {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": input_schema,
                    }
                return tools

    def discover_tools(self) -> dict[str, dict[str, Any]]:
        """Discover tools from the MCP server (synchronous wrapper).

        Returns:
            Dict mapping tool names to their definitions (name, description, input_schema).
        """
        if self._discovered_tools is None:
            # Always use a thread for discovery to avoid event loop conflicts
            # (e.g., Ray actors may have async infrastructure that interferes with asyncio.run())
            logger.info(f"Discovering tools from MCP server: {self.server_url or self.command}")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self._discover_tools_async())
                    self._discovered_tools = future.result(timeout=self.timeout)
                logger.info(f"Discovered {len(self._discovered_tools)} tools: {list(self._discovered_tools.keys())}")
            except concurrent.futures.TimeoutError as err:
                raise TimeoutError(
                    f"MCP tool discovery timed out after {self.timeout}s. Server: {self.server_url or self.command}"
                ) from err
            except Exception as e:
                raise RuntimeError(f"MCP tool discovery failed: {e}") from e

        return self._discovered_tools

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server asynchronously."""
        async with self._get_client_context() as (read_stream, write_stream, *_):  # noqa: SIM117
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

                # Extract text content from result
                if hasattr(result, "content"):
                    text_parts = []
                    for content in result.content:
                        if hasattr(content, "text"):
                            text_parts.append(content.text)
                        elif hasattr(content, "data"):
                            text_parts.append(str(content.data))
                        else:
                            text_parts.append(str(content))
                    return "\n".join(text_parts)
                return str(result)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolOutput:
        """Call an MCP tool with retry logic.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            ToolOutput with the result.
        """
        start_time = time.time()

        # Check if tool exists
        discovered = self.discover_tools()
        if tool_name not in discovered:
            result = ToolOutput(
                output="",
                error=f"Unknown MCP tool: {tool_name}. Available: {list(discovered.keys())}",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )
            _log_tool_call(tool_name, str(arguments), result)
            return result

        last_error: str | None = None
        retryable_exceptions = (ConnectionError, TimeoutError, OSError)

        for attempt in range(self.max_retries):
            try:
                output = await asyncio.wait_for(self._call_tool_async(tool_name, arguments), timeout=self.timeout)
                result = ToolOutput(
                    output=output, called=True, error="", timeout=False, runtime=time.time() - start_time
                )
                _log_tool_call(tool_name, str(arguments), result)
                return result

            except asyncio.TimeoutError:
                result = ToolOutput(
                    output="",
                    error=f"Timeout after {self.timeout} seconds",
                    called=True,
                    timeout=True,
                    runtime=time.time() - start_time,
                )
                _log_tool_call(tool_name, str(arguments), result)
                return result

            except retryable_exceptions as e:
                last_error = str(e)
                if attempt + 1 < self.max_retries:
                    sleep_time = self.retry_backoff * (2**attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{self.max_retries} for MCP tool {tool_name}: {e}. "
                        f"Sleeping {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time)
                    continue

            except Exception as e:
                # Non-retryable error
                result = ToolOutput(
                    output="", error=str(e), called=True, timeout=False, runtime=time.time() - start_time
                )
                _log_tool_call(tool_name, str(arguments), result)
                return result

        # All retries exhausted
        result = ToolOutput(
            output="",
            error=last_error or "Unknown error after retries",
            called=True,
            timeout=False,
            runtime=time.time() - start_time,
        )
        _log_tool_call(tool_name, str(arguments), result)
        return result


class GenericMCPTool(Tool):
    """
    A generic MCP (Model Context Protocol) tool that connects to any MCP server.

    This tool connects to an MCP server and exposes a single tool from it.
    Use the `tool_name` parameter to specify which tool to expose. If not
    specified, the first discovered tool is used.

    For exposing multiple tools from the same MCP server, create multiple
    GenericMCPTool instances with different `tool_name` values, or use
    the GenericMCPToolConfig which can auto-expand to discover all tools.

    Example usage:
        # Connect via HTTP and expose the "search" tool
        tool = GenericMCPTool(
            call_name="search",
            server_url="http://localhost:8000/mcp",
            tool_name="search"
        )

        # Call the tool
        result = await tool.execute(query="test")

        # Connect via stdio
        tool = GenericMCPTool(
            call_name="read_file",
            transport="stdio",
            command="python",
            args=["my_mcp_server.py"],
            tool_name="read_file"
        )
    """

    config_name = "generic_mcp"

    def __init__(
        self,
        call_name: str,
        server_url: str | None = None,
        transport: str = "http",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        tool_name: str | None = None,
    ) -> None:
        """Initialize a GenericMCPTool.

        Args:
            call_name: The name used to call this tool (e.g., in XML tags).
            server_url: URL for HTTP/SSE transport.
            transport: Transport type (http, sse, stdio).
            command: Command for stdio transport.
            args: Arguments for stdio command.
            env: Environment variables for stdio transport.
            timeout: Timeout in seconds.
            max_retries: Maximum retries for transient errors.
            retry_backoff: Backoff factor for retries.
            tool_name: Name of the MCP tool to expose. If None, uses first discovered.
        """
        self.call_name = call_name
        self._factory = MCPToolFactory(
            server_url=server_url,
            transport=transport,
            command=command,
            args=args,
            env=env,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )

        # Store config for lazy discovery
        self._requested_tool_name = tool_name

        # Lazy-initialized after discovery
        self._tool_name: str | None = None
        self._tool_info: dict[str, Any] | None = None

    def _ensure_discovered(self) -> None:
        """Ensure tool discovery has happened. Called lazily on first access."""
        if self._tool_name is not None:
            return

        discovered = self._factory.discover_tools()
        if not discovered:
            raise ValueError("No tools discovered from MCP server")

        if self._requested_tool_name:
            if self._requested_tool_name not in discovered:
                raise ValueError(f"Tool '{self._requested_tool_name}' not found. Available: {list(discovered.keys())}")
            self._tool_name = self._requested_tool_name
        else:
            # Use first discovered tool
            self._tool_name = next(iter(discovered.keys()))
            logger.info(f"No tool_name specified, using first discovered: {self._tool_name}")

        self._tool_info = discovered[self._tool_name]

    @property
    def description(self) -> str:
        """Return the tool's description from MCP."""
        self._ensure_discovered()
        return self._tool_info.get("description", "")  # type: ignore[union-attr]

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the tool's parameter schema from MCP."""
        self._ensure_discovered()
        return self._tool_info.get("input_schema", {"type": "object", "properties": {}})  # type: ignore[union-attr]

    async def execute(self, **kwargs: Any) -> ToolOutput:
        """Call the MCP tool."""
        self._ensure_discovered()
        return await self._factory.call_tool(self._tool_name, kwargs)  # type: ignore[arg-type]


@dataclass
class GenericMCPToolConfig(BaseToolConfig):
    """Configuration for generic MCP tools.

    Connects to an MCP server and exposes its tools. Use `tool_name` to specify
    which tool to expose. If not specified, the first discovered tool is used.

    For auto-discovery of ALL tools, use the `expand_tools` method which returns
    a list of configs, one per discovered tool.

    Example CLI usage:
        # Expose a specific tool
        --tools generic_mcp --tool_configs '{"server_url": "http://localhost:8000/mcp", "tool_name": "search"}'
        --tool_call_names search

        # Stdio transport
        --tools generic_mcp --tool_configs '{"transport": "stdio", "command": "python", "args": ["server.py"]}'
    """

    tool_class: ClassVar[type[Tool]] = GenericMCPTool

    server_url: str | None = None
    """URL for HTTP/SSE transport (e.g., 'http://localhost:8000/mcp')."""
    transport: str = "http"
    """Transport type: 'http', 'sse', or 'stdio'."""
    command: str | None = None
    """Command to run for stdio transport."""
    args: list[str] = field(default_factory=list)
    """Arguments for the stdio command."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables for stdio transport."""
    timeout: int = 60
    """Timeout in seconds for tool calls."""
    max_retries: int = 3
    """Maximum number of retries for transient errors."""
    retry_backoff: float = 0.5
    """Backoff factor for retries."""
    tool_name: str | None = None
    """Name of a specific MCP tool to expose. If None, first discovered tool is used."""

    def discover_tool_names(self) -> list[str]:
        """Discover all tool names from the MCP server.

        Returns:
            List of tool names available on the MCP server.
        """
        factory = MCPToolFactory(
            server_url=self.server_url,
            transport=self.transport,
            command=self.command,
            args=self.args,
            env=self.env,
            timeout=self.timeout,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
        )
        discovered = factory.discover_tools()
        return list(discovered.keys())

    def expand_tools(self) -> list["GenericMCPToolConfig"]:
        """Expand this config to create one config per discovered tool.

        If `tool_name` is already specified, returns [self] unchanged.
        Otherwise, discovers all tools and returns a config for each.

        Returns:
            List of GenericMCPToolConfig instances, one per tool.
        """
        if self.tool_name is not None:
            return [self]

        tool_names = self.discover_tool_names()
        logger.info(f"MCP server discovered {len(tool_names)} tools: {tool_names}")

        return [replace(self, tool_name=name) for name in tool_names]
