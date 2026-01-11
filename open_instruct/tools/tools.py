"""
Tools that follow the Tool(ABC) pattern.

Each tool has a corresponding Config dataclass that inherits from ToolConfig.
The config's build() method creates the tool instance. Most configs use the
generic build() from ToolConfig; override only when custom logic is needed.
"""

import asyncio
import inspect
import logging
import os
import time
import traceback
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Annotated, Any, ClassVar

import httpx
from pydantic import Field

from open_instruct.tools.utils import BaseToolConfig, RetryConfig, Tool, ToolOutput

logger = logging.getLogger(__name__)


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


# Optional imports for MCP tools
try:
    import httpcore
    import httpx
    from dr_agent.tool_interface.mcp_tools import (
        Crawl4AIBrowseTool,
        MassiveServeSearchTool,
        SemanticScholarSnippetSearchTool,
        SerperSearchTool,
    )

    MCP_AVAILABLE = True
    MCP_TOOL_REGISTRY: dict[str, type] = {
        "snippet_search": SemanticScholarSnippetSearchTool,
        "google_search": SerperSearchTool,
        "massive_serve": MassiveServeSearchTool,
        "browse_webpage": Crawl4AIBrowseTool,
    }
except ImportError:
    MCP_AVAILABLE = False
    MCP_TOOL_REGISTRY = {}


# =============================================================================
# MaxCallsExceededTool (no config needed)
# =============================================================================


class MaxCallsExceededTool(Tool):
    """Tool that returns a message when max tool calls have been exceeded."""

    _default_tool_function_name = "max_calls_exceeded"
    _default_tool_description = "Returns an error when max tool calls limit is hit"
    # No parameters needed - explicit empty schema
    _default_tool_parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    async def __call__(self, **kwargs: Any) -> ToolOutput:
        """Return an error message indicating max tool calls exceeded."""
        return ToolOutput(output="Max tool calls exceeded.", called=False, error="", timeout=False, runtime=0)


# =============================================================================
# PythonCodeTool + Config
# =============================================================================


class PythonCodeTool(Tool):
    """
    Executes Python code via a FastAPI endpoint.

    @vwxyzjn: I recommend using something like a FastAPI for this kind of stuff; 1) you
    won't accidentally block the main vLLM process and 2) way easier to parallelize via load balancing.
    """

    _default_tool_function_name = "python"
    _default_tool_description = "Executes Python code and returns printed output."
    # Parameters inferred from __call__ signature

    def __init__(self, api_endpoint: str, timeout: int = 3, override_name: str | None = None) -> None:
        self.api_endpoint = api_endpoint
        self.timeout = timeout
        self._override_name = override_name

    async def __call__(self, code: Annotated[str, Field(description="Python code to execute")]) -> ToolOutput:
        """Execute Python code via the API."""
        if not code or not code.strip():
            result = ToolOutput(
                output="",
                error="Empty code. Please provide some code to execute.",
                called=True,
                timeout=False,
                runtime=0,
            )
            _log_tool_call(self.tool_function_name, code or "", result)
            return result

        start_time = time.time()
        all_outputs = []
        timed_out = False
        error = ""

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_endpoint, json={"code": code, "timeout": self.timeout}, timeout=self.timeout
                )
                res = response.json()
                output = res["output"]
                error = res.get("error") or ""

                all_outputs.append(output)
                if error:
                    all_outputs.append("\n" + error)

        except httpx.TimeoutException:
            all_outputs.append(f"Timeout after {self.timeout} seconds")
            timed_out = True

        except Exception as e:
            all_outputs.append(f"Error calling API: {e}\n{traceback.format_exc()}")

        result = ToolOutput(
            output="\n".join(all_outputs),
            called=True,
            error=error,
            timeout=timed_out,
            runtime=time.time() - start_time,
        )
        _log_tool_call(self.tool_function_name, code, result)
        return result


@dataclass
class PythonCodeToolConfig(BaseToolConfig):
    """Configuration for the Python code execution tool."""

    tool_class: ClassVar[type[Tool]] = PythonCodeTool

    api_endpoint: str | None = None
    """The API endpoint for the code execution server."""
    timeout: int = 3
    """Timeout in seconds for code execution."""

    def build(self) -> PythonCodeTool:
        if not self.api_endpoint:
            raise ValueError("api_endpoint must be set to use the Python code tool")
        return PythonCodeTool(**asdict(self))


# =============================================================================
# MassiveDSSearchTool + Config
# =============================================================================


class MassiveDSSearchTool(Tool):
    """Search tool using the massive_ds API."""

    _default_tool_function_name = "massive_ds_search"  # Distinct from SerperSearchTool's "search"
    _default_tool_description = "Searches Wikipedia/documents using the massive_ds retrieval system"
    # Parameters inferred from __call__ signature

    def __init__(
        self, api_endpoint: str | None = None, num_results: int = 3, override_name: str | None = None
    ) -> None:
        self.api_endpoint = api_endpoint
        self.num_results = num_results
        self._override_name = override_name

    async def __call__(self, query: Annotated[str, Field(description="The search query")]) -> ToolOutput:
        """Search for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query or "", result)
            return result

        url = self.api_endpoint or os.environ.get("MASSIVE_DS_URL")
        if not url:
            result = ToolOutput(
                output="", error="Missing MASSIVE_DS_URL environment variable.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query, result)
            return result

        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    url,
                    json={"query": query, "n_docs": self.num_results, "domains": "dpr_wiki_contriever"},
                    headers={"Content-Type": "application/json"},
                    timeout=15,
                )
                res.raise_for_status()
                data = res.json()
                passages = data.get("results", {}).get("passages", [[]])[0]
                passages = passages[: self.num_results]
                passages = ["\n" + passage for passage in passages]
                all_snippets = "\n".join(passages).strip()

                result = ToolOutput(
                    output=all_snippets, called=True, error="", timeout=False, runtime=time.time() - start_time
                )
        except httpx.HTTPError as e:
            result = ToolOutput(output="", error=str(e), called=True, timeout=False, runtime=time.time() - start_time)

        _log_tool_call(self.tool_function_name, query, result)
        return result


@dataclass
class MassiveDSSearchToolConfig(BaseToolConfig):
    """Configuration for the massive_ds search tool."""

    tool_class: ClassVar[type[Tool]] = MassiveDSSearchTool

    api_endpoint: str | None = None
    """The API endpoint for the search engine."""
    num_results: int = 3
    """The maximum number of documents to retrieve for each query."""


# =============================================================================
# S2SearchTool + Config
# =============================================================================


class S2SearchTool(Tool):
    """
    Search tool using the Semantic Scholar API.
    Requires S2_API_KEY environment variable.
    """

    _default_tool_function_name = "s2_search"
    _default_tool_description = "Searches Semantic Scholar for academic papers and citations"
    # Parameters inferred from __call__ signature

    def __init__(self, num_results: int = 10, override_name: str | None = None) -> None:
        self.num_results = num_results
        self._override_name = override_name

    async def __call__(
        self, query: Annotated[str, Field(description="The search query for Semantic Scholar")]
    ) -> ToolOutput:
        """Search Semantic Scholar for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query or "", result)
            return result

        api_key = os.environ.get("S2_API_KEY")
        if not api_key:
            result = ToolOutput(
                output="", error="Missing S2_API_KEY environment variable.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query, result)
            return result

        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    "https://api.semanticscholar.org/graph/v1/snippet/search",
                    params={"limit": self.num_results, "query": query},
                    headers={"x-api-key": api_key},
                    timeout=60,
                )
                res.raise_for_status()
                data = res.json().get("data", [])
                snippets = [item["snippet"]["text"] for item in data if item.get("snippet")]

                if not snippets:
                    result = ToolOutput(
                        output="",
                        error="Query returned no results.",
                        called=True,
                        timeout=False,
                        runtime=time.time() - start_time,
                    )
                else:
                    result = ToolOutput(
                        output="\n".join(snippets).strip(),
                        called=True,
                        error="",
                        timeout=False,
                        runtime=time.time() - start_time,
                    )
        except httpx.HTTPError as e:
            result = ToolOutput(output="", error=str(e), called=True, timeout=False, runtime=time.time() - start_time)

        _log_tool_call(self.tool_function_name, query, result)
        return result


@dataclass
class S2SearchToolConfig(BaseToolConfig):
    """Configuration for the Semantic Scholar search tool."""

    tool_class: ClassVar[type[Tool]] = S2SearchTool

    num_results: int = 10
    """Number of results to return from Semantic Scholar."""


# =============================================================================
# SerperSearchTool + Config
# =============================================================================


class SerperSearchTool(Tool):
    """
    Search tool using the Serper API (Google Search results).
    Requires SERPER_API_KEY environment variable.

    Serper provides fast Google Search results via API. Sign up at https://serper.dev
    """

    _default_tool_function_name = "serper_search"  # Use "search" to match model training format
    _default_tool_description = "Google search via the Serper API"
    # Parameters inferred from __call__ signature

    def __init__(self, num_results: int = 5, override_name: str | None = None) -> None:
        self.num_results = num_results
        self._override_name = override_name

    async def __call__(self, query: Annotated[str, Field(description="The search query for Google")]) -> ToolOutput:
        """Search Google via Serper for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query or "", result)
            return result

        api_key = os.environ.get("SERPER_API_KEY")
        if not api_key:
            result = ToolOutput(
                output="", error="Missing SERPER_API_KEY environment variable.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query, result)
            return result

        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": self.num_results},
                    headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                    timeout=10,
                )
                response.raise_for_status()
                data = response.json()

                snippets = []

                # Extract snippets from organic results
                for item in data.get("organic", [])[: self.num_results]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    if snippet:
                        snippets.append(f"**{title}**\n{snippet}\nSource: {link}")

                # Also include answer box if present
                if "answerBox" in data:
                    answer_box = data["answerBox"]
                    if "answer" in answer_box:
                        snippets.insert(0, f"**Direct Answer:** {answer_box['answer']}")
                    elif "snippet" in answer_box:
                        snippets.insert(0, f"**Featured Snippet:** {answer_box['snippet']}")

                if not snippets:
                    result = ToolOutput(
                        output="",
                        error="Query returned no results.",
                        called=True,
                        timeout=False,
                        runtime=time.time() - start_time,
                    )
                else:
                    result = ToolOutput(
                        output="\n\n".join(snippets).strip(),
                        called=True,
                        error="",
                        timeout=False,
                        runtime=time.time() - start_time,
                    )
        except httpx.HTTPError as e:
            result = ToolOutput(output="", error=str(e), called=True, timeout=False, runtime=time.time() - start_time)

        _log_tool_call(self.tool_function_name, query, result)
        return result


@dataclass
class SerperSearchToolConfig(BaseToolConfig):
    """Configuration for the Serper (Google Search) tool."""

    tool_class: ClassVar[type[Tool]] = SerperSearchTool

    num_results: int = 5
    """Number of results to return from Serper."""


# =============================================================================
# DrAgentMCPTool + Config
# =============================================================================


class DrAgentMCPTool(Tool):
    """
    A wrapper for MCP (Model Context Protocol) tools from dr_agent.

    Unlike other tools, this handles multiple MCP tools that share the same
    end string (</tool>). It routes calls to the appropriate underlying tool.

    Requires the dr_agent package to be installed.
    """

    _default_tool_function_name = "mcp"
    _default_tool_description = (
        "MCP tools wrapper supporting snippet_search, google_search, massive_serve, browse_webpage"
    )
    # Parameters inferred from __call__ signature

    def __init__(
        self,
        tool_names: list[str] | str,
        parser_name: str = "unified",
        transport_type: str | None = None,
        host: str | None = None,
        port: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        timeout: int = 180,
        base_url: str | None = None,
        num_results: int = 10,
        use_localized_snippets: bool = False,
        context_chars: int = 6000,
        override_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not MCP_AVAILABLE:
            raise ImportError("MCP tools require dr_agent package. Install it with: uv sync --extra dr-tulu")

        self._override_name = override_name
        self.mcp_tools: list[Any] = []
        self.stop_strings: list[str] = []

        # Configure retry behavior
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            backoff_factor=retry_backoff,
            retryable_exceptions=(httpcore.RemoteProtocolError, httpx.ReadError, ConnectionError, TimeoutError)
            if MCP_AVAILABLE
            else (ConnectionError, TimeoutError),
        )

        # Configure transport
        self.transport_type = transport_type or os.environ.get("MCP_TRANSPORT", "StreamableHttpTransport")
        self.host = host or os.environ.get("MCP_TRANSPORT_HOST", "0.0.0.0")
        self.port = port or os.environ.get("MCP_TRANSPORT_PORT", "8000")

        if self.host is not None:
            os.environ["MCP_TRANSPORT_HOST"] = str(self.host)
        if self.port is not None:
            os.environ["MCP_TRANSPORT_PORT"] = str(self.port)

        # Support comma-separated string for tool_names
        if isinstance(tool_names, str):
            tool_names = [n.strip() for n in tool_names.split(",") if n.strip()]

        for mcp_tool_name in tool_names:
            if mcp_tool_name not in MCP_TOOL_REGISTRY:
                raise ValueError(f"Unknown MCP tool: {mcp_tool_name}. Available: {list(MCP_TOOL_REGISTRY.keys())}")

            mcp_tool_cls = MCP_TOOL_REGISTRY[mcp_tool_name]
            sig = inspect.signature(mcp_tool_cls.__init__)
            valid_params = set(sig.parameters.keys())

            # Filter kwargs to only valid params
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            if "base_url" in valid_params:
                filtered_kwargs["base_url"] = base_url
            if "number_documents_to_search" in valid_params:
                filtered_kwargs["number_documents_to_search"] = num_results
            if "use_localized_snippets" in valid_params:
                filtered_kwargs["use_localized_snippets"] = use_localized_snippets
            if "context_chars" in valid_params:
                filtered_kwargs["context_chars"] = context_chars

            # Special config for crawl4ai
            if mcp_tool_name == "browse_webpage":
                filtered_kwargs["use_docker_version"] = True
                filtered_kwargs["use_ai2_config"] = True

            self.mcp_tools.append(
                mcp_tool_cls(
                    timeout=timeout,
                    name=mcp_tool_name,
                    tool_parser=parser_name,
                    transport_type=self.transport_type,
                    **filtered_kwargs,
                )
            )
            self.stop_strings += self.mcp_tools[-1].tool_parser.stop_sequences

    def get_stop_strings(self) -> list[str]:
        """Return the stop strings for all MCP tools."""
        return self.stop_strings

    async def _call_mcp_tool_async(self, mcp_tool: Any, text: str) -> Any:
        """Call an MCP tool with retry logic (async)."""
        last_error: Exception | None = None
        for attempt in range(self._retry_config.max_retries):
            try:
                return await mcp_tool(text)
            except self._retry_config.retryable_exceptions as e:
                last_error = e
                if attempt + 1 < self._retry_config.max_retries:
                    sleep_time = self._retry_config.backoff_factor * (2**attempt)
                    await asyncio.sleep(sleep_time)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry logic error")

    async def __call__(
        self, text: Annotated[str, Field(description="The full prompt text containing MCP tool call tags")]
    ) -> ToolOutput:
        """Execute the appropriate MCP tool based on the text content.

        Note: Unlike other tools, DrAgentMCPTool still parses the text to determine
        which underlying tool to call, as MCP tools share common tags.
        """
        if not MCP_AVAILABLE:
            raise ImportError("MCP tools require dr_agent package. Install it with: uv sync --extra dr-tulu")
        start_time = time.time()

        document_tool_output = None
        error = None
        found_tool = False
        text_output = ""

        try:
            for mcp_tool in self.mcp_tools:
                if mcp_tool.tool_parser.has_calls(text, mcp_tool.name):
                    document_tool_output = await self._call_mcp_tool_async(mcp_tool, text)
                    text_output = mcp_tool._format_output(document_tool_output)
                    text_output = mcp_tool.tool_parser.format_result(text_output, document_tool_output)
                    found_tool = True
                    break

        except Exception as e:
            error = str(e)

        if document_tool_output is None:
            if error is None and not found_tool:
                error = "No valid tool calls found."
            elif error is None:
                error = "Unknown error, no MCP response and no error found."

            result = ToolOutput(
                output=error or "", called=False, error=error or "", timeout=False, runtime=time.time() - start_time
            )
            _log_tool_call(self.tool_function_name, text, result)
            return result

        result = ToolOutput(
            output=text_output,
            called=True,
            error=document_tool_output.error or "",
            timeout=document_tool_output.timeout,
            runtime=document_tool_output.runtime,
        )
        _log_tool_call(self.tool_function_name, text, result)
        return result


@dataclass
class DrAgentMCPToolConfig(BaseToolConfig):
    """Configuration for MCP (Model Context Protocol) tools."""

    tool_class: ClassVar[type[Tool]] = DrAgentMCPTool

    tool_names: str = "snippet_search"
    """Comma-separated list of MCP tool names to use."""
    parser_name: str = "unified"
    """The parser name for MCP tools."""
    transport_type: str | None = None
    """Transport type for MCP (default: StreamableHttpTransport)."""
    host: str | None = None
    """Host for MCP transport."""
    port: int | None = None
    """Port for MCP transport."""
    timeout: int = 180
    """Timeout in seconds for MCP tool calls."""
    max_retries: int = 3
    """Maximum retries for transient MCP errors."""
    retry_backoff: float = 0.5
    """Backoff factor for MCP retries."""
    base_url: str | None = None
    """Base URL for MCP tools."""
    num_results: int = 10
    """Number of documents/results to retrieve."""
    use_localized_snippets: bool = False
    """Whether to use localized snippets."""
    context_chars: int = 6000
    """Number of context characters for MCP tools."""


# =============================================================================
# GenericMCPTool + MCPToolFactory + Config
# =============================================================================

# Optional imports for generic MCP tools (official MCP SDK)
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamablehttp_client

    GENERIC_MCP_AVAILABLE = True
except ImportError:
    GENERIC_MCP_AVAILABLE = False


class MCPTransport(str, Enum):
    """Transport types for MCP connections."""

    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"


class MCPToolWrapper(Tool):
    """
    A thin wrapper around a single MCP tool discovered from a server.

    This is a lightweight tool that delegates calls to an MCPToolFactory.
    Each discovered MCP tool gets its own wrapper instance.
    """

    def __init__(self, name: str, description: str, input_schema: dict[str, Any], factory: "MCPToolFactory") -> None:
        """Initialize an MCPToolWrapper.

        Args:
            name: The tool's name (used as tool_function_name).
            description: The tool's description.
            input_schema: JSON Schema for the tool's parameters.
            factory: The parent factory that handles the actual calls.
        """
        self._name = name
        self._description = description
        self._input_schema = input_schema
        self._factory = factory

    @property
    def tool_function_name(self) -> str:
        return self._name

    @property
    def tool_description(self) -> str:
        return self._description

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return self._input_schema

    async def __call__(self, **kwargs: Any) -> ToolOutput:
        """Call the MCP tool via the factory."""
        return await self._factory.call_tool(self._name, kwargs)


class MCPToolFactory:
    """
    Factory that discovers tools from an MCP server and creates individual wrappers.

    This replaces the old GenericMCPTool pattern. Instead of a single tool that
    handles multiple names via _mcp_tool_name, this factory creates separate
    MCPToolWrapper instances for each discovered tool.

    Example usage:
        # Create factory
        factory = MCPToolFactory(server_url="http://localhost:8000/mcp")

        # Discover and create tool wrappers
        tools = factory.create_tools()  # dict[str, MCPToolWrapper]

        # Each tool is independent
        result = tools["search"](query="test")

    Requires the mcp package: uv sync --extra mcp
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
        override_name: str | None = None,
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
            override_name: Not used by factory, included for config compatibility.
        """
        if not GENERIC_MCP_AVAILABLE:
            raise ImportError("MCP tools require the mcp package. Install it with: uv sync --extra mcp")

        self.server_url = server_url
        self.transport = MCPTransport(transport) if isinstance(transport, str) else transport
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self._override_name = override_name  # Not used, for config compatibility

        # Configure retry behavior
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            backoff_factor=retry_backoff,
            retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        )

        # Validate configuration
        if self.transport in (MCPTransport.HTTP, MCPTransport.SSE) and not server_url:
            raise ValueError(f"server_url is required for {self.transport.value} transport")
        if self.transport == MCPTransport.STDIO and not command:
            raise ValueError("command is required for stdio transport")

        # Cache for discovered tools
        self._discovered_tools: dict[str, dict[str, Any]] | None = None
        self._tool_wrappers: dict[str, MCPToolWrapper] | None = None

    async def _get_client_context(self):
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
        async with await self._get_client_context() as (read_stream, write_stream, *_):  # noqa: SIM117
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
        """Discover tools from the MCP server.

        Returns:
            Dict mapping tool names to their definitions.
        """
        if self._discovered_tools is None:
            # Always use a thread for discovery to avoid event loop conflicts
            # (e.g., Ray actors may have async infrastructure that interferes with asyncio.run())
            import concurrent.futures

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

    def create_tools(self) -> dict[str, MCPToolWrapper]:
        """Create MCPToolWrapper instances for each discovered tool.

        Returns:
            Dict mapping tool names to their wrapper instances.
        """
        if self._tool_wrappers is None:
            discovered = self.discover_tools()
            self._tool_wrappers = {}
            for name, info in discovered.items():
                self._tool_wrappers[name] = MCPToolWrapper(
                    name=name,
                    description=info.get("description", ""),
                    input_schema=info.get("input_schema", {"type": "object", "properties": {}}),
                    factory=self,
                )
        return self._tool_wrappers

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server asynchronously."""
        async with await self._get_client_context() as (read_stream, write_stream, *_):  # noqa: SIM117
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

    def _call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Synchronous wrapper for MCP tool call (runs in thread)."""
        logger.debug(f"MCP tool call (in thread): {tool_name} with {arguments}")
        result = asyncio.run(self._call_tool_async(tool_name, arguments))
        logger.debug(f"MCP tool call completed: {tool_name}")
        return result

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolOutput:
        """Call an MCP tool with retry logic (async).

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            ToolOutput with the result.
        """
        import concurrent.futures

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
        for attempt in range(self._retry_config.max_retries):
            try:
                # Run MCP call in a thread to avoid event loop conflicts in Ray
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    output = await asyncio.wait_for(
                        loop.run_in_executor(executor, self._call_tool_sync, tool_name, arguments),
                        timeout=self.timeout,
                    )
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

            except self._retry_config.retryable_exceptions as e:
                last_error = str(e)
                if attempt + 1 < self._retry_config.max_retries:
                    sleep_time = self._retry_config.backoff_factor * (2**attempt)
                    logger.debug(
                        f"Retry {attempt + 1}/{self._retry_config.max_retries} for MCP tool {tool_name}: {e}. "
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
    GenericMCPTool instances with different `tool_name` values.

    Requires the mcp package: uv sync --extra mcp

    Example usage:
        # Connect via HTTP and expose the "search" tool
        tool = GenericMCPTool(
            server_url="http://localhost:8000/mcp",
            tool_name="search"
        )

        # Call the tool
        result = await tool(query="test")

        # Connect via stdio
        tool = GenericMCPTool(
            transport="stdio",
            command="python",
            args=["my_mcp_server.py"],
            tool_name="read_file"
        )
    """

    _default_tool_function_name = "generic_mcp"
    _default_tool_description = "Generic MCP tool that connects to any MCP server"

    def __init__(
        self,
        server_url: str | None = None,
        transport: str = "http",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        tool_name: str | None = None,
        override_name: str | None = None,
    ) -> None:
        """Initialize a GenericMCPTool.

        Args:
            server_url: URL for HTTP/SSE transport.
            transport: Transport type (http, sse, stdio).
            command: Command for stdio transport.
            args: Arguments for stdio command.
            env: Environment variables for stdio transport.
            timeout: Timeout in seconds.
            max_retries: Maximum retries for transient errors.
            retry_backoff: Backoff factor for retries.
            tool_name: Name of the MCP tool to expose. If None, uses first discovered.
            override_name: Override the tool's function name.
        """
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
        self._override_name = override_name

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
    def tool_function_name(self) -> str:
        """Return the tool's function name."""
        if self._override_name:
            return self._override_name
        self._ensure_discovered()
        return self._tool_name  # type: ignore[return-value]

    @property
    def tool_description(self) -> str:
        """Return the tool's description from MCP."""
        self._ensure_discovered()
        return self._tool_info.get("description", "")  # type: ignore[union-attr]

    @property
    def tool_parameters(self) -> dict[str, Any]:
        """Return the tool's parameter schema from MCP."""
        self._ensure_discovered()
        return self._tool_info.get("input_schema", {"type": "object", "properties": {}})  # type: ignore[union-attr]

    async def __call__(self, **kwargs: Any) -> ToolOutput:
        """Call the MCP tool."""
        self._ensure_discovered()
        return await self._factory.call_tool(self._tool_name, kwargs)  # type: ignore[arg-type]


@dataclass
class GenericMCPToolConfig(BaseToolConfig):
    """Configuration for the generic MCP tool.

    This tool connects to any MCP server and exposes a single tool from it.
    Use the `tool_name` field to specify which tool to expose. If not specified,
    the first discovered tool is used.

    For multiple tools from the same server, create multiple config entries
    with different `tool_name` values.

    Example CLI usage:
        # Expose a specific tool
        --tools generic_mcp --tool_configs '{"server_url": "http://localhost:8000/mcp", "tool_name": "search"}'

        # Expose multiple tools from same server
        --tools generic_mcp generic_mcp --tool_configs \\
            '{"server_url": "http://localhost:8000/mcp", "tool_name": "search"}' \\
            '{"server_url": "http://localhost:8000/mcp", "tool_name": "read_file"}'

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
    args: list[str] | None = None
    """Arguments for the stdio command."""
    env: dict[str, str] | None = None
    """Environment variables for stdio transport."""
    timeout: int = 60
    """Timeout in seconds for tool calls."""
    max_retries: int = 3
    """Maximum number of retries for transient errors."""
    retry_backoff: float = 0.5
    """Backoff factor for retries."""
    tool_name: str | None = None
    """Name of the MCP tool to expose. If None, uses first discovered tool."""
