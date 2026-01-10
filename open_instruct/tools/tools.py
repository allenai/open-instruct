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
from collections.abc import Collection
from dataclasses import asdict, dataclass
from typing import Annotated, Any, ClassVar

import requests
from pydantic import Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_instruct.tools.utils import BaseToolConfig, Tool, ToolOutput

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


def _create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Collection[int] = (500, 502, 504),
    allowed_methods: Collection[str] = ("GET", "POST"),
) -> requests.Session:
    """Create a requests session with automatic retries."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# =============================================================================
# MaxCallsExceededTool (no config needed)
# =============================================================================


class MaxCallsExceededTool(Tool):
    """Tool that returns a message when max tool calls have been exceeded."""

    _default_tool_function_name = "max_calls_exceeded"
    _default_tool_description = "Returns an error when max tool calls limit is hit"
    # No parameters needed - explicit empty schema
    _default_tool_parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    def __call__(self) -> ToolOutput:
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

    def __call__(self, code: Annotated[str, Field(description="Python code to execute")]) -> ToolOutput:
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

        all_outputs = []
        timed_out = False
        error = ""
        start_time = time.time()

        try:
            response = requests.post(
                self.api_endpoint, json={"code": code, "timeout": self.timeout}, timeout=self.timeout
            )
            result = response.json()
            output = result["output"]
            error = result.get("error") or ""

            all_outputs.append(output)
            if len(error) > 0:
                all_outputs.append("\n" + error)

        except requests.Timeout:
            all_outputs.append(f"Timeout after {self.timeout} seconds")
            timed_out = True

        except Exception as e:
            error_message = f"Error calling API: {str(e)}\n"
            error_traceback = traceback.format_exc()
            all_outputs.append(error_message + error_traceback)

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

    def __call__(self, query: Annotated[str, Field(description="The search query")]) -> ToolOutput:
        """Search for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query or "", result)
            return result

        start_time = time.time()

        # Resolve API endpoint
        url = self.api_endpoint
        if not url:
            url = os.environ.get("MASSIVE_DS_URL")
            if not url:
                result = ToolOutput(
                    output="",
                    error="Missing MASSIVE_DS_URL environment variable.",
                    called=True,
                    timeout=False,
                    runtime=time.time() - start_time,
                )
                _log_tool_call(self.tool_function_name, query, result)
                return result

        session = _create_session_with_retries()

        try:
            res = session.post(
                url,
                json={"query": query, "n_docs": self.num_results, "domains": "dpr_wiki_contriever"},
                headers={"Content-Type": "application/json"},
                timeout=(3, 15),
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
            _log_tool_call(self.tool_function_name, query, result)
            return result

        except requests.exceptions.RequestException as e:
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

    def __call__(
        self, query: Annotated[str, Field(description="The search query for Semantic Scholar")]
    ) -> ToolOutput:
        """Search Semantic Scholar for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query or "", result)
            return result

        start_time = time.time()

        api_key = os.environ.get("S2_API_KEY")
        if not api_key:
            result = ToolOutput(
                output="",
                error="Missing S2_API_KEY environment variable.",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )
            _log_tool_call(self.tool_function_name, query, result)
            return result

        session = _create_session_with_retries()

        try:
            res = session.get(
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
                _log_tool_call(self.tool_function_name, query, result)
                return result

            all_snippets = "\n".join(snippets).strip()
            result = ToolOutput(
                output=all_snippets, called=True, error="", timeout=False, runtime=time.time() - start_time
            )
            _log_tool_call(self.tool_function_name, query, result)
            return result

        except requests.exceptions.RequestException as e:
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

    def __call__(self, query: Annotated[str, Field(description="The search query for Google")]) -> ToolOutput:
        """Search Google via Serper for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.tool_function_name, query or "", result)
            return result

        start_time = time.time()

        api_key = os.environ.get("SERPER_API_KEY")
        if not api_key:
            result = ToolOutput(
                output="",
                error="Missing SERPER_API_KEY environment variable.",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )
            _log_tool_call(self.tool_function_name, query, result)
            return result

        session = _create_session_with_retries()

        try:
            response = session.post(
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
                _log_tool_call(self.tool_function_name, query, result)
                return result

            all_snippets = "\n\n".join(snippets).strip()
            result = ToolOutput(
                output=all_snippets, called=True, error="", timeout=False, runtime=time.time() - start_time
            )
            _log_tool_call(self.tool_function_name, query, result)
            return result

        except requests.exceptions.RequestException as e:
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
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

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

    def __call__(
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
                    # Retry on transient errors
                    for attempt in range(self.max_retries):
                        try:
                            document_tool_output = asyncio.run(mcp_tool(text))
                            break
                        except Exception as e:
                            # Check for transient network errors
                            if MCP_AVAILABLE and isinstance(
                                e, (httpcore.RemoteProtocolError, httpx.ReadError, ConnectionError, TimeoutError)
                            ):
                                if attempt + 1 >= self.max_retries:
                                    raise
                                time.sleep(self.retry_backoff * (2**attempt))
                            else:
                                raise

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
# GenericMCPTool + Config
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


class GenericMCPTool(Tool):
    """
    A generic MCP (Model Context Protocol) tool that connects to any MCP server.

    This tool discovers available tools from the connected MCP server and exposes
    them as first-class tools. Each discovered MCP tool appears as its own tool
    to the model, allowing direct calls like `search(query="...")` instead of
    `generic_mcp(tool_name="search", arguments={...})`.

    Use `get_openai_tool_definitions()` to get all tool definitions, and
    `get_tool_names()` to get the list of discovered tool names.

    Requires the mcp package: uv sync --extra mcp

    Example usage:
        # Connect via HTTP
        tool = GenericMCPTool(server_url="http://localhost:8000/mcp")

        # Connect via stdio
        tool = GenericMCPTool(
            transport="stdio",
            command="python",
            args=["my_mcp_server.py"]
        )

        # Discover tools
        tool_names = tool.get_tool_names()  # e.g., ["search", "read_file", "write_file"]

        # Call a tool directly (model calls search(query="test"))
        result = tool(_mcp_tool_name="search", query="test")
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
        override_name: str | None = None,
    ) -> None:
        """Initialize a GenericMCPTool.

        Args:
            server_url: URL for HTTP transport (e.g., "http://localhost:8000/mcp").
            transport: Transport type, either "http" or "stdio".
            command: Command to run for stdio transport.
            args: Arguments for the stdio command.
            env: Environment variables for stdio transport.
            timeout: Timeout in seconds for tool calls.
            max_retries: Maximum number of retries for transient errors.
            retry_backoff: Backoff factor for retries (uses exponential backoff).
            override_name: Override the default tool function name (not typically used for multi-tools).
        """
        if not GENERIC_MCP_AVAILABLE:
            raise ImportError("Generic MCP tools require the mcp package. Install it with: uv sync --extra mcp")

        self._override_name = override_name
        self.server_url = server_url
        self.transport = transport
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Validate configuration
        if transport == "http" and not server_url:
            raise ValueError("server_url is required for HTTP transport")
        if transport == "stdio" and not command:
            raise ValueError("command is required for stdio transport")

        # Cache for discovered tools (populated lazily)
        self._discovered_tools: dict[str, dict[str, Any]] | None = None

    async def _get_client_context(self):
        """Get the appropriate client context manager based on transport type."""
        if self.transport == "http":
            return streamablehttp_client(self.server_url)
        elif self.transport == "sse":
            return sse_client(self.server_url)
        elif self.transport == "stdio":
            server_params = StdioServerParameters(
                command=self.command, args=self.args, env=self.env if self.env else None
            )
            return stdio_client(server_params)
        else:
            raise ValueError(f"Unknown transport type: {self.transport}. Supported: http, sse, stdio")

    async def _discover_tools(self) -> dict[str, dict[str, Any]]:
        """Discover available tools from the MCP server.

        Returns:
            Dict mapping tool names to their definitions.
        """
        async with await self._get_client_context() as (read_stream, write_stream, *_):  # noqa: SIM117
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_response = await session.list_tools()

                tools = {}
                for tool in tools_response.tools:
                    input_schema = tool.inputSchema or {"type": "object", "properties": {}}
                    logger.info(f"Discovered MCP tool: {tool.name}, schema: {input_schema}")
                    tools[tool.name] = {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": input_schema,
                    }
                return tools

    async def _call_tool_async(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server asynchronously.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's response as a string.
        """
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

    def get_discovered_tools(self) -> dict[str, dict[str, Any]]:
        """Get the list of discovered tools from the MCP server.

        This will connect to the server and discover tools if not already cached.

        Returns:
            Dict mapping tool names to their definitions.
        """
        if self._discovered_tools is None:
            self._discovered_tools = asyncio.run(self._discover_tools())
        return self._discovered_tools

    def get_tool_names(self) -> list[str]:
        """Get the names of all tools exposed by this multi-tool.

        Returns:
            List of tool names that this tool handles.
        """
        return list(self.get_discovered_tools().keys())

    def handles_tool(self, tool_name: str) -> bool:
        """Check if this tool handles the given tool name.

        Args:
            tool_name: The name of the tool to check.

        Returns:
            True if this tool handles the given tool name.
        """
        return tool_name in self.get_discovered_tools()

    def get_openai_tool_definitions(self) -> list[dict[str, Any]]:
        """Get OpenAI-format tool definitions for all discovered MCP tools.

        For multi-tools, this returns one definition per discovered tool.
        This is used instead of get_openai_tool_definition() for multi-tools.

        Returns:
            List of tool definitions in OpenAI function calling format.
        """
        discovered = self.get_discovered_tools()
        definitions = []
        for tool_name, tool_info in discovered.items():
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_info.get("description", ""),
                        "parameters": tool_info.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        return definitions

    def __call__(self, _mcp_tool_name: str | None = None, **kwargs: Any) -> ToolOutput:
        """Call an MCP tool on the connected server.

        This method is called with the tool name passed via _mcp_tool_name,
        and the actual tool arguments passed as **kwargs.

        Args:
            _mcp_tool_name: The name of the MCP tool to call (set by the dispatcher).
            **kwargs: Arguments to pass to the MCP tool.

        Returns:
            ToolOutput with the result.
        """
        start_time = time.time()

        if not _mcp_tool_name:
            result = ToolOutput(
                output="",
                error="No tool name provided. Use _mcp_tool_name to specify which MCP tool to call.",
                called=True,
                timeout=False,
                runtime=0,
            )
            _log_tool_call(self.tool_function_name, str(kwargs), result)
            return result

        # Check if tool exists
        if not self.handles_tool(_mcp_tool_name):
            result = ToolOutput(
                output="",
                error=f"Unknown MCP tool: {_mcp_tool_name}. Available: {self.get_tool_names()}",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )
            _log_tool_call(_mcp_tool_name, str(kwargs), result)
            return result

        last_error = None
        for attempt in range(self.max_retries):
            try:
                output = asyncio.run(
                    asyncio.wait_for(self._call_tool_async(_mcp_tool_name, kwargs), timeout=self.timeout)
                )
                result = ToolOutput(
                    output=output, called=True, error="", timeout=False, runtime=time.time() - start_time
                )
                _log_tool_call(_mcp_tool_name, str(kwargs), result)
                return result

            except asyncio.TimeoutError:
                result = ToolOutput(
                    output="",
                    error=f"Timeout after {self.timeout} seconds",
                    called=True,
                    timeout=True,
                    runtime=time.time() - start_time,
                )
                _log_tool_call(_mcp_tool_name, str(kwargs), result)
                return result

            except Exception as e:
                last_error = str(e)
                # Retry on transient errors
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff * (2**attempt))
                    continue

        # All retries exhausted
        result = ToolOutput(
            output="",
            error=last_error or "Unknown error",
            called=True,
            timeout=False,
            runtime=time.time() - start_time,
        )
        _log_tool_call(_mcp_tool_name, str(kwargs), result)
        return result


@dataclass
class GenericMCPToolConfig(BaseToolConfig):
    """Configuration for the generic MCP tool.

    This tool connects to any MCP server and exposes its tools. It supports
    both HTTP and stdio transports.

    Example CLI usage:
        # HTTP transport
        --tools generic_mcp --tool_configs '{"server_url": "http://localhost:8000/mcp"}'

        # Stdio transport
        --tools generic_mcp --tool_configs '{"transport": "stdio", "command": "python", "args": ["server.py"]}'
    """

    tool_class: ClassVar[type[Tool]] = GenericMCPTool

    server_url: str | None = None
    """URL for HTTP transport (e.g., 'http://localhost:8000/mcp')."""
    transport: str = "http"
    """Transport type: 'http' or 'stdio'."""
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
