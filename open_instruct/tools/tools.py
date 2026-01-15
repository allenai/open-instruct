"""
Basic tools that are built-in to open-instruct.
"""

import asyncio
import inspect
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, ClassVar

import aiohttp

from open_instruct import logger_utils
from open_instruct.tools.utils import BaseToolConfig, Tool, ToolOutput, make_api_request

logger = logger_utils.setup_logger(__name__)


# Optional imports for DR Agent MCP tools
try:
    from dr_agent.tool_interface.mcp_tools import (
        Crawl4AIBrowseTool,
        MassiveServeSearchTool,
        SemanticScholarSnippetSearchTool,
        SerperSearchTool as DrAgentSerperSearchTool,
    )

    DR_AGENT_MCP_AVAILABLE = True
    DR_AGENT_MCP_TOOLS: dict[str, type] = {
        "snippet_search": SemanticScholarSnippetSearchTool,
        "google_search": DrAgentSerperSearchTool,
        "massive_serve": MassiveServeSearchTool,
        "browse_webpage": Crawl4AIBrowseTool,
    }
except ImportError:
    DR_AGENT_MCP_AVAILABLE = False
    DR_AGENT_MCP_TOOLS: dict[str, type] = {}


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


class PythonCodeTool(Tool):
    """
    Executes Python code via a FastAPI endpoint.
    """

    config_name = "python"
    description = "Executes Python code and returns printed output."
    parameters = {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python code to execute"}},
        "required": ["code"],
    }

    def __init__(self, call_name: str, api_endpoint: str, timeout: int = 3) -> None:
        self.call_name = call_name
        self.api_endpoint = api_endpoint
        self.timeout = timeout

    async def execute(self, code: str) -> ToolOutput:
        """Execute Python code via the API."""
        if not code or not code.strip():
            result = ToolOutput(
                output="",
                error="Empty code. Please provide some code to execute.",
                called=True,
                timeout=False,
                runtime=0,
            )
            _log_tool_call(self.call_name, code or "", result)
            return result

        start_time = time.time()
        api_response = await make_api_request(
            url=self.api_endpoint, timeout_seconds=self.timeout, json_payload={"code": code, "timeout": self.timeout}
        )

        if api_response.error:
            result = ToolOutput(
                output=api_response.error,
                called=True,
                error=api_response.error,
                timeout=api_response.timed_out,
                runtime=time.time() - start_time,
            )
        else:
            output = api_response.data.get("output") or ""
            error = api_response.data.get("error") or ""
            full_output = output + ("\n" + error if error else "")
            result = ToolOutput(
                output=full_output, called=True, error=error, timeout=False, runtime=time.time() - start_time
            )

        _log_tool_call(self.call_name, code, result)
        return result


@dataclass
class PythonCodeToolConfig(BaseToolConfig):
    """Configuration for the Python code execution tool."""

    tool_class: ClassVar[type[Tool]] = PythonCodeTool

    api_endpoint: str
    """The API endpoint for the code execution server."""
    timeout: int = 3
    """Timeout in seconds for code execution."""


class JinaBrowseTool(Tool):
    """
    Tool for fetching webpage content using Jina Reader API.
    Converts webpages to clean, LLM-friendly markdown format.

    Jina Reader is a free API for converting web pages to clean text.
    Get an API key at https://jina.ai/reader/.
    """

    config_name = "jina_browse"
    description = "Fetches and converts webpage content to clean markdown using Jina Reader API"
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The URL of the webpage to fetch"}},
        "required": ["url"],
    }

    def __init__(self, call_name: str, timeout: int = 30) -> None:
        self.call_name = call_name
        self.timeout = timeout
        self.api_key = os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Missing JINA_API_KEY environment variable.")

    async def execute(self, url: str) -> ToolOutput:
        """Fetch webpage content via Jina Reader API."""
        if not url or not url.strip():
            result = ToolOutput(
                output="", error="Empty URL. Please provide a URL to fetch.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.call_name, url or "", result)
            return result

        start_time = time.time()
        api_response = await make_api_request(
            url=f"https://r.jina.ai/{urllib.parse.quote(url.strip(), safe=':/')}",
            timeout_seconds=self.timeout,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Return-Format": "markdown",
            },
            method="GET",
        )

        if api_response.error:
            result = ToolOutput(
                output=api_response.error,
                called=True,
                error=api_response.error,
                timeout=api_response.timed_out,
                runtime=time.time() - start_time,
            )
            _log_tool_call(self.call_name, url, result)
            return result

        # Extract content from Jina response
        data = api_response.data
        content = ""
        error = ""

        if data.get("code") == 200:
            inner_data = data.get("data", {})
            content = inner_data.get("content") or ""
            title = inner_data.get("title") or ""

            # Format output with title if available
            if title and content:
                content = f"# {title}\n\n{content}"
        else:
            error = f"Jina API error: {data.get('message', 'Unknown error')}"

        output = error if error else content
        result = ToolOutput(output=output, called=True, error=error, timeout=False, runtime=time.time() - start_time)
        _log_tool_call(self.call_name, url, result)
        return result


@dataclass
class JinaBrowseToolConfig(BaseToolConfig):
    """Configuration for the Jina Reader browse tool."""

    tool_class: ClassVar[type[Tool]] = JinaBrowseTool

    timeout: int = 30
    """Timeout in seconds for webpage fetching."""


class S2SearchTool(Tool):
    """
    Search tool using the Semantic Scholar API.
    Requires S2_API_KEY environment variable.
    Get an API key at https://www.semanticscholar.org/product/api.
    """

    config_name = "s2_search"
    description = "Searches Semantic Scholar for academic papers and citations"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query for Semantic Scholar"}},
        "required": ["query"],
    }

    def __init__(self, call_name: str, num_results: int = 10, timeout: int = 60) -> None:
        self.call_name = call_name
        self.num_results = num_results
        self.timeout = timeout
        self.api_key = os.environ.get("S2_API_KEY")
        if not self.api_key:
            raise ValueError("Missing S2_API_KEY environment variable.")

    async def execute(self, query: str) -> ToolOutput:
        """Search Semantic Scholar for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.call_name, query or "", result)
            return result

        start_time = time.time()
        api_response = await make_api_request(
            url="https://api.semanticscholar.org/graph/v1/snippet/search",
            timeout_seconds=self.timeout,
            headers={"x-api-key": self.api_key},
            params={"limit": self.num_results, "query": query},
            method="GET",
        )

        if api_response.error:
            result = ToolOutput(
                output=api_response.error,
                called=True,
                error=api_response.error,
                timeout=api_response.timed_out,
                runtime=time.time() - start_time,
            )
            _log_tool_call(self.call_name, query, result)
            return result

        # Extract snippets from response
        data = api_response.data.get("data", [])
        snippets = [text for item in data if (text := item.get("snippet", {}).get("text"))]

        error = "" if snippets else "Query returned no results."
        output = "\n".join(snippets).strip() if snippets else error

        result = ToolOutput(output=output, called=True, error=error, timeout=False, runtime=time.time() - start_time)
        _log_tool_call(self.call_name, query, result)
        return result


@dataclass
class S2SearchToolConfig(BaseToolConfig):
    """Configuration for the Semantic Scholar search tool."""

    tool_class: ClassVar[type[Tool]] = S2SearchTool

    num_results: int = 10
    """Number of results to return from Semantic Scholar."""
    timeout: int = 60
    """Timeout in seconds for the API request."""


class SerperSearchTool(Tool):
    """
    Search tool using the Serper API (Google Search results).
    Requires SERPER_API_KEY environment variable.
    """

    config_name = "serper_search"
    description = "Google search via the Serper API"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query for Google"}},
        "required": ["query"],
    }

    def __init__(self, call_name: str, num_results: int = 5, timeout: int = 10) -> None:
        self.call_name = call_name
        self.num_results = num_results
        self.timeout = timeout
        self.api_key = os.environ.get("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing SERPER_API_KEY environment variable.")

    async def execute(self, query: str) -> ToolOutput:
        """Search Google via Serper for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.call_name, query or "", result)
            return result

        start_time = time.time()
        api_response = await make_api_request(
            url="https://google.serper.dev/search",
            timeout_seconds=self.timeout,
            json_payload={"q": query, "num": self.num_results},
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
        )

        if api_response.error:
            result = ToolOutput(
                output=api_response.error,
                called=True,
                error=api_response.error,
                timeout=api_response.timed_out,
                runtime=time.time() - start_time,
            )
            _log_tool_call(self.call_name, query, result)
            return result

        # Process the response data
        data = api_response.data
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

        error = "" if snippets else "Query returned no results."
        output = "\n\n".join(snippets).strip() if snippets else error

        result = ToolOutput(output=output, called=True, error=error, timeout=False, runtime=time.time() - start_time)
        _log_tool_call(self.call_name, query, result)
        return result


@dataclass
class SerperSearchToolConfig(BaseToolConfig):
    """Configuration for the Serper (Google Search) tool."""

    tool_class: ClassVar[type[Tool]] = SerperSearchTool

    num_results: int = 5
    """Number of results to return from Serper."""
    timeout: int = 10
    """Timeout in seconds for the API request."""


# DrAgentMCPTool + Config (requires dr_agent package)


class DrAgentMCPTool(Tool):
    """
    Wrapper for MCP (Model Context Protocol) tools from dr_agent.

    Routes calls to the appropriate underlying MCP tool based on the tool name
    in the <call_tool name="..."> tag.

    Requires the dr_agent package: uv sync --extra dr-tulu
    """

    config_name = "mcp"
    description = "MCP tools wrapper supporting snippet_search, google_search, massive_serve, browse_webpage"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The full prompt text containing MCP tool call tags"}},
        "required": ["text"],
    }

    def __init__(
        self,
        call_name: str,
        tool_names: str = "snippet_search",
        parser_name: str = "unified",
        transport_type: str | None = None,
        host: str | None = None,
        port: int | None = None,
        timeout: int = 180,
        max_retries: int = 3,
        base_url: str | None = None,
        num_results: int = 10,
    ) -> None:
        if not DR_AGENT_MCP_AVAILABLE:
            raise ImportError("MCP tools require dr_agent package. Install it with: uv sync --extra dr-tulu")

        self.call_name = call_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.mcp_tools: list[Any] = []
        self.stop_strings: list[str] = []

        # Configure transport from params or environment
        transport = transport_type or os.environ.get("MCP_TRANSPORT", "StreamableHttpTransport")
        mcp_host = host or os.environ.get("MCP_TRANSPORT_HOST", "0.0.0.0")
        mcp_port = port or os.environ.get("MCP_TRANSPORT_PORT", "8000")

        os.environ["MCP_TRANSPORT_HOST"] = str(mcp_host)
        os.environ["MCP_TRANSPORT_PORT"] = str(mcp_port)

        # Parse comma-separated tool names
        tool_name_list = [n.strip() for n in tool_names.split(",") if n.strip()]

        for mcp_tool_name in tool_name_list:
            if mcp_tool_name not in DR_AGENT_MCP_TOOLS:
                raise ValueError(f"Unknown MCP tool: {mcp_tool_name}. Available: {list(DR_AGENT_MCP_TOOLS.keys())}")

            mcp_tool_cls = DR_AGENT_MCP_TOOLS[mcp_tool_name]
            sig = inspect.signature(mcp_tool_cls.__init__)
            valid_params = set(sig.parameters.keys())

            # Build kwargs based on what the tool accepts
            tool_kwargs: dict[str, Any] = {}
            if "base_url" in valid_params:
                tool_kwargs["base_url"] = base_url
            if "number_documents_to_search" in valid_params:
                tool_kwargs["number_documents_to_search"] = num_results
            if mcp_tool_name == "browse_webpage":
                tool_kwargs["use_docker_version"] = True
                tool_kwargs["use_ai2_config"] = True

            self.mcp_tools.append(
                mcp_tool_cls(
                    timeout=timeout,
                    name=mcp_tool_name,
                    tool_parser=parser_name,
                    transport_type=transport,
                    **tool_kwargs,
                )
            )
            self.stop_strings += self.mcp_tools[-1].tool_parser.stop_sequences

        logger.info(f"DrAgentMCPTool: initialized with tools {tool_name_list}")

    def get_stop_strings(self) -> list[str]:
        """Return the stop strings for all MCP tools."""
        return self.stop_strings

    async def _call_with_retry(self, mcp_tool: Any, text: str) -> Any:
        """Call an MCP tool with retry logic."""
        last_error: Exception | None = None
        retryable = (aiohttp.ClientError, ConnectionError, TimeoutError, asyncio.TimeoutError)

        for attempt in range(self.max_retries):
            try:
                return await mcp_tool(text)
            except retryable as e:
                last_error = e
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry logic error")

    async def execute(self, text: str) -> ToolOutput:
        """Execute the appropriate MCP tool based on the text content."""
        if not text or not text.strip():
            result = ToolOutput(
                output="",
                error="Empty text. Please provide text containing tool calls.",
                called=True,
                timeout=False,
                runtime=0,
            )
            _log_tool_call(self.call_name, text or "", result)
            return result

        start_time = time.time()

        # Find and execute the matching MCP tool
        for mcp_tool in self.mcp_tools:
            if mcp_tool.tool_parser.has_calls(text, mcp_tool.name):
                try:
                    mcp_output = await self._call_with_retry(mcp_tool, text)
                    text_output = mcp_tool._format_output(mcp_output)
                    text_output = mcp_tool.tool_parser.format_result(text_output, mcp_output)

                    result = ToolOutput(
                        output=text_output,
                        called=True,
                        error=mcp_output.error or "",
                        timeout=mcp_output.timeout,
                        runtime=mcp_output.runtime,
                    )
                    _log_tool_call(self.call_name, text, result)
                    return result

                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"DrAgentMCPTool error: {error_msg}")
                    result = ToolOutput(
                        output=error_msg,
                        called=True,
                        error=error_msg,
                        timeout=False,
                        runtime=time.time() - start_time,
                    )
                    _log_tool_call(self.call_name, text, result)
                    return result

        # No matching tool found
        error_msg = "No valid tool calls found in text."
        result = ToolOutput(
            output=error_msg, called=False, error=error_msg, timeout=False, runtime=time.time() - start_time
        )
        _log_tool_call(self.call_name, text, result)
        return result


@dataclass
class DrAgentMCPToolConfig(BaseToolConfig):
    """Configuration for MCP (Model Context Protocol) tools.

    Requires the dr_agent package: uv sync --extra dr-tulu
    """

    tool_class: ClassVar[type[Tool]] = DrAgentMCPTool

    tool_names: str = "snippet_search"
    """Comma-separated list of MCP tool names.
    Available: snippet_search, google_search, massive_serve, browse_webpage"""
    parser_name: str = "unified"
    """The parser name for MCP tools."""
    transport_type: str | None = None
    """Transport type for MCP (default: StreamableHttpTransport)."""
    host: str | None = None
    """Host for MCP transport (default: from MCP_TRANSPORT_HOST env var)."""
    port: int | None = None
    """Port for MCP transport (default: from MCP_TRANSPORT_PORT env var)."""
    timeout: int = 180
    """Timeout in seconds for MCP tool calls."""
    max_retries: int = 3
    """Maximum retries for transient errors."""
    base_url: str | None = None
    """Base URL for MCP tools."""
    num_results: int = 10
    """Number of results to retrieve."""


# Tool Registry: Maps tool names to their config classes
TOOL_REGISTRY: dict[str, type[BaseToolConfig]] = {
    PythonCodeToolConfig.tool_class.config_name: PythonCodeToolConfig,
    JinaBrowseToolConfig.tool_class.config_name: JinaBrowseToolConfig,
    S2SearchToolConfig.tool_class.config_name: S2SearchToolConfig,
    SerperSearchToolConfig.tool_class.config_name: SerperSearchToolConfig,
    DrAgentMCPToolConfig.tool_class.config_name: DrAgentMCPToolConfig,
}
