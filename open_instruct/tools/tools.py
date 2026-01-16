"""
Basic tools that are built-in to open-instruct.
"""

import asyncio
import concurrent.futures
import inspect
import os
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from open_instruct import logger_utils
from open_instruct.tools.utils import BaseToolConfig, Tool, ToolOutput, make_api_request

logger = logger_utils.setup_logger(__name__)


# Optional dr_agent imports for legacy MCP tools
try:
    from dr_agent.tool_interface.mcp_tools import (
        Crawl4AIBrowseTool,
        MassiveServeSearchTool,
        SemanticScholarSnippetSearchTool,
    )
    from dr_agent.tool_interface.mcp_tools import SerperSearchTool as DrAgentSerperSearchTool

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


def _parse_crawl4ai_response(data: dict, include_html: bool, max_content_length: int | None) -> tuple[str, str]:
    """Parse Crawl4AI API response and extract content.

    Returns:
        Tuple of (content, error). If successful, error is empty.
    """
    results = data.get("results", [])
    if not results:
        return "", f"Crawl4AI error: {data.get('error', 'No results returned')}"

    result_data = results[0]
    if not result_data.get("success", False):
        error_msg = result_data.get("error_message", result_data.get("error", "Unknown error"))
        return "", f"Crawl4AI error: {error_msg}"

    # Prefer fit_markdown (pruned), then markdown, then html
    markdown_data = result_data.get("markdown", "")
    if isinstance(markdown_data, dict):
        content = markdown_data.get("fit_markdown") or markdown_data.get("raw_markdown") or ""
    else:
        content = markdown_data or ""

    if not content and include_html:
        content = result_data.get("html") or result_data.get("cleaned_html") or ""

    metadata = result_data.get("metadata", {})
    title = metadata.get("title", "") if isinstance(metadata, dict) else ""
    if title and content:
        content = f"# {title}\n\n{content}"

    if max_content_length and len(content) > max_content_length:
        content = content[:max_content_length] + "\n\n[Content truncated]"

    return content, ""


class Crawl4AIBrowseTool(Tool):
    """
    Tool for fetching webpage content using Crawl4AI Docker API.
    Requires CRAWL4AI_API_URL, CRAWL4AI_API_KEY, and CRAWL4AI_BLOCKLIST_PATH environment variables.

    This tool uses the Docker version with AI2 configuration by default.
    Based on: https://github.com/rlresearch/dr-tulu/blob/main/agent/dr_agent/tool_interface/mcp_tools.py
    """

    config_name = "crawl4ai_browse"
    description = "Fetches and converts webpage content to clean markdown using Crawl4AI"
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The URL of the webpage to fetch"}},
        "required": ["url"],
    }

    def __init__(
        self,
        call_name: str,
        timeout: int = 180,
        ignore_links: bool = True,
        bypass_cache: bool = False,
        include_html: bool = False,
        max_content_length: int | None = 5000,
    ) -> None:
        self.call_name = call_name
        self.timeout = timeout
        self.ignore_links = ignore_links
        self.bypass_cache = bypass_cache
        self.include_html = include_html
        self.max_content_length = max_content_length

        self.api_url = os.environ.get("CRAWL4AI_API_URL")
        if not self.api_url:
            raise ValueError("Missing CRAWL4AI_API_URL environment variable.")
        self.api_key = os.environ.get("CRAWL4AI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing CRAWL4AI_API_KEY environment variable.")

        blocklist_path = os.environ.get("CRAWL4AI_BLOCKLIST_PATH")
        if not blocklist_path:
            raise ValueError("Missing CRAWL4AI_BLOCKLIST_PATH environment variable.")
        if not os.path.exists(blocklist_path):
            raise FileNotFoundError(f"Blocklist file not found: {blocklist_path}")
        with open(blocklist_path, encoding="utf-8") as f:
            self.blocklist = [line.strip() for line in f if line.strip()]

    async def execute(self, url: str) -> ToolOutput:
        """Fetch webpage content via Crawl4AI Docker API."""
        if not url or not url.strip():
            result = ToolOutput(
                output="", error="Empty URL. Please provide a URL to fetch.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.call_name, url or "", result)
            return result

        start_time = time.time()

        # Build request payload matching Crawl4AI Docker API format
        # See: https://docs.crawl4ai.com/core/docker-deployment/
        crawler_params: dict = {
            "cache_mode": "bypass" if self.bypass_cache else "enabled",
            "word_count_threshold": 10,
            "exclude_social_media_links": True,
            "excluded_tags": ["form", "header", "footer", "nav"],
            "page_timeout": (self.timeout - 1) * 1000,  # Convert to ms, leave 1s buffer
            "exclude_domains": self.blocklist,
        }

        if self.ignore_links:
            crawler_params["exclude_external_links"] = True

        payload = {
            "urls": [url.strip()],
            "browser_config": {"type": "BrowserConfig", "params": {"headless": True}},
            "crawler_config": {"type": "CrawlerRunConfig", "params": crawler_params},
        }

        headers = {"Content-Type": "application/json", "x-api-key": self.api_key}

        crawl_endpoint = f"{self.api_url.rstrip('/')}/crawl"
        api_response = await make_api_request(
            url=crawl_endpoint, timeout_seconds=self.timeout, json_payload=payload, headers=headers
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

        content, error = _parse_crawl4ai_response(api_response.data, self.include_html, self.max_content_length)
        output = error if error else content
        result = ToolOutput(output=output, called=True, error=error, timeout=False, runtime=time.time() - start_time)
        _log_tool_call(self.call_name, url, result)
        return result


class DrAgentMCPTool(Tool):
    """Wrapper for MCP tools from dr_agent. Requires: uv sync --extra dr-tulu"""

    config_name = "dr_agent_mcp"
    description = "MCP tools wrapper"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text containing MCP tool calls"}},
        "required": ["text"],
    }

    def __init__(
        self,
        call_name: str,
        tool_names: str,
        parser_name: str,
        transport_type: str | None = None,
        host: str | None = None,
        port: int | None = None,
        timeout: int = 180,
        base_url: str | None = None,
        num_results: int = 10,
    ) -> None:
        if not DR_AGENT_MCP_AVAILABLE:
            raise ImportError("MCP tools require dr_agent package. Install with: uv sync --extra dr-tulu")

        self.call_name = call_name
        self.mcp_tools: list[Any] = []
        self.stop_strings: list[str] = []

        transport = transport_type or os.environ.get("MCP_TRANSPORT", "StreamableHttpTransport")
        resolved_host = host or os.environ.get("MCP_TRANSPORT_HOST", "0.0.0.0")
        resolved_port = port or int(os.environ.get("MCP_TRANSPORT_PORT", "8000"))

        for name in [n.strip() for n in tool_names.split(",") if n.strip()]:
            if name not in DR_AGENT_MCP_TOOLS:
                raise ValueError(f"Unknown MCP tool: {name}. Available: {list(DR_AGENT_MCP_TOOLS.keys())}")

            cls = DR_AGENT_MCP_TOOLS[name]
            valid_params = set(inspect.signature(cls.__init__).parameters.keys())
            kwargs: dict[str, Any] = {}
            if "host" in valid_params:
                kwargs["host"] = resolved_host
            if "port" in valid_params:
                kwargs["port"] = resolved_port
            if "base_url" in valid_params:
                kwargs["base_url"] = base_url
            if "number_documents_to_search" in valid_params:
                kwargs["number_documents_to_search"] = num_results
            if name == "browse_webpage":
                kwargs["use_docker_version"] = True
                kwargs["use_ai2_config"] = True

            tool = cls(timeout=timeout, name=name, tool_parser=parser_name, transport_type=transport, **kwargs)
            self.mcp_tools.append(tool)
            self.stop_strings.extend(tool.tool_parser.stop_sequences)

    def get_stop_strings(self) -> list[str]:
        return self.stop_strings

    async def execute(self, text: str) -> ToolOutput:
        if not text or not text.strip():
            return ToolOutput(output="", error="Empty input", called=True, timeout=False, runtime=0)

        start_time = time.time()
        outputs: list[str] = []
        errors: list[str] = []
        any_timeout = False

        for mcp_tool in self.mcp_tools:
            if mcp_tool.tool_parser.has_calls(text, mcp_tool.name):
                try:
                    output = await mcp_tool(text)
                    formatted = mcp_tool.tool_parser.format_result(mcp_tool._format_output(output), output)
                    outputs.append(formatted)
                    if output.error:
                        errors.append(output.error)
                    if output.timeout:
                        any_timeout = True
                except Exception as e:
                    outputs.append(str(e))
                    errors.append(str(e))

        if not outputs:
            result = ToolOutput(
                output="", called=False, error="No tool calls found", timeout=False, runtime=time.time() - start_time
            )
        else:
            result = ToolOutput(
                output="\n".join(outputs),
                called=True,
                error="; ".join(errors) if errors else "",
                timeout=any_timeout,
                runtime=time.time() - start_time,
            )
        _log_tool_call(self.call_name, text, result)
        return result


@dataclass
class Crawl4AIBrowseToolConfig(BaseToolConfig):
    """Configuration for the Crawl4AI browse tool."""

    tool_class: ClassVar[type[Tool]] = Crawl4AIBrowseTool

    timeout: int = 180
    """Timeout in seconds for webpage fetching."""
    ignore_links: bool = True
    """Whether to exclude external and social media links from the content."""
    bypass_cache: bool = False
    """Whether to bypass the cache and fetch fresh content."""
    include_html: bool = False
    """Whether to include HTML content as fallback if markdown is unavailable."""
    max_content_length: int | None = 5000
    """Maximum content length in characters. None means no limit."""


@dataclass
class DrAgentMCPToolConfig(BaseToolConfig):
    """Config for MCP tools. Requires: uv sync --extra dr-tulu"""

    tool_class: ClassVar[type[Tool]] = DrAgentMCPTool

    tool_names: str
    parser_name: str
    transport_type: str | None = None
    host: str | None = None
    port: int | None = None
    timeout: int = 180
    base_url: str | None = None
    num_results: int = 10


# =============================================================================
# Generic MCP Tool (connects to any MCP server)
# =============================================================================


class MCPTransport(str, Enum):
    """Transport types for MCP connections."""

    HTTP = "http"
    SSE = "sse"
    STDIO = "stdio"


class MCPToolFactory:
    """
    Handles MCP server connections, tool discovery, and tool calls.

    Used internally by GenericMCPTool and GenericMCPToolConfig.
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

    Requires the mcp package: uv sync --extra mcp

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

        configs = []
        for name in tool_names:
            configs.append(
                GenericMCPToolConfig(
                    server_url=self.server_url,
                    transport=self.transport,
                    command=self.command,
                    args=self.args,
                    env=self.env,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    retry_backoff=self.retry_backoff,
                    tool_name=name,
                )
            )
        return configs


# Tool Registry: Maps tool names to their config classes
TOOL_REGISTRY: dict[str, type[BaseToolConfig]] = {
    PythonCodeToolConfig.tool_class.config_name: PythonCodeToolConfig,
    JinaBrowseToolConfig.tool_class.config_name: JinaBrowseToolConfig,
    S2SearchToolConfig.tool_class.config_name: S2SearchToolConfig,
    SerperSearchToolConfig.tool_class.config_name: SerperSearchToolConfig,
    Crawl4AIBrowseToolConfig.tool_class.config_name: Crawl4AIBrowseToolConfig,
    DrAgentMCPToolConfig.tool_class.config_name: DrAgentMCPToolConfig,
    GenericMCPToolConfig.tool_class.config_name: GenericMCPToolConfig,
}
