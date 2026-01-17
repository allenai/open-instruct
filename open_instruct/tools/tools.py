"""
Basic tools that are built-in to open-instruct.
"""

import inspect
import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, ClassVar

from open_instruct import logger_utils
from open_instruct.tools.utils import BaseToolConfig, Tool, ToolOutput, make_api_request

logger = logger_utils.setup_logger(__name__)


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


# Tool Registry: Maps tool names to their config classes
TOOL_REGISTRY: dict[str, type[BaseToolConfig]] = {
    PythonCodeToolConfig.tool_class.config_name: PythonCodeToolConfig,
    JinaBrowseToolConfig.tool_class.config_name: JinaBrowseToolConfig,
    S2SearchToolConfig.tool_class.config_name: S2SearchToolConfig,
    SerperSearchToolConfig.tool_class.config_name: SerperSearchToolConfig,
    Crawl4AIBrowseToolConfig.tool_class.config_name: Crawl4AIBrowseToolConfig,
    DrAgentMCPToolConfig.tool_class.config_name: DrAgentMCPToolConfig,
}
