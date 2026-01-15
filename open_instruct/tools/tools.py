"""
Basic tools that are built-in to open-instruct.
"""

import os
import time
import urllib.parse
from dataclasses import dataclass
from typing import ClassVar

from open_instruct import logger_utils
from open_instruct.tools.utils import BaseToolConfig, Tool, ToolOutput, make_api_request

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


# Tool Registry: Maps tool names to their config classes
TOOL_REGISTRY: dict[str, type[BaseToolConfig]] = {
    PythonCodeToolConfig.tool_class.config_name: PythonCodeToolConfig,
    JinaBrowseToolConfig.tool_class.config_name: JinaBrowseToolConfig,
    S2SearchToolConfig.tool_class.config_name: S2SearchToolConfig,
    SerperSearchToolConfig.tool_class.config_name: SerperSearchToolConfig,
}
