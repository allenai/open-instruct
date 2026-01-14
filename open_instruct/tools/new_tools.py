"""
Basic tools that are built-in to open-instruct.
"""

import os
import time
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
        # Set instance-specific attributes
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
            url=self.api_endpoint, json_payload={"code": code, "timeout": self.timeout}, timeout_seconds=self.timeout
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
            output = api_response.data.get("output", "")
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
            json_payload={"q": query, "num": self.num_results},
            timeout_seconds=self.timeout,
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
        )

        if api_response.error:
            result = ToolOutput(
                output="",
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

        output = "\n\n".join(snippets).strip() if snippets else ""
        error = "" if snippets else "Query returned no results."

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
    SerperSearchToolConfig.tool_class.config_name: SerperSearchToolConfig,
}
