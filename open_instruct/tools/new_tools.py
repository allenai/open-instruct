"""
Basic tools that are built-in to open-instruct.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import ClassVar

import aiohttp

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


class PythonCodeTool(Tool):
    """
    Executes Python code via a FastAPI endpoint.
    """

    def __init__(self, api_endpoint: str, timeout: int = 3) -> None:
        super().__init__(
            config_name="python",
            description="Executes Python code and returns printed output.",
            call_name="python",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
        )
        self.api_endpoint = api_endpoint
        self.timeout = timeout

    async def __call__(self, code: str) -> ToolOutput:
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
        all_outputs = []
        timed_out = False
        error = ""

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:  # noqa: SIM117
                async with session.post(self.api_endpoint, json={"code": code, "timeout": self.timeout}) as response:
                    response.raise_for_status()
                    res = await response.json()
                    output = res.get("output", "")
                    error = res.get("error") or ""

                    all_outputs.append(output)
                    if error:
                        all_outputs.append("\n" + error)
        except asyncio.TimeoutError:
            error = f"Timeout after {self.timeout} seconds"
            all_outputs.append(error)
            timed_out = True
        except aiohttp.ClientResponseError as e:
            error = f"HTTP error: {e.status} {e.message}"
            all_outputs.append(error)
        except aiohttp.ClientError as e:
            error = f"Connection error: {e}"
            all_outputs.append(error)

        result = ToolOutput(
            output="\n".join(all_outputs),
            called=True,
            error=error,
            timeout=timed_out,
            runtime=time.time() - start_time,
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


class S2SearchTool(Tool):
    """
    Search tool using the Semantic Scholar API.
    Requires S2_API_KEY environment variable.

    Semantic Scholar is a free AI-powered research tool for scientific literature.
    Get an API key at https://www.semanticscholar.org/product/api
    """

    def __init__(self, num_results: int = 10, timeout: int = 60) -> None:
        super().__init__(
            config_name="s2_search",
            description="Searches Semantic Scholar for academic papers and citations",
            call_name="s2_search",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query for Semantic Scholar"}},
                "required": ["query"],
            },
        )
        self.num_results = num_results
        self.timeout = timeout
        self.api_key = os.environ.get("S2_API_KEY")
        if not self.api_key:
            raise ValueError("Missing S2_API_KEY environment variable.")

    async def __call__(self, query: str) -> ToolOutput:
        """Search Semantic Scholar for documents matching the query."""
        if not query or not query.strip():
            result = ToolOutput(
                output="", error="Empty query. Please provide a search query.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.call_name, query or "", result)
            return result

        start_time = time.time()
        timed_out = False
        error = ""
        snippets = []

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"x-api-key": self.api_key}

            async with aiohttp.ClientSession(timeout=timeout) as session:  # noqa: SIM117
                async with session.get(
                    "https://api.semanticscholar.org/graph/v1/snippet/search",
                    params={"limit": self.num_results, "query": query},
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    res = await response.json()
                    data = res.get("data", [])
                    snippets = snippets = [text for item in data if (text := item.get("snippet", {}).get("text"))]

                    if not snippets:
                        error = "Query returned no results."

        except asyncio.TimeoutError:
            error = f"Timeout after {self.timeout} seconds"
            timed_out = True
        except aiohttp.ClientResponseError as e:
            error = f"HTTP error: {e.status} {e.message}"
        except aiohttp.ClientError as e:
            error = f"Connection error: {e}"
        except Exception as e:
            error = f"Unexpected error: {e}"

        result = ToolOutput(
            output="\n".join(snippets).strip() if not error else "",
            called=True,
            error=error,
            timeout=timed_out,
            runtime=time.time() - start_time,
        )
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
