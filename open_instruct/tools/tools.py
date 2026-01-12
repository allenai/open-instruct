"""
Tools that follow the Tool(ABC) pattern.

Each tool has a corresponding Config dataclass that inherits from ToolConfig.
The config's build() method creates the tool instance. Most configs use the
generic build() from ToolConfig; override only when custom logic is needed.
"""

import logging
import os
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Annotated, Any, ClassVar

import httpx
from pydantic import Field

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


@dataclass
class MaxCallsExceededToolConfig(BaseToolConfig):
    """Config for MaxCallsExceededTool."""

    def build(self) -> MaxCallsExceededTool:
        return MaxCallsExceededTool()


# =============================================================================
# PythonCodeTool + Config
# =============================================================================


class PythonCodeTool(Tool):
    """
    Executes Python code via a FastAPI endpoint.
    """

    _default_tool_function_name = "python"
    _default_tool_description = "Executes Python code and returns printed output."

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
# S2SearchTool + Config
# =============================================================================


class S2SearchTool(Tool):
    """
    Search tool using the Semantic Scholar API.
    Requires S2_API_KEY environment variable.
    """

    _default_tool_function_name = "s2_search"
    _default_tool_description = "Searches Semantic Scholar for academic papers and citations"

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

    _default_tool_function_name = "serper_search"
    _default_tool_description = "Google search via the Serper API"

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
