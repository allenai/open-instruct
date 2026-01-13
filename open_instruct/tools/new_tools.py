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


class JinaBrowseTool(Tool):
    """
    Tool for fetching webpage content using Jina Reader API.
    Converts webpages to clean, LLM-friendly markdown format.

    Jina Reader is a free API for converting web pages to clean text.
    Get an API key at https://jina.ai/reader/
    """

    def __init__(self, timeout: int = 30) -> None:
        super().__init__(
            config_name="jina_browse",
            description="Fetches and converts webpage content to clean markdown using Jina Reader API",
            call_name="jina_browse",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL of the webpage to fetch"}},
                "required": ["url"],
            },
        )
        self.timeout = timeout
        self.api_key = os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Missing JINA_API_KEY environment variable.")

    async def __call__(self, url: str) -> ToolOutput:
        """Fetch webpage content via Jina Reader API."""
        if not url or not url.strip():
            result = ToolOutput(
                output="", error="Empty URL. Please provide a URL to fetch.", called=True, timeout=False, runtime=0
            )
            _log_tool_call(self.call_name, url or "", result)
            return result

        start_time = time.time()
        timed_out = False
        error = ""
        content = ""

        try:
            # Jina Reader API endpoint
            api_url = f"https://r.jina.ai/{url.strip()}"

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Return-Format": "markdown",
            }

            async with aiohttp.ClientSession(timeout=timeout) as session:  # noqa: SIM117
                async with session.get(api_url, headers=headers) as response:
                    response.raise_for_status()
                    res = await response.json()

                    # Extract content from Jina response
                    if res.get("code") == 200:
                        data = res.get("data", {})
                        content = data.get("content", "")
                        title = data.get("title", "")

                        # Format output with title if available
                        if title and content:
                            content = f"# {title}\n\n{content}"
                    else:
                        error = f"Jina API error: {res.get('message', 'Unknown error')}"

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
            output=content if not error else "",
            called=True,
            error=error,
            timeout=timed_out,
            runtime=time.time() - start_time,
        )
        _log_tool_call(self.call_name, url, result)
        return result


@dataclass
class JinaBrowseToolConfig(BaseToolConfig):
    """Configuration for the Jina Reader browse tool."""

    tool_class: ClassVar[type[Tool]] = JinaBrowseTool

    timeout: int = 30
    """Timeout in seconds for webpage fetching."""
