"""
Basic tools that are built-in to open-instruct.
"""

import asyncio
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
            name="python",
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
            _log_tool_call(self.name, code or "", result)
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
        _log_tool_call(self.name, code, result)
        return result


@dataclass
class PythonCodeToolConfig(BaseToolConfig):
    """Configuration for the Python code execution tool."""

    tool_class: ClassVar[type[Tool]] = PythonCodeTool

    api_endpoint: str
    """The API endpoint for the code execution server."""
    timeout: int = 3
    """Timeout in seconds for code execution."""
