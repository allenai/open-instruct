"""
Basic tools that are built-in to open-instruct.
"""

import time
import traceback
from dataclasses import asdict, dataclass
from typing import Annotated, ClassVar

import httpx
from pydantic import Field

from open_instruct.logger_utils import setup_logger
from open_instruct.tools.utils import BaseToolConfig, Tool, ToolOutput

logger = setup_logger(__name__)


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

    default_tool_name = "python"
    default_description = "Executes Python code and returns printed output."

    def __init__(self, api_endpoint: str, timeout: int = 3, override_name: str | None = None) -> None:
        self.api_endpoint = api_endpoint
        self.timeout = timeout
        self.override_name = override_name

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
                response.raise_for_status()
                res = response.json()
                output = res.get("output", "")
                error = res.get("error") or ""

                all_outputs.append(output)
                if error:
                    all_outputs.append("\n" + error)
        except httpx.TimeoutException:
            error = f"Timeout after {self.timeout} seconds"
            all_outputs.append(error)
            timed_out = True
        except Exception as e:
            error = f"Error calling API: {e}\n{traceback.format_exc()}"
            all_outputs.append(error)

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
