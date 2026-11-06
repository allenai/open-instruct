"""Opt-in anonymous web search using Parallel's MCP server."""

import asyncio
import time
from builtins import ExceptionGroup
from dataclasses import dataclass
from typing import ClassVar

from httpx import TimeoutException
from mcp import ClientSession, McpError
from mcp.client.streamable_http import streamablehttp_client

from open_instruct.environments.base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult
from open_instruct.environments.tools.generic_mcp import _extract_mcp_result
from open_instruct.environments.tools.utils import Tool, coerce_args, log_env_call

PARALLEL_MCP_URL = "https://mcp.parallel.ai/mcp"
PARALLEL_USER_AGENT = "allenai-open-instruct/parallel-search"


def _error_details(error: Exception) -> tuple[str, bool]:
    """Keep nested SDK/HTTP errors useful at the tool boundary."""
    if isinstance(error, ExceptionGroup):
        details = [_error_details(e) for e in error.exceptions]
        return "; ".join(message for message, _ in details), any(timed_out for _, timed_out in details)
    if isinstance(error, McpError):
        return error.error.model_dump_json(exclude_none=True), error.error.code == 408
    return f"{type(error).__name__}: {error}", isinstance(error, (TimeoutError, TimeoutException))


class ParallelSearchTool(Tool):
    """Anonymous Parallel search, enabled only by selecting parallel_search."""

    config_name = "parallel_search"
    description = "Web search via Parallel's anonymous MCP server"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query for the web"}},
        "required": ["query"],
    }

    def __init__(self, call_name: str, timeout: int = 60) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.call_name = call_name
        self.timeout = timeout

    async def _search(self, query: str) -> tuple[str, bool]:
        # The SDK owns HTTP connections, protocol framing and best-effort teardown.
        # These headers apply to every request in this session, including notifications.
        async with streamablehttp_client(  # noqa: SIM117
            PARALLEL_MCP_URL,
            headers={"User-Agent": PARALLEL_USER_AGENT},
            timeout=self.timeout,
            sse_read_timeout=self.timeout,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool("web_search", {"objective": query, "search_queries": [query]})
                return _extract_mcp_result(response), bool(response.isError)

    async def step(self, call: EnvCall) -> StepResult:
        args = coerce_args(self.parameters, call.args)
        query = args.get("query", "")
        if not query or not query.strip():
            result = StepResult(
                result="", metadata={"error": "Empty query. Please provide a search query.", "runtime": 0}
            )
            log_env_call(self.call_name, query or "", result)
            return result

        start = time.monotonic()
        timed_out = False
        try:
            output, is_error = await asyncio.wait_for(self._search(query), timeout=self.timeout)
            error = output if is_error else ""
            if not output.strip():
                output = error = "Query returned no results."
        except Exception as exc:
            output, timed_out = _error_details(exc)
            error = output

        result = StepResult(
            result=output, metadata={"error": error, "timeout": timed_out, "runtime": time.monotonic() - start}
        )
        log_env_call(self.call_name, query, result)
        return result


@dataclass
class ParallelSearchToolConfig(BaseEnvConfig):
    """Configuration for opt-in anonymous Parallel search. No API key is read."""

    tool_class: ClassVar[type[RLEnvironment]] = ParallelSearchTool

    timeout: int = 60
    """Total timeout in seconds, including MCP initialization and cleanup."""
