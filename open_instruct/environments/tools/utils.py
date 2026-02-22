import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import backoff
from openenv.core.env_server.types import State

from open_instruct import logger_utils
from open_instruct.data_types import ToolCallStats
from open_instruct.environments.base import RLEnvironment, StepResult

logger = logger_utils.setup_logger(__name__)


@dataclass
class ParsedEnvConfig:
    """A parsed tool configuration combining name, call name, and config."""

    name: str
    """The tool name (e.g., "python", "search")."""

    call_name: str
    """The name used in tool calls (e.g., "code", "web_search")."""

    config: dict[str, Any]
    """The parsed configuration dictionary for this tool."""


@dataclass
class EnvsConfig:
    """Configuration for tools used during generation."""

    tools: list[str] = field(default_factory=list)
    """List of tool names to enable (e.g., ["python", "search"])."""

    tool_call_names: list[str] = field(default_factory=list)
    """Override names used in tool calls (e.g., '<name>...</name>').
    Must match length of tools if set. Defaults to tools if not specified."""

    tool_configs: list[str] = field(default_factory=list)
    """JSON strings for configuring each tool. Must match length of tools. Use '{}' for defaults."""

    tool_parser_type: str = "legacy"
    """Type of tool parser to use. See parsers.get_available_parsers() for valid options."""

    max_steps: int = 5
    """Maximum number of tool calls or environment steps per generation."""

    only_reward_good_outputs: bool = False
    """Only apply rewards to outputs from tools that didn't error."""

    pass_tools_to_chat_template: bool = True
    """Pass tool definitions to the chat template. Set to False if using a custom system prompt."""

    pool_size: int | None = None
    """Number of actors per tool pool. Defaults to num_unique_prompts_rollout * num_samples_per_prompt_rollout."""

    _parsed_tools: list[ParsedEnvConfig] = field(default_factory=list, init=False)
    """Parsed tool configurations. Populated during __post_init__."""

    def __post_init__(self):
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")

        if not self.tools:
            return

        # Set defaults for empty lists
        if not self.tool_call_names:
            self.tool_call_names = list(self.tools)
        if not self.tool_configs:
            self.tool_configs = ["{}"] * len(self.tools)

        # Validate lengths
        if len(self.tool_call_names) != len(self.tools):
            raise ValueError(
                f"tool_call_names must have same length as tools. "
                f"Got {len(self.tool_call_names)} names for {len(self.tools)} tools."
            )
        if len(self.tool_configs) != len(self.tools):
            raise ValueError(
                f"tool_configs must have same length as tools. "
                f"Got {len(self.tool_configs)} configs for {len(self.tools)} tools."
            )

        # Parse and combine into ParsedTool instances
        for i, (tool_name, call_name, config_str) in enumerate(
            zip(self.tools, self.tool_call_names, self.tool_configs)
        ):
            try:
                config = json.loads(config_str)
            except Exception as e:
                raise ValueError(f"Invalid tool_config for tool {tool_name} at index {i}: {e}") from e
            self._parsed_tools.append(ParsedEnvConfig(name=tool_name, call_name=call_name, config=config))

    @property
    def enabled(self) -> bool:
        """Return True if any tools are configured."""
        return bool(self.tools)


def truncate(text: str, max_length: int = 500) -> str:
    """Truncate text for logging, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [{len(text) - max_length} more chars]"


def log_env_call(tool_name: str, input_text: str, result: StepResult) -> None:
    """Log a tool call at DEBUG level with truncated input/output."""
    logger.debug(
        f"Tool '{tool_name}' called:\n"
        f"  Input: {truncate(input_text)}\n"
        f"  Output: {truncate(result.result)}\n"
        f"  Error: {result.metadata.get('error', '') or 'None'}\n"
        f"  Runtime: {result.metadata.get('runtime', 0):.3f}s, Timeout: {result.metadata.get('timeout', False)}"
    )


class EnvStatistics:
    """Manages aggregated tool call statistics across rollouts.

    Provides methods to add rollout stats and compute per-tool and aggregate metrics.
    """

    def __init__(self, tool_names: list[str] | None = None):
        """Initialize tool statistics tracker.

        Args:
            tool_names: List of tool names to track. New tool names seen in add_rollout are added automatically.
        """
        self.tool_names: set[str] = set(tool_names) if tool_names else set()
        self.num_rollouts = 0
        self._counts: defaultdict[str, int] = defaultdict(int)
        self._failures: defaultdict[str, int] = defaultdict(int)
        self._runtimes: defaultdict[str, float] = defaultdict(float)

    def add_rollout(self, tool_call_stats: list[ToolCallStats]) -> None:
        """Add statistics from a single rollout."""
        self.num_rollouts += 1
        for s in tool_call_stats:
            self.tool_names.add(s.tool_name)
            self._counts[s.tool_name] += 1
            self._failures[s.tool_name] += not s.success
            self._runtimes[s.tool_name] += s.runtime

    def compute_metrics(self) -> dict[str, float]:
        """Compute per-tool and aggregate metrics.

        Returns:
            Dictionary with metrics for each tool and aggregate totals:
            - tools/{name}/avg_calls_per_rollout
            - tools/{name}/failure_rate
            - tools/{name}/avg_runtime
            - tools/aggregate/avg_calls_per_rollout
            - tools/aggregate/failure_rate
            - tools/aggregate/avg_runtime
        """
        if not self.num_rollouts or not self.tool_names:
            return {}

        metrics: dict[str, float] = {}
        total_calls = 0
        total_failures = 0
        total_runtime = 0.0

        for name in self.tool_names:
            calls, failures, runtime = self._counts[name], self._failures[name], self._runtimes[name]
            metrics[f"tools/{name}/avg_calls_per_rollout"] = calls / self.num_rollouts
            metrics[f"tools/{name}/failure_rate"] = failures / calls if calls else 0.0
            metrics[f"tools/{name}/avg_runtime"] = runtime / calls if calls else 0.0
            total_calls += calls
            total_failures += failures
            total_runtime += runtime

        metrics["tools/aggregate/avg_calls_per_rollout"] = total_calls / self.num_rollouts
        metrics["tools/aggregate/failure_rate"] = total_failures / total_calls if total_calls else 0.0
        metrics["tools/aggregate/avg_runtime"] = total_runtime / total_calls if total_calls else 0.0

        return metrics


@dataclass
class APIResponse:
    """Response from an async API request."""

    data: dict = field(default_factory=dict)
    error: str = ""
    timed_out: bool = False


class _RetryableHTTPError(Exception):
    """Exception raised for HTTP errors that should be retried."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(f"HTTP {status}: {message}")


def _is_retryable_status(status: int) -> bool:
    """Check if an HTTP status code is retryable (429 or 5xx)."""
    return status == 429 or status >= 500


def _should_giveup(exception: Exception) -> bool:
    """Determine if we should give up retrying based on the exception type.

    Returns True (give up) for non-retryable errors, False (retry) for retryable ones.

    Retries on:
    - Timeouts (asyncio.TimeoutError)
    - Connection errors (aiohttp.ClientConnectionError - includes ClientConnectorError,
      ServerDisconnectedError, etc.)
    - Our custom _RetryableHTTPError (for 429/5xx detected before raise_for_status)

    Does NOT retry on:
    - aiohttp.ClientResponseError (4xx errors from raise_for_status)
    - Other non-connection ClientError subclasses
    """
    # Return False (don't give up, retry) for retryable errors, True (give up) for others
    return not isinstance(exception, (asyncio.TimeoutError, _RetryableHTTPError, aiohttp.ClientConnectionError))


async def make_api_request(
    url: str,
    timeout_seconds: int,
    headers: dict | None = None,
    json_payload: dict | None = None,
    params: dict | None = None,
    method: str = "POST",
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> APIResponse:
    """Make an async HTTP request with standard error handling and retries.

    Retries on:
    - Timeouts
    - Connection errors
    - HTTP 429 (Too Many Requests)
    - HTTP 5xx (Server errors)

    Does NOT retry on HTTP 4xx errors (except 429).

    Args:
        url: The API endpoint URL.
        timeout_seconds: Request timeout in seconds.
        headers: Optional HTTP headers.
        json_payload: JSON data to send in the request body.
        params: Query parameters.
        method: HTTP method ("GET" or "POST"). Defaults to "POST".
        max_retries: Maximum number of retry attempts. Defaults to 3.
        base_delay: Base delay in seconds for exponential backoff. Defaults to 1.0.
        max_delay: Maximum delay in seconds between retries. Defaults to 60.0.

    Returns:
        APIResponse with data on success, or error details on failure.
    """
    if method not in ["GET", "POST"]:
        raise ValueError(f"Invalid method: {method}")
    if method == "GET" and json_payload:
        raise ValueError("JSON payload cannot be provided for GET requests")

    @backoff.on_exception(
        backoff.expo,
        (asyncio.TimeoutError, aiohttp.ClientError, _RetryableHTTPError),
        max_tries=max_retries + 1,
        base=base_delay,
        max_value=max_delay,
        giveup=_should_giveup,
        logger=logger,
        backoff_log_level=logging.DEBUG,
        giveup_log_level=logging.DEBUG,
    )
    async def _do_request() -> dict:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if method == "GET":
                request_context = session.get(url, params=params, headers=headers)
            else:
                request_context = session.post(url, params=params, json=json_payload, headers=headers)

            async with request_context as response:
                if _is_retryable_status(response.status):
                    raise _RetryableHTTPError(response.status, response.reason or "")

                response.raise_for_status()
                return await response.json()

    try:
        data = await _do_request()
        return APIResponse(data=data)
    except asyncio.TimeoutError:
        return APIResponse(error=f"Timeout after {timeout_seconds} seconds", timed_out=True)
    except _RetryableHTTPError as e:
        return APIResponse(error=f"HTTP error: {e.status} {e.message}")
    except aiohttp.ClientResponseError as e:
        return APIResponse(error=f"HTTP error: {e.status} {e.message}")
    except aiohttp.ClientError as e:
        return APIResponse(error=f"Connection error: {e}")

    # we should never reach this point, so raise an error if we do
    raise RuntimeError(f"Failed to make API request to {url}")


def get_openai_tool_definitions(tool: "Tool") -> dict[str, Any]:
    """Helper function to export tool definition in OpenAI function calling format.
    Note that we rely on parameters being correctly set in the tool class.

    This format is compatible with vLLM's tool calling and chat templates.
    See: https://docs.vllm.ai/en/latest/features/tool_calling/

    Args:
        tool: The tool instance to export definition for.

    Returns:
        Tool definition in OpenAI format.
    """
    return {
        "type": "function",
        "function": {"name": tool.call_name, "description": tool.description, "parameters": tool.parameters},
    }


def _coerce_bool(value: Any) -> bool:
    """Coerce a value to boolean, handling string 'true'/'false' correctly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError(f"Cannot coerce '{value}' to boolean")
    return bool(value)


# JSON schema type to Python coercion function
_COERCERS: dict[str, Callable[[Any], Any]] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": _coerce_bool,
    "array": list,
    "object": dict,
}


def coerce_args(parameters: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    """Coerce arguments to match the expected types from a JSON schema.

    Args:
        parameters: A JSON schema dict (with a "properties" key).
        args: The arguments to coerce.

    Returns:
        A new dict with coerced values.

    Raises:
        ValueError or TypeError if coercion fails.
    """
    properties = parameters.get("properties", {})
    coerced = dict(args)
    for name, value in args.items():
        if value is None or name not in properties:
            continue
        expected_type = properties[name].get("type")
        if expected_type not in _COERCERS:
            continue
        coerced[name] = _COERCERS[expected_type](value)
    return coerced


class Tool(RLEnvironment):
    """Base class for stateless tools (e.g. code execution, web search).

    Subclasses implement step() directly and return StepResult.
    Use coerce_args(self.parameters, call.args) to coerce model args to the expected types.
    """

    config_name: str
    """Name used to specify the tool in the CLI."""
    description: str
    """Default description used for e.g. system prompts."""
    call_name: str
    """Name used to identify the tool when function calling."""
    parameters: dict[str, Any]
    """JSON schema for tool parameters. Exposed to the model when calling the tool."""
    observation_role: str = "tool"
    """Role for observations/feedback in conversation."""

    # -- RLEnvironment defaults for stateless tools --

    async def reset(self, task_id: str | None = None, **kwargs) -> tuple[StepResult, list[dict]]:
        return StepResult(result=""), [get_openai_tool_definitions(self)]

    def state(self) -> State:
        return State()

    # -- Utilities for tool implementations --

    def get_observation_role(self) -> str:
        """Return the role to use for observations in conversation."""
        return self.observation_role

    def get_call_name(self) -> str:
        """Get the tool's call name (used when function calling)."""
        return self.call_name

    def get_description(self) -> str:
        """Get the tool's description."""
        return self.description

    def get_parameters(self) -> dict[str, Any]:
        """Get the tool's parameter schema."""
        return self.parameters

    def get_stop_strings(self) -> list[str]:
        """Get stop strings for this tool. Override in subclasses that define custom stop sequences."""
        return []

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [get_openai_tool_definitions(self)]
