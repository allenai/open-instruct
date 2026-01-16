import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar

import aiohttp

from open_instruct import logger_utils
from open_instruct.data_types import ToolCallStats

logger = logger_utils.setup_logger(__name__)


@dataclass
class ParsedToolConfig:
    """A parsed tool configuration combining name, call name, and config."""

    name: str
    """The tool name (e.g., "python", "search")."""

    call_name: str
    """The name used in tool calls (e.g., "code", "web_search")."""

    config: dict[str, Any]
    """The parsed configuration dictionary for this tool."""


@dataclass
class ToolsConfig:
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

    max_tool_calls: int = 5
    """Maximum number of tool calls allowed per generation."""

    only_reward_good_outputs: bool = False
    """Only apply rewards to outputs from tools that didn't error."""

    pass_tools_to_chat_template: bool = True
    """Pass tool definitions to the chat template. Set to False if using a custom system prompt."""

    _parsed_tools: list[ParsedToolConfig] = field(default_factory=list, init=False)
    """Parsed tool configurations. Populated during __post_init__."""

    def __post_init__(self):
        self.max_tool_calls = int(self.max_tool_calls)

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
            self._parsed_tools.append(ParsedToolConfig(name=tool_name, call_name=call_name, config=config))

    @property
    def enabled(self) -> bool:
        """Return True if any tools are configured."""
        return bool(self.tools)


@dataclass
class ToolOutput:
    output: str
    called: bool
    error: str
    timeout: bool
    runtime: float


class ToolStatistics:
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
        self._excess_calls: defaultdict[str, int] = defaultdict(int)

    def add_rollout(
        self, tool_call_stats: list[ToolCallStats], excess_tool_calls: dict[str, int] | None = None
    ) -> None:
        """Add statistics from a single rollout.

        Args:
            tool_call_stats: List of ToolCallStats from a single rollout.
            excess_tool_calls: Dict mapping tool name to count of calls that exceeded the limit.
        """
        self.num_rollouts += 1
        for s in tool_call_stats:
            self.tool_names.add(s.tool_name)
            self._counts[s.tool_name] += 1
            self._failures[s.tool_name] += not s.success
            self._runtimes[s.tool_name] += s.runtime

        if excess_tool_calls:
            for tool_name, count in excess_tool_calls.items():
                self.tool_names.add(tool_name)
                self._excess_calls[tool_name] += count

    def compute_metrics(self) -> dict[str, float]:
        """Compute per-tool and aggregate metrics.

        Returns:
            Dictionary with metrics for each tool and aggregate totals:
            - tools/{name}/avg_calls_per_rollout
            - tools/{name}/failure_rate
            - tools/{name}/avg_runtime
            - tools/{name}/avg_excess_calls_per_rollout
            - tools/aggregate/avg_calls_per_rollout
            - tools/aggregate/failure_rate
            - tools/aggregate/avg_runtime
            - tools/aggregate/avg_excess_calls_per_rollout
        """
        if not self.num_rollouts or not self.tool_names:
            return {}

        metrics: dict[str, float] = {}
        total_calls = 0
        total_failures = 0
        total_runtime = 0.0
        total_excess = 0

        for name in self.tool_names:
            calls, failures, runtime = self._counts[name], self._failures[name], self._runtimes[name]
            excess = self._excess_calls[name]
            metrics[f"tools/{name}/avg_calls_per_rollout"] = calls / self.num_rollouts
            metrics[f"tools/{name}/failure_rate"] = failures / calls if calls else 0.0
            metrics[f"tools/{name}/avg_runtime"] = runtime / calls if calls else 0.0
            metrics[f"tools/{name}/avg_excess_calls_per_rollout"] = excess / self.num_rollouts
            total_calls += calls
            total_failures += failures
            total_runtime += runtime
            total_excess += excess

        metrics["tools/aggregate/avg_calls_per_rollout"] = total_calls / self.num_rollouts
        metrics["tools/aggregate/failure_rate"] = total_failures / total_calls if total_calls else 0.0
        metrics["tools/aggregate/avg_runtime"] = total_runtime / total_calls if total_calls else 0.0
        metrics["tools/aggregate/avg_excess_calls_per_rollout"] = total_excess / self.num_rollouts

        return metrics


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class APIResponse:
    """Response from an async API request."""

    data: dict = field(default_factory=dict)
    error: str = ""
    timed_out: bool = False


async def make_api_request(
    url: str,
    timeout_seconds: int,
    headers: dict | None = None,
    json_payload: dict | None = None,
    params: dict | None = None,
    method: str = "POST",
) -> APIResponse:
    """Make an async HTTP request with standard error handling.

    Args:
        url: The API endpoint URL.
        timeout_seconds: Request timeout in seconds.
        headers: Optional HTTP headers.
        json_payload: JSON data to send in the request body.
        params: Query parameters.
        method: HTTP method ("GET" or "POST"). Defaults to "POST".

    Returns:
        APIResponse with data on success, or error details on failure.
    """
    if method not in ["GET", "POST"]:
        raise ValueError(f"Invalid method: {method}")
    if method == "GET" and json_payload:
        raise ValueError("JSON payload cannot be provided for GET requests")
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if method == "GET":
                request_context = session.get(url, params=params, headers=headers)
            else:
                request_context = session.post(url, params=params, json=json_payload, headers=headers)

            async with request_context as response:
                response.raise_for_status()
                data = await response.json()
                return APIResponse(data=data)
    except asyncio.TimeoutError:
        return APIResponse(error=f"Timeout after {timeout_seconds} seconds", timed_out=True)
    except aiohttp.ClientResponseError as e:
        return APIResponse(error=f"HTTP error: {e.status} {e.message}")
    except aiohttp.ClientError as e:
        return APIResponse(error=f"Connection error: {e}")


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


class Tool(ABC):
    config_name: str
    """Name used to specify the tool in the CLI."""
    description: str
    """Default description used for e.g. system prompts."""
    call_name: str
    """Name used to identify the tool when function calling."""
    parameters: dict[str, Any]
    """JSON schema for tool parameters. Exposed to the model when calling the tool."""

    def __init__(self, config_name: str, description: str, call_name: str, parameters: dict[str, Any]) -> None:
        self.config_name = config_name
        self.description = description
        self.call_name = call_name
        self.parameters = parameters
        # validate parameters are valid JSON
        try:
            json.loads(json.dumps(parameters))
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def get_call_name(self) -> str:
        """Get the tool's call name (used when function calling)."""
        return self.call_name

    def get_description(self) -> str:
        """Get the tool's description."""
        return self.description

    def get_parameters(self) -> dict[str, Any]:
        """Get the tool's parameter schema."""
        return self.parameters

    def get_openai_tool_definitions(self) -> dict[str, Any]:
        """Get tool definition in OpenAI format."""
        return get_openai_tool_definitions(self)

    @abstractmethod
    async def execute(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the tool, must be implemented by subclasses."""
        raise NotImplementedError("execute must be implemented by subclasses.")

    async def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Alias for execute, useful for inference scripts."""
        return await self.execute(*args, **kwargs)


@dataclass
class BaseToolConfig:
    """Base configuration class for individual tools.

    Subclasses must also define a config, which is used also used to instantiate the tool itself.
    """

    tool_class: ClassVar[type[Tool]]
    """Related tool class for this config."""
