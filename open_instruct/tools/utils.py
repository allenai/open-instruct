import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import aiohttp

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


@dataclass
class ToolsConfig:
    """Configuration for tools used during generation."""

    tools: list[str] | None = None
    """List of tool names to enable (e.g., ["python", "search"])."""

    tool_call_names: list[str] | None = None
    """Override names used in tool calls (e.g., '<name>...</name>').
    Must match length of tools if set. Defaults to tools if not specified."""

    tool_configs: list[str] = field(default_factory=list)
    """JSON strings for configuring each tool. Must match length of tools. Use '{}' for defaults."""

    tool_parser_type: str = "legacy"
    """Type of tool parser to use: 'legacy' is the only option for now."""

    max_tool_calls: int = 5
    """Maximum number of tool calls allowed per generation."""

    only_reward_good_outputs: bool = False
    """Only apply rewards to outputs from tools that didn't error."""

    _parsed_tool_configs: list[dict[str, Any]] = field(default_factory=list, init=False)
    """Parsed tool configurations as dictionaries. Populated from tool_configs during __post_init__."""

    def __post_init__(self):
        self.max_tool_calls = int(self.max_tool_calls)

        if self.tools:
            # Set default tool_call_names if not provided
            if not self.tool_call_names:
                self.tool_call_names = self.tools
            elif len(self.tool_call_names) != len(self.tools):
                raise ValueError(
                    f"tool_call_names must have same length as tools. "
                    f"Got {len(self.tool_call_names)} names for {len(self.tools)} tools."
                )

            # Set default tool_configs if not provided
            if not self.tool_configs:
                self.tool_configs = ["{}"] * len(self.tools)
            elif len(self.tool_configs) != len(self.tools):
                raise ValueError(
                    f"tool_configs must have same length as tools. "
                    f"Got {len(self.tool_configs)} configs for {len(self.tools)} tools."
                )

            # Parse all tool_configs into dicts and store in _parsed_tool_configs
            # using a simple loop to make the error message more informative
            self._parsed_tool_configs = []
            for i, (tool_name, config) in enumerate(zip(self.tools, self.tool_configs)):
                try:
                    self._parsed_tool_configs.append(json.loads(config))
                except Exception as e:
                    raise ValueError(f"Invalid tool_config for tool {tool_name} at index {i}: {e}") from e

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


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class APIResponse:
    """Response from an async API request."""

    data: dict | None = None
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
