import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import ray

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


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
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the tool, must be implemented by subclasses."""
        raise NotImplementedError("__call__ must be implemented by subclasses.")


@dataclass
class BaseToolConfig:
    """Base configuration class for individual tools.

    Subclasses must also define a config, which is used also used to instantiate the tool itself.
    """

    tool_class: ClassVar[type[Tool]]
    """Related tool class for this config."""

    def _get_init_kwargs(self) -> dict[str, Any]:
        """Get kwargs for initializing the tool.

        Passes all dataclass instance fields as kwargs.
        Override this method for tools with validation or non-standard initialization.

        Returns:
            Dictionary of kwargs to pass to tool_class.__init__
        """
        return asdict(self)

    def build(self, call_name: str | None = None) -> Tool:
        """Build the tool instance from this config.

        Args:
            call_name: Name used to identify in function calls. If not provided, uses tool's config name.

        Returns:
            A Tool instance.
        """
        args = self._get_init_kwargs()
        if call_name:
            args["call_name"] = call_name
        else:
            # Use tool class's default call name
            args["call_name"] = self.tool_class.config_name
        return self.tool_class(**args)

    def build_remote(self, call_name: str | None = None, max_concurrency: int = 512) -> ray.actor.ActorHandle:
        """Build the tool as a Ray remote actor.

        Allows for passing to vllm actors without needing to serialize tool itself,
        and to centralize control over tool concurrency.

        Args:
            call_name: Name used to identify in function calls. If not provided, uses tool's config name.
            max_concurrency: Maximum number of concurrent calls the actor can handle.

        Returns:
            A Ray actor handle for the Tool.
        """
        args = self._get_init_kwargs()
        if call_name:
            args["call_name"] = call_name
        else:
            # Use tool class's default call name
            args["call_name"] = self.tool_class.config_name
        return ray.remote(self.tool_class).options(max_concurrency=max_concurrency).remote(**args)
