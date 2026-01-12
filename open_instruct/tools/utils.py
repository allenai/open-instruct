import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

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

    def get_tool_names(self) -> list[str]:
        """Get the tool names this tool exposes.

        Returns:
            List of tool names.
        """
        return [self.call_name]

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

    def build(self) -> Tool:
        """Build the tool instance from this config.

        Passes all dataclass instance fields as kwargs to tool_class.
        Override this method for tools with validation or non-standard initialization.
        """
        return self.tool_class(**asdict(self))
