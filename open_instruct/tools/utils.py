from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar


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


# base class. All tools must have call method and output ToolOutput.
# they must also return the argument list and their own name.
class Tool(ABC):
    # Default tool function name (subclasses set this as class attribute)
    _default_tool_function_name: str = "tool"
    # Default description (subclasses can override as class attribute)
    _default_tool_description: str = ""
    # Default parameters schema (subclasses can override as class attribute)
    _default_tool_parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    # Instance-level override name (set in __init__ if provided, else uses default)
    _override_name: str | None = None

    @property
    def tool_function_name(self) -> str:
        """The tag/function name used for this tool instance.

        This is used both as the XML tag name (e.g., <search>...</search>)
        and as the key for looking up the tool.

        Can be overridden at instantiation via override_name parameter to allow
        the same tool implementation to use different tags.
        """
        if self._override_name is not None:
            return self._override_name
        return self._default_tool_function_name

    @property
    def tool_description(self) -> str:
        """Description of what the tool does. Used for function calling."""
        return self._default_tool_description

    @property
    def tool_parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters. Used for function calling."""
        return self._default_tool_parameters

    def get_openai_tool_definition(self) -> dict[str, Any]:
        """Export tool definition in OpenAI function calling format.

        This format is compatible with vLLM's tool calling and chat templates.
        See: https://docs.vllm.ai/en/latest/features/tool_calling/

        Returns:
            Dict in OpenAI tool format with type, function name, description, and parameters.
        """
        return {
            "type": "function",
            "function": {
                "name": self.tool_function_name,
                "description": self.tool_description,
                "parameters": self.tool_parameters,
            },
        }

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        pass


@dataclass
class BaseToolConfig:
    """Base configuration class for individual tools.

    Subclasses should set `tool_class` to their corresponding Tool class.
    The generic `build()` method passes all dataclass fields as kwargs to the tool constructor.

    Override `build()` only if:
    - Validation is needed before construction
    - Field names don't match constructor params
    - Custom initialization logic is required

    Example:
        @dataclass
        class MyToolConfig(BaseToolConfig):
            tool_class: ClassVar[type[Tool]] = MyTool

            param1: str = "default"
            param2: int = 10

        config = MyToolConfig(param1="custom")
        tool = config.build()  # Equivalent to MyTool(param1="custom", param2=10)
    """

    if TYPE_CHECKING:
        # ClassVars are not included in asdict(), so they won't be passed to build()
        tool_class: ClassVar[type[Tool]]

    override_name: str | None = None
    """Override the default tag/function name for this tool instance."""

    def build(self) -> Tool:
        """Build the tool instance from this config.

        Passes all dataclass instance fields as kwargs to tool_class.
        Override this method for tools with validation or non-standard initialization.
        """
        return self.tool_class(**asdict(self))
