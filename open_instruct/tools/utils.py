import inspect
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, get_type_hints

from pydantic import create_model

from open_instruct.logger_utils import setup_logger

logger = setup_logger(__name__)


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


def infer_tool_parameters(call_method: Any) -> dict[str, Any]:
    """Infer JSON Schema parameters from a __call__ method's signature using Pydantic.

    Use Annotated types with Field for descriptions:
        query: Annotated[str, Field(description="The search query")]

    Args:
        call_method: The __call__ method to inspect.

    Returns:
        JSON Schema dict with 'type', 'properties', and 'required' fields.
    """
    sig = inspect.signature(call_method)

    try:
        hints = get_type_hints(call_method, include_extras=True)
    except Exception as e:
        logger.warning(
            f"Could not get type hints for {getattr(call_method, '__name__', 'unknown_method')}: {e}. Proceeding without them."
        )
        hints = {}

    # Build field definitions for Pydantic model
    field_definitions: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        param_type = hints.get(param_name, str)
        if param.default is inspect.Parameter.empty:
            field_definitions[param_name] = (param_type, ...)
        else:
            field_definitions[param_name] = (param_type, param.default)

    if not field_definitions:
        return {"type": "object", "properties": {}, "required": []}

    dynamic_model = create_model("ToolParams", **field_definitions)
    schema = dynamic_model.model_json_schema()

    # some cleaning up
    schema.pop("title", None)
    if "$defs" in schema and not schema["$defs"]:
        schema.pop("$defs", None)

    return schema


class Tool(ABC):
    default_tool_name: str
    """Default name used when calling this tool."""
    default_description: str
    """Default description used for e.g. system prompts."""
    override_name: str | None = None
    """Override name from default tool name. Useful if you want to swap different tools but use the same call name."""
    _default_tool_parameters: dict[str, Any] | None = None
    """If not set, auto-inferred from __call__ signature using pydantic."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "default_tool_name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define 'default_tool_name' class attribute")
        if "default_description" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define 'default_description' class attribute")

    @property
    def tool_function_name(self) -> str:
        if self.override_name is not None:
            return self.override_name
        return self.default_tool_name

    @property
    def tool_description(self) -> str:
        return self.default_description

    @property
    def tool_parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters. Used for function calling.

        If `_default_tool_parameters` is set explicitly on the subclass, returns that.
        Otherwise, infers the schema from the `__call__` method's type hints and caches it.
        """
        cls = type(self)
        if cls._default_tool_parameters is not None:
            return cls._default_tool_parameters

        cls._default_tool_parameters = infer_tool_parameters(self.__call__)
        return cls._default_tool_parameters

    def get_openai_tool_definitions(self) -> list[dict[str, Any]]:
        """Export tool definitions in OpenAI function calling format.

        This format is compatible with vLLM's tool calling and chat templates.
        See: https://docs.vllm.ai/en/latest/features/tool_calling/

        For most tools, this returns a single-item list. In some cases, we may want a tool to expose multiple functions, in which case we return the list of all functions attached to the tool.

        Returns:
            List of tool definitions in OpenAI format.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": self.tool_function_name,
                    "description": self.tool_description,
                    "parameters": self.tool_parameters,
                },
            }
        ]

    def get_tool_names(self) -> list[str]:
        """Get the tool names this tool exposes.

        Returns:
            List of tool names.
        """
        return [self.tool_function_name]

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Execute the tool, must be implemented by subclasses."""
        pass


@dataclass
class BaseToolConfig:
    """Base configuration class for individual tools.

    Subclasses must also define a config, which is used also used to instantiate the tool itself.
    """

    tool_class: ClassVar[type[Tool]]
    """Related tool class for this config."""
    override_name: str | None = None
    """Override the default tag/function name for this tool instance."""

    def build(self) -> Tool:
        """Build the tool instance from this config.

        Passes all dataclass instance fields as kwargs to tool_class.
        Override this method for tools with validation or non-standard initialization.
        """
        return self.tool_class(**asdict(self))
