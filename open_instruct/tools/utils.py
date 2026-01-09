from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, get_type_hints

from pydantic import create_model


# =============================================================================
# Parameter Inference
# =============================================================================


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

    # Get type hints with Annotated metadata preserved
    try:
        hints = get_type_hints(call_method, include_extras=True)
    except Exception:
        hints = {}

    # Build field definitions for Pydantic model
    field_definitions: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        # Skip 'self' and *args/**kwargs
        if param_name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Get type annotation (default to str if not provided)
        param_type = hints.get(param_name, str)

        # Handle default values
        if param.default is inspect.Parameter.empty:
            # Required parameter: (type, ...)
            field_definitions[param_name] = (param_type, ...)
        else:
            # Optional parameter with default: (type, default)
            field_definitions[param_name] = (param_type, param.default)

    # Create a dynamic Pydantic model and get its schema
    if not field_definitions:
        return {"type": "object", "properties": {}, "required": []}

    dynamic_model = create_model("ToolParams", **field_definitions)
    schema = dynamic_model.model_json_schema()

    # Clean up Pydantic's schema output (remove title, $defs if not needed)
    schema.pop("title", None)
    if "$defs" in schema and not schema["$defs"]:
        schema.pop("$defs", None)

    return schema


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
    # Default parameters schema (subclasses can override as class attribute, or leave as None to auto-infer)
    _default_tool_parameters: dict[str, Any] | None = None
    # Instance-level override name (set in __init__ if provided, else uses default)
    _override_name: str | None = None
    # Cache for inferred parameters (class-level)
    _inferred_parameters: ClassVar[dict[str, Any] | None] = None

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
        """JSON Schema for tool parameters. Used for function calling.

        If `_default_tool_parameters` is set explicitly, returns that.
        Otherwise, infers the schema from the `__call__` method's type hints.
        """
        # Use explicit schema if provided
        if self._default_tool_parameters is not None:
            return self._default_tool_parameters

        # Check class-level cache
        cls = type(self)
        if cls._inferred_parameters is not None:
            return cls._inferred_parameters

        # Infer from __call__ signature
        cls._inferred_parameters = infer_tool_parameters(self.__call__)
        return cls._inferred_parameters

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

    def get_openai_tool_definitions(self) -> list[dict[str, Any]]:
        """Export tool definitions in OpenAI function calling format.

        For most tools, this returns a single-item list. Tools that expose
        multiple functions (like GenericMCPTool) can override this to return
        multiple definitions.

        Returns:
            List of tool definitions in OpenAI format.
        """
        return [self.get_openai_tool_definition()]

    def get_tool_names(self) -> list[str]:
        """Get the tool names this tool exposes.

        For most tools, returns a single-item list with the tool's function name.
        Tools that expose multiple functions can override this to return all names.

        Returns:
            List of tool names.
        """
        return [self.tool_function_name]

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
