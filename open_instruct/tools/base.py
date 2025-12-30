from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.entrypoints.openai.protocol import ChatCompletionRequest

if TYPE_CHECKING:
    from vllm.entrypoints.openai.tool_parsers import ToolParser as VllmNativeToolParser

logger = logging.getLogger(__name__)


# our only requirement is that tools must output a string
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
    # Instance-level tag name (set in __init__ if provided, else uses default)
    _tag_name: str | None = None

    @property
    def tool_function_name(self) -> str:
        """The tag/function name used for this tool instance.

        This is used both as the XML tag name (e.g., <search>...</search>)
        and as the key for looking up the tool.

        Can be overridden at instantiation via tag_name parameter to allow
        the same tool implementation to use different tags.
        """
        if self._tag_name is not None:
            return self._tag_name
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
                "name": self._tag_name,
                "description": self.tool_description,
                "parameters": self.tool_parameters,
            },
        }

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        pass


# parser base class
class ToolParser(ABC):
    @abstractmethod
    def get_tool_calls(self, text: str) -> list[ToolCall]:
        pass

    @abstractmethod
    def format_tool_calls(self, tool_output: str) -> str:
        pass

    @abstractmethod
    def stop_sequences(self) -> list[str]:
        pass


class VllmToolParser(ToolParser):
    """
    Wraps a vLLM tool parser to extract tool calls and format responses.

    See: https://docs.vllm.ai/en/latest/features/tool_calling/

    Usage:
        >>> from open_instruct.tools.vllm_parsers import create_vllm_parser
        >>> tools = [tool.get_openai_tool_definition() for tool in my_tools]
        >>> parser = create_vllm_parser("hermes", tokenizer, tool_definitions=tools)
        >>> tool_calls = parser.get_tool_calls(model_output)
        >>> formatted = parser.format_tool_calls(tool_result)
    """

    def __init__(
        self,
        tool_parser: VllmNativeToolParser,
        output_formatter: Callable[[str], str],
        stop_sequences: list[str] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
    ):
        self.tool_parser = tool_parser
        self.output_formatter = output_formatter
        self._stop_sequences = stop_sequences or []
        self._tool_definitions = tool_definitions

    def _make_request(self) -> Any:
        """
        Create a dummy ChatCompletionRequest for vLLM parsers.
        Usually these only need the list of tools.
        """
        return ChatCompletionRequest(model="dummy", messages=[], tools=self._tool_definitions)

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract tool calls from model output.

        Args:
            text: The model output text to parse.
        """
        request = self._make_request()
        result = self.tool_parser.extract_tool_calls(model_output=text, request=request)

        if not result.tools_called:
            return []

        tool_calls = []
        for call in result.tool_calls:
            try:
                args = json.loads(call.function.arguments)
                tool_calls.append(ToolCall(name=call.function.name, args=args))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"VllmToolParser: Failed to parse tool arguments: {e}\nArguments: {call.function.arguments!r}"
                )
                continue
        return tool_calls

    def format_tool_calls(self, tool_output: str) -> str:
        return self.output_formatter(tool_output)

    def stop_sequences(self) -> list[str]:
        return self._stop_sequences


class OpenInstructLegacyToolParser(ToolParser):
    """
    Parser that recreates the older open-instruct style,
    which also matches DR Tulu and Open-R1 styles.

    Tools are invoked via <tool_name>content</tool_name> tags.
    The content between tags is passed to the tool as a text argument.
    """

    def __init__(self, tool_list: list[Tool], output_wrap_name: str = "output"):
        self.tool_names = [tool.tool_function_name for tool in tool_list]
        self.output_wrap_name = output_wrap_name
        assert len(self.tool_names) == len(set(self.tool_names)), "Tool names must be unique"
        self.tool_stop_strings = [f"</{tool_name}>" for tool_name in self.tool_names]
        self.tool_start_strings = [f"<{tool_name}>" for tool_name in self.tool_names]
        # Build regex per tool to extract content
        self.tool_regexes = {
            tool_name: re.compile(re.escape(f"<{tool_name}>") + r"(.*?)" + re.escape(f"</{tool_name}>"), re.DOTALL)
            for tool_name in self.tool_names
        }

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        # Check each tool's regex
        for tool_name, tool_regex in self.tool_regexes.items():
            match = tool_regex.search(text)
            if match:
                # The content between tags is passed as the first argument
                tool_content = match.group(1)
                tool_calls.append(ToolCall(name=tool_name, args={"text": tool_content}))
        return tool_calls

    def format_tool_calls(self, tool_output: str) -> str:
        return f"<{self.output_wrap_name}>\n{tool_output}\n</{self.output_wrap_name}>\n"

    def stop_sequences(self) -> list[str]:
        return self.tool_stop_strings


class DRTuluToolParser(ToolParser):
    """
    Tool parser for DR Tulu / MCP style tools.

    These tools handle their own internal parsing and routing, so this parser
    just detects if ANY tool call pattern is present (by checking for stop strings)
    and routes the full text to the tool for execution.
    """

    def __init__(self, tool_list: list[Tool]):
        self.tools = tool_list

        # Collect stop strings from tools (e.g., "</tool>")
        self._stop_strings: list[str] = []
        for tool in tool_list:
            if hasattr(tool, "get_stop_strings"):
                self._stop_strings.extend(tool.get_stop_strings())

        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_stops: list[str] = []
        for s in self._stop_strings:
            if s not in seen:
                seen.add(s)
                unique_stops.append(s)
        self._stop_strings = unique_stops

        logger.info(f"DRTuluToolParser: Initialized with {len(tool_list)} tools, stop_strings={self._stop_strings}")

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        """Detect tool calls by checking for stop string patterns.

        Since MCP tools handle their own internal routing, we just need to detect
        if ANY tool was called and route the full text to that tool.
        """
        # Check if any stop string appears in text (indicates a tool call)
        for stop_str in self._stop_strings:
            if stop_str in text:
                # Route to the first tool (MCP tool handles internal routing)
                for tool in self.tools:
                    return [ToolCall(name=tool.tool_function_name, args={"text": text})]
        return []

    def format_tool_calls(self, tool_output: str) -> str:
        return "\n" + tool_output + "\n\n"

    def stop_sequences(self) -> list[str]:
        return self._stop_strings
