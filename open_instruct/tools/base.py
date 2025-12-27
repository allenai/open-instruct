from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.entrypoints.openai.tool_parsers import ToolParser as VllmNativeToolParser


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
    tool_args: dict[str, Any]
    # Default tool function name (subclasses set this as class attribute)
    _default_tool_function_name: str = "tool"
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
    Parser that wraps around vllm tool parsers.
    """

    def __init__(self, tool_parser: VllmNativeToolParser, output_formatter: Callable[[str], str]):
        # output formatter is a function that wraps around the tool output
        # has to be unique to the chat template used.
        self.output_formatter = output_formatter
        self.tool_parser = tool_parser

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        result = self.tool_parser.extract_tool_calls(model_output=text)
        if not result.tools_called:
            return tool_calls
        for call in result.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            tool_calls.append(ToolCall(name=name, args=args))
        return tool_calls

    def format_tool_calls(self, tool_output: str) -> str:
        return self.output_formatter(tool_output)

    def stop_sequences(self) -> list[str]:
        """vllm parsers use native stop sequences."""
        return []


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
    Tool parser that recreates the DR Tulu style.
    Basically a wrapper around the mcp tools, that themselves handle parsing/calls/etc.
    """

    def __init__(self, mcp_tool_list: list[Tool]):
        self.mcp_tools = mcp_tool_list
        self._stop_strings: list[str] = []
        # Collect stop strings from all MCP tools
        for mcp_tool in self.mcp_tools:
            if hasattr(mcp_tool, "tool_parser") and hasattr(mcp_tool.tool_parser, "stop_sequences"):
                self._stop_strings.extend(mcp_tool.tool_parser.stop_sequences)

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for mcp_tool in self.mcp_tools:
            if hasattr(mcp_tool, "tool_parser") and mcp_tool.tool_parser.has_calls(text, mcp_tool.name):
                # DR agent tools parse text themselves
                tool_calls.append(ToolCall(name=mcp_tool.name, args={"text": text}))
        return tool_calls

    def format_tool_calls(self, tool_output: str) -> str:
        # DR Tulu uses the MCP tool's own formatting
        return tool_output

    def stop_sequences(self) -> list[str]:
        return self._stop_strings
