import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm import ToolParser as VllmNativeToolParser


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
    tool_function_name: str

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
    """

    def __init__(self, tool_list: list[Tool], output_wrap_name: str = "output"):
        self.tool_names = [tool.tool_function_name for tool in tool_list]
        assert len(self.tool_names) == len(set(self.tool_names)), "Tool names must be unique"
        self.tool_stop_strings = [f"</{tool_name}>" for tool_name in self.tool_names]
        self.tool_start_strings = [f"<{tool_name}>" for tool_name in self.tool_names]
        self.tool_regexes = [
            re.escape(tool_start_string) + r"(.*?)" + re.escape(tool_end_string)
            for tool_start_string, tool_end_string in zip(self.tool_start_strings, self.tool_stop_strings)
        ]

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        # search the tools
        for tool_regex in self.tool_regexes:
            match = re.search(tool_regex, text, re.DOTALL)
            if match:
                tool_name = match.group(1)
                tool_args = match.group(2)
                tool_calls.append(ToolCall(name=tool_name, args=tool_args))
        return tool_calls

    def format_tool_calls(self, tool_output: str) -> str:
        return f"<{self.output_wrap_name}>\n{tool_output}\n</{self.output_wrap_name}>"

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
