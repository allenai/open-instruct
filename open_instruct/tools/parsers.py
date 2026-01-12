"""
Tool parsers for extracting tool calls from model output.

Includes:
- Base ToolParser ABC
- OpenInstructLegacyToolParser: <tool_name>content</tool_name> style
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

from open_instruct.tools.utils import Tool, ToolCall

logger = logging.getLogger(__name__)


# parser base class
class ToolParser(ABC):
    @abstractmethod
    def get_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract tool calls from model outputs."""
        pass

    def format_tool_outputs(self, tool_outputs: list[str]) -> str:
        """Format multiple tool outputs with any necessary prefix/postfix.

        By default, just concatenates individually formatted outputs.
        Subclasses can override to add prefix/postfix for continuation prompts.
        """
        return "".join(self.format_tool_calls(output) for output in tool_outputs)

    @abstractmethod
    def stop_sequences(self) -> list[str]:
        """Get stop strings this parser relies on."""
        pass


class OpenInstructLegacyToolParser(ToolParser):
    """
    Parser that recreates the older open-instruct style,
    which also matches DR Tulu and Open-R1 styles.

    Tools are invoked via <tool_name>content</tool_name> tags.
    The content between tags is passed to the tool's first required parameter.
    Only works for tools that take a single string parameter.
    """

    def __init__(self, tool_list: list[Tool], output_wrap_name: str = "output"):
        self.tool_names = [tool.tool_function_name for tool in tool_list]
        self.output_wrap_name = output_wrap_name
        assert len(self.tool_names) == len(set(self.tool_names)), "Tool names must be unique"
        self.tool_stop_strings = [f"</{tool_name}>" for tool_name in self.tool_names]
        self.tool_start_strings = [f"<{tool_name}>" for tool_name in self.tool_names]

        self.tool_param_names: dict[str, str] = {}
        for tool in tool_list:
            params = tool.tool_parameters
            required = params.get("required", [])
            if required:
                self.tool_param_names[tool.tool_function_name] = required[0]
            else:
                properties = params.get("properties", {})
                if properties:
                    self.tool_param_names[tool.tool_function_name] = next(iter(properties))
                else:
                    self.tool_param_names[tool.tool_function_name] = "text"

        self.tool_regexes = {
            tool_name: re.compile(re.escape(f"<{tool_name}>") + r"(.*?)" + re.escape(f"</{tool_name}>"), re.DOTALL)
            for tool_name in self.tool_names
        }

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for tool_name, tool_regex in self.tool_regexes.items():
            match = tool_regex.search(text)
            if match:
                tool_content = match.group(1)
                param_name = self.tool_param_names.get(tool_name, "text")
                tool_calls.append(ToolCall(name=tool_name, args={param_name: tool_content}))
        return tool_calls

    def _format_tool_output(self, tool_output: str) -> str:
        return f"<{self.output_wrap_name}>\n{tool_output}\n</{self.output_wrap_name}>\n"

    def format_tool_outputs(self, tool_outputs: list[str]) -> str:
        return "\n".join(self._format_tool_output(tool_output) for tool_output in tool_outputs)

    def stop_sequences(self) -> list[str]:
        return self.tool_stop_strings


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return ["legacy"]
