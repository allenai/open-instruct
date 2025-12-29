from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

    Works with both DrAgentMCPTool instances and ToolProxy wrappers around them.
    """

    def __init__(self, mcp_tool_list: list[Tool]):
        # Store the tool wrappers (either DrAgentMCPTool or ToolProxy)
        self.tool_wrappers = mcp_tool_list
        logger.info(f"DRTuluToolParser: Initializing with {len(mcp_tool_list)} tool wrappers")

        # Get MCP tool names from each wrapper
        self.mcp_tool_names: list[str] = []
        for tool in mcp_tool_list:
            if hasattr(tool, "get_mcp_tool_names"):
                # It's a ToolProxy or DrAgentMCPTool with this method
                names = tool.get_mcp_tool_names()
                logger.info(f"DRTuluToolParser: Got MCP tool names from proxy: {names}")
                self.mcp_tool_names.extend(names)
            elif hasattr(tool, "mcp_tools"):
                # Direct DrAgentMCPTool access (legacy)
                names = [t.name for t in tool.mcp_tools]
                logger.info(f"DRTuluToolParser: Got MCP tool names directly: {names}")
                self.mcp_tool_names.extend(names)

        logger.info(f"DRTuluToolParser: All MCP tool names: {self.mcp_tool_names}")

        self._stop_strings: list[str] = []
        # Collect stop strings from all tools
        for tool in mcp_tool_list:
            if hasattr(tool, "get_stop_strings"):
                stop_strs = tool.get_stop_strings()
                logger.info(f"DRTuluToolParser: Got stop strings: {stop_strs}")
                self._stop_strings.extend(stop_strs)

        logger.info(f"DRTuluToolParser: All stop strings: {self._stop_strings}")

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        text_preview = text[:200] if len(text) > 200 else text
        logger.debug(f"DRTuluToolParser.get_tool_calls: Checking text: {text_preview!r}...")
        logger.debug(f"DRTuluToolParser.get_tool_calls: mcp_tool_names={self.mcp_tool_names}")

        for tool in self.tool_wrappers:
            for mcp_tool_name in self.mcp_tool_names:
                has_calls_attr = hasattr(tool, "has_calls")
                if has_calls_attr:
                    result = tool.has_calls(text, mcp_tool_name)
                    logger.debug(f"DRTuluToolParser: has_calls({mcp_tool_name}) = {result}")
                    if result:
                        # Use the wrapper's tool_function_name for the dict lookup,
                        # not the individual MCP tool name
                        tool_calls.append(ToolCall(name=tool.tool_function_name, args={"text": text}))
                        logger.info(f"DRTuluToolParser: Found tool call for {tool.tool_function_name}")
                        break  # Only one call per wrapper needed

        logger.debug(f"DRTuluToolParser.get_tool_calls: Returning {len(tool_calls)} tool calls")
        return tool_calls

    def format_tool_calls(self, tool_output: str) -> str:
        # DR Tulu uses the MCP tool's own formatting
        return "\n" + tool_output + "\n\n"

    def stop_sequences(self) -> list[str]:
        return self._stop_strings
