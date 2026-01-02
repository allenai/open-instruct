"""
Tool parsers for extracting tool calls from model output.

Includes:
- Base ToolParser ABC
- VllmToolParser: Wraps vLLM's native tool parsers
- OpenInstructLegacyToolParser: <tool_name>content</tool_name> style
- DRTuluToolParser: MCP-style tools

For vLLM parsers, use create_vllm_parser() factory function.
See: https://docs.vllm.ai/en/latest/features/tool_calling/
"""

from __future__ import annotations

import importlib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers import ToolParser as VllmNativeToolParser

from open_instruct.tools.utils import Tool, ToolCall

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        tool_parser: VllmNativeToolParser,
        output_formatter: Callable[[str], str],
        stop_sequences: list[str] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        output_postfix: str = "",
        output_prefix: str = "",
    ):
        self.tool_parser = tool_parser
        self.output_formatter = output_formatter
        self._stop_sequences = stop_sequences or []
        self._tool_definitions = tool_definitions
        self.output_postfix = output_postfix
        self.output_prefix = output_prefix

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

    def format_tool_outputs(self, tool_outputs: list[str]) -> str:
        """Format multiple tool outputs with prefix and postfix.

        Args:
            tool_outputs: List of tool output strings to format.

        Returns:
            Formatted string with prefix, all tool outputs, and postfix.
        """
        formatted_parts = [self.output_formatter(output) for output in tool_outputs]
        return self.output_prefix + "".join(formatted_parts) + self.output_postfix

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


# =============================================================================
# vLLM Parser Registry & Factory
# =============================================================================


@dataclass
class VllmParserConfig:
    """Configuration for a vLLM tool parser."""

    import_path: str  # e.g., "vllm.entrypoints.openai.tool_parsers.hermes_tool_parser:Hermes2ProToolParser"
    output_template: str  # Template for formatting each tool output, uses {} as placeholder
    stop_sequences: list[str]  # Stop sequences to use for this parser
    output_postfix: str  # Postfix to add after all tool outputs (includes generation prompt)
    output_prefix: str = ""  # Prefix to add before all tool outputs (for grouped tool responses)


VLLM_PARSERS: dict[str, VllmParserConfig] = {
    # Hermes-style (also works for Qwen2.5/3)
    "hermes": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.hermes_tool_parser:Hermes2ProToolParser",
        output_template="<|im_start|>tool\n<tool_response>\n{}\n</tool_response>\n<|im_end|>\n",
        stop_sequences=["</tool_call>"],
        output_postfix="<|im_start|>assistant\n",
    ),
    # Llama 3.x JSON style
    "llama3_json": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.llama_tool_parser:Llama3JsonToolParser",
        output_template="<|start_header_id|>ipython<|end_header_id|>\n\n{}<|eot_id|>",
        stop_sequences=["<|eom_id|>"],
        output_postfix="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    # Qwen3 Coder
    "qwen3_coder": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.qwen3coder_tool_parser:Qwen3CoderToolParser",
        output_template="<tool_response>\n{}\n</tool_response>\n",
        stop_sequences=["</tool_call>"],
        output_postfix="<|im_end|>\n<|im_start|>assistant\n",
        output_prefix="<|im_start|>user\n",
    ),
}


def _import_parser_class(import_path: str) -> type:
    """Import a parser class from module:ClassName path."""
    module_name, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_available_vllm_parsers() -> list[str]:
    """Return list of available vLLM parser names."""
    return list(VLLM_PARSERS.keys())


def get_vllm_parser_mapping() -> dict[str, str]:
    """Get mapping from CLI parser names (vllm_X) to internal names (X).

    To add a new vLLM parser, just add it to VLLM_PARSERS above.
    The CLI argument will automatically be available as --tool_parser vllm_{name}.
    """
    return {f"vllm_{name}": name for name in VLLM_PARSERS}


def create_vllm_parser(
    parser_name: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | Any,
    output_template: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> VllmToolParser:
    """Create a VllmToolParser by name.

    Args:
        parser_name: Name of the parser (e.g., "hermes", "llama3_json").
        tokenizer: The tokenizer for the model.
        output_template: Optional custom output template. Uses {} for the tool output.
                        If None, uses the default for this parser.
        tool_definitions: Optional list of tool definitions in OpenAI format.
                         These are stored and used by default in get_tool_calls().

    Returns:
        VllmToolParser configured for the specified model family.

    Example:
        >>> tools = [tool.get_openai_tool_definition() for tool in my_tools]
        >>> parser = create_vllm_parser("hermes", tokenizer, tool_definitions=tools)
    """
    if parser_name not in VLLM_PARSERS:
        available = get_available_vllm_parsers()
        raise ValueError(f"Unknown parser: {parser_name}. Available: {available}")

    config = VLLM_PARSERS[parser_name]
    template = output_template or config.output_template

    parser_cls = _import_parser_class(config.import_path)
    native_parser = parser_cls(tokenizer)

    return VllmToolParser(
        tool_parser=native_parser,
        output_formatter=lambda x, t=template: t.format(x),
        stop_sequences=[],  # for vLLM parser, we allow the models to decide when to stop.
        tool_definitions=tool_definitions,
        output_postfix=config.output_postfix,
        output_prefix=config.output_prefix,
    )
