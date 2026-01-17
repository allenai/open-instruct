"""
Tool parsers for extracting tool calls from model output.

Includes:
- Base ToolParser ABC
- VllmToolParser: Wraps vLLM's native tool parsers
- OpenInstructLegacyToolParser: <tool_name>content</tool_name> style
- DRTuluToolParser: <call_tool name="...">query</call_tool> style

For vLLM parsers, add to VLLM_PARSERS!
See: https://docs.vllm.ai/en/latest/features/tool_calling/
"""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import ray
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers import ToolParser as VllmNativeToolParser

from open_instruct.logger_utils import setup_logger
from open_instruct.tools.utils import ToolCall
from open_instruct.utils import import_class_from_string

logger = setup_logger(__name__)


class ToolParser(ABC):
    """Base class for tool parsers."""

    stop_sequences: list[str] = []
    """Stop strings this parser relies on."""

    @abstractmethod
    def get_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract tool calls from model outputs."""
        pass

    @abstractmethod
    def format_tool_outputs(self, tool_outputs: list[str]) -> str:
        """Format multiple tool outputs with any necessary prefixes/postfixes."""
        pass


class OpenInstructLegacyToolParser(ToolParser):
    """
    Parser that recreates the older open-instruct style,
    which also matches DR Tulu and Open-R1 styles.

    Tools are invoked via <tool_name>content</tool_name> tags.
    The content between tags is passed to the tool's first required parameter.
    Only works for tools that take a single string parameter.
    """

    def __init__(self, tool_actors: list[ray.actor.ActorHandle], output_wrap_name: str = "output"):
        """Initialize the parser.

        Args:
            tool_actors: List of Ray actor handles for Tools.
            output_wrap_name: Name to wrap tool outputs with.
        """
        # Fetch metadata from actors
        self.tool_names = [ray.get(actor.get_call_name.remote()) for actor in tool_actors]
        self.output_wrap_name = output_wrap_name
        assert len(self.tool_names) == len(set(self.tool_names)), "Tool names must be unique"
        self.stop_sequences = [f"</{tool_name}>" for tool_name in self.tool_names]
        self.tool_start_strings = [f"<{tool_name}>" for tool_name in self.tool_names]

        self.tool_param_names: dict[str, str] = {}
        for actor, tool_name in zip(tool_actors, self.tool_names):
            params = ray.get(actor.get_parameters.remote())
            required = params.get("required", [])
            if required:
                self.tool_param_names[tool_name] = required[0]
            else:
                properties = params.get("properties", {})
                if properties:
                    self.tool_param_names[tool_name] = next(iter(properties))
                else:
                    self.tool_param_names[tool_name] = "text"

        self.tool_regexes = {
            tool_name: re.compile(re.escape(f"<{tool_name}>") + r"(.*?)" + re.escape(f"</{tool_name}>"), re.DOTALL)
            for tool_name in self.tool_names
        }

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        # Collect all matches with their positions for proper ordering
        matches: list[tuple[int, str, str]] = []  # (position, tool_name, content)
        for tool_name, tool_regex in self.tool_regexes.items():
            for match in tool_regex.finditer(text):
                matches.append((match.start(), tool_name, match.group(1)))

        # Sort by position in text to preserve order of tool calls
        matches.sort(key=lambda x: x[0])

        tool_calls: list[ToolCall] = []
        for _, tool_name, tool_content in matches:
            param_name = self.tool_param_names.get(tool_name, "text")
            tool_calls.append(ToolCall(name=tool_name, args={param_name: tool_content}))
        return tool_calls

    def _format_tool_output(self, tool_output: str) -> str:
        return f"<{self.output_wrap_name}>\n{tool_output}\n</{self.output_wrap_name}>\n"

    def format_tool_outputs(self, tool_outputs: list[str]) -> str:
        return "\n".join(self._format_tool_output(tool_output) for tool_output in tool_outputs)


class VllmToolParser(ToolParser):
    """
    Wraps a vLLM tool parser to extract tool calls and format responses.

    This parser delegates to vLLM's native tool parsing implementations
    (e.g., Hermes, Llama3, Qwen3) while providing a consistent interface.
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
        """Initialize the vLLM tool parser wrapper.

        Args:
            tool_parser: A vLLM native tool parser instance.
            output_formatter: Function to format tool output strings.
            stop_sequences: Stop sequences to use for this parser.
            tool_definitions: Tool definitions in OpenAI format for parsing.
            output_postfix: Postfix to add after all tool outputs (e.g., generation prompt).
            output_prefix: Prefix to add before all tool outputs.
        """
        self.tool_parser = tool_parser
        self.output_formatter = output_formatter
        self.stop_sequences = stop_sequences or []
        self._tool_definitions = tool_definitions
        self._output_postfix = output_postfix
        self._output_prefix = output_prefix

    def _make_request(self) -> Any:
        """Create a dummy ChatCompletionRequest for vLLM parsers.

        Usually these only need the list of tools.
        """
        return ChatCompletionRequest(model="dummy", messages=[], tools=self._tool_definitions)

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract tool calls from model output.

        Args:
            text: The model output text to parse.

        Returns:
            List of ToolCall objects extracted from the text.
        """
        request = self._make_request()
        result = self.tool_parser.extract_tool_calls(model_output=text, request=request)

        if not result.tools_called:
            return []

        tool_calls = []
        for call in result.tool_calls:
            try:
                tool_calls.append(ToolCall(name=call.function.name, args=json.loads(call.function.arguments)))
            except json.JSONDecodeError as e:
                # the model may have mungled the tool call somehow, catch the error here.
                logger.warning(
                    f"VllmToolParser: Failed to parse tool arguments: {e}\nArguments: {call.function.arguments!r}"
                )
                continue
        return tool_calls

    def format_tool_outputs(self, tool_outputs: list[str]) -> str:
        """Format multiple tool outputs with prefix and postfix.

        Args:
            tool_outputs: List of tool output strings to format.

        Returns:
            Formatted string with prefix, all tool outputs, and postfix.
        """
        return f"{self._output_prefix}{''.join(self.output_formatter(output) for output in tool_outputs)}{self._output_postfix}"


@dataclass
class VllmParserConfig:
    """Configuration for a vLLM tool parser."""

    import_path: str
    """Import path for the parser class (e.g., 'vllm.entrypoints.openai.tool_parsers.hermes_tool_parser:Hermes2ProToolParser')."""
    output_template: str
    """Template for formatting each tool output, uses {} as placeholder."""
    output_postfix: str
    """Postfix to add after all tool outputs (includes generation prompt)."""
    stop_sequences: list[str] = field(default_factory=list)
    """Stop sequences to use for this parser. If empty, we rely on the model's native stop sequences."""
    output_prefix: str = ""
    """Prefix to add before all tool outputs (for grouped tool responses)."""


# Registry of available vLLM tool parsers
VLLM_PARSERS: dict[str, VllmParserConfig] = {
    # Hermes-style (also works for Qwen2.5/3)
    "vllm_hermes": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.hermes_tool_parser:Hermes2ProToolParser",
        output_template="<|im_start|>tool\n<tool_response>\n{}\n</tool_response>\n<|im_end|>\n",
        output_postfix="<|im_start|>assistant\n",
    ),
    # Llama 3.x JSON style
    "vllm_llama3_json": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.llama_tool_parser:Llama3JsonToolParser",
        output_template="<|start_header_id|>ipython<|end_header_id|>\n\n{}<|eot_id|>",
        output_postfix="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    # Olmo 3
    "vllm_olmo3": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.olmo3_tool_parser:Olmo3PythonicToolParser",
        output_template="<|im_start|>environment\n{}<|im_end|>\n",
        output_postfix="<|im_start|>assistant\n",
    ),
}


def create_vllm_parser(
    parser_name: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    output_template: str | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> VllmToolParser:
    """Create a VllmToolParser by name.

    Args:
        parser_name: Name of the parser (e.g., "hermes", "llama3_json", "qwen3_coder").
        tokenizer: The tokenizer for the model.
        output_template: Optional custom output template. Uses {} for the tool output.
                        If None, uses the default for this parser.
        tool_definitions: Optional list of tool definitions in OpenAI format.
                         These are stored and used by default in get_tool_calls().

    Returns:
        VllmToolParser configured for the specified model family.
    """
    if parser_name not in VLLM_PARSERS:
        available = list(VLLM_PARSERS.keys())
        raise ValueError(f"Unknown parser: {parser_name}. Available: {available}")

    config = VLLM_PARSERS[parser_name]
    template = output_template or config.output_template

    parser_cls = import_class_from_string(config.import_path)
    native_parser = parser_cls(tokenizer)

    return VllmToolParser(
        tool_parser=native_parser,
        output_formatter=lambda x, t=template: t.format(x),
        stop_sequences=config.stop_sequences,
        tool_definitions=tool_definitions,
        output_postfix=config.output_postfix,
        output_prefix=config.output_prefix,
    )


class DRTuluToolParser(ToolParser):
    """
    Parser for DR Tulu style tool calls. Delegates actual parsing to the tool itself.
    Only detects that a tool call occurred (via stop strings) and passes text to the tool.
    """

    def __init__(self, tool_actors: list[ray.actor.ActorHandle]):
        if len(tool_actors) != 1:
            raise ValueError(f"DRTuluToolParser requires exactly one tool (dr_agent_mcp), got {len(tool_actors)}")

        actor = tool_actors[0]
        self.tool_call_name = ray.get(actor.get_call_name.remote())

        if self.tool_call_name != "dr_agent_mcp":
            raise ValueError(f"DRTuluToolParser requires dr_agent_mcp tool, got {self.tool_call_name}")

        stop_strings = ray.get(actor.get_stop_strings.remote())
        # Use dict.fromkeys to deduplicate while preserving order
        self.stop_sequences = list(dict.fromkeys(stop_strings)) if stop_strings else []

    def get_tool_calls(self, text: str) -> list[ToolCall]:
        for stop in self.stop_sequences:
            if stop in text:
                return [ToolCall(name=self.tool_call_name, args={"text": text})]
        return []

    def format_tool_outputs(self, tool_outputs: list[str]) -> str:
        return "\n".join(f"<tool_output>\n{output}\n</tool_output>\n" for output in tool_outputs)


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return ["legacy", "dr_tulu"] + list(VLLM_PARSERS.keys())


def create_tool_parser(
    parser_type: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    tool_actors: list[ray.actor.ActorHandle],
    tool_definitions: list[dict[str, Any]] | None = None,
) -> ToolParser:
    """Create a tool parser instance by type.

    Args:
        parser_type: Type of parser to create. Options:
            - "legacy": OpenInstructLegacyToolParser for <tool_name>content</tool_name> format
            - "dr_tulu": DRTuluToolParser for <call_tool name="...">content</call_tool> format
            - "vllm_*": VllmToolParser variants (vllm_hermes, vllm_llama3_json, vllm_olmo3)
        tokenizer: Tokenizer for the model (required for all parser types).
        tool_actors: List of Ray actor handles for the tools.
        tool_definitions: OpenAI-format tool definitions (required for vllm_* parsers).

    Returns:
        A ToolParser instance configured for the specified type.

    Raises:
        ValueError: If parser_type is unknown.
    """
    if parser_type == "legacy":
        return OpenInstructLegacyToolParser(tool_actors, output_wrap_name="output")

    if parser_type == "dr_tulu":
        return DRTuluToolParser(tool_actors)

    if parser_type in VLLM_PARSERS:
        return create_vllm_parser(parser_type, tokenizer, tool_definitions=tool_definitions)

    raise ValueError(f"Unknown parser type: '{parser_type}'. Available: {get_available_parsers()}")
