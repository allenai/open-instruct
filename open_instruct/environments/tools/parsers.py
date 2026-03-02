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
from dataclasses import dataclass, field
from typing import Any

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParser as VllmNativeToolParser

from open_instruct.environments.base import EnvCall
from open_instruct.logger_utils import setup_logger
from open_instruct.utils import import_class_from_string

logger = setup_logger(__name__)


class ToolParser(ABC):
    """Base class for tool parsers."""

    stop_sequences: list[str] = []
    """Stop strings this parser relies on."""

    @abstractmethod
    def get_tool_calls(self, text: str) -> list[EnvCall]:
        """Extract tool calls from model outputs."""
        pass

    @abstractmethod
    def format_tool_outputs(self, tool_outputs: list[str], role: str = "tool") -> str:
        """Format tool outputs with any necessary prefixes/postfixes.

        Args:
            tool_outputs: Raw output strings from tool/env calls.
            role: The output role (e.g. "tool", "user" for text envs).
        """
        pass


class OpenInstructLegacyToolParser(ToolParser):
    """
    Parser that recreates the older open-instruct style,
    which also matches DR Tulu and Open-R1 styles.

    Tools are invoked via <tool_name>content</tool_name> tags.
    The content between tags is passed to the tool's first required parameter.
    Only works for tools that take a single string parameter.

    Tool names and parameter names are derived from OpenAI-format tool definitions.
    """

    def __init__(self, tool_definitions: list[dict[str, Any]] | None = None, output_wrap_name: str = "output"):
        self.output_wrap_name = output_wrap_name

        if tool_definitions:
            self.tool_names = [td["function"]["name"] for td in tool_definitions]
            self.tool_param_names: dict[str, str] = {}
            for td in tool_definitions:
                func = td["function"]
                name = func["name"]
                params = func.get("parameters", {})
                required = params.get("required", [])
                if required:
                    self.tool_param_names[name] = required[0]
                else:
                    properties = params.get("properties", {})
                    self.tool_param_names[name] = next(iter(properties)) if properties else "text"
        else:
            self.tool_names = []
            self.tool_param_names = {}

        assert len(self.tool_names) == len(set(self.tool_names)), "Tool names must be unique"
        self.stop_sequences = [f"</{tool_name}>" for tool_name in self.tool_names]
        self.tool_start_strings = [f"<{tool_name}>" for tool_name in self.tool_names]
        self.tool_regexes = {
            tool_name: re.compile(re.escape(f"<{tool_name}>") + r"(.*?)" + re.escape(f"</{tool_name}>"), re.DOTALL)
            for tool_name in self.tool_names
        }

    def get_tool_calls(self, text: str) -> list[EnvCall]:
        # Collect all matches with their positions for proper ordering
        matches: list[tuple[int, str, str]] = []  # (position, tool_name, content)
        for tool_name, tool_regex in self.tool_regexes.items():
            for match in tool_regex.finditer(text):
                matches.append((match.start(), tool_name, match.group(1)))

        # Sort by position in text to preserve order of tool calls
        matches.sort(key=lambda x: x[0])

        tool_calls: list[EnvCall] = []
        for _, tool_name, tool_content in matches:
            param_name = self.tool_param_names.get(tool_name, "text")
            tool_calls.append(EnvCall(id="", name=tool_name, args={param_name: tool_content}))
        return tool_calls

    def _format_tool_output(self, tool_output: str) -> str:
        return f"<{self.output_wrap_name}>\n{tool_output}\n</{self.output_wrap_name}>\n"

    def format_tool_outputs(self, tool_outputs: list[str], role: str = "tool") -> str:
        return "\n".join(self._format_tool_output(output) for output in tool_outputs)


class VllmToolParser(ToolParser):
    """Wraps a vLLM tool parser to extract tool calls and format responses.

    Delegates tool-call extraction to vLLM's native parsers (Hermes, Llama3,
    Qwen3, etc.) and formats outputs using per-role templates from
    ``role_templates`` (each value has an ``{output}`` placeholder with the
    role name baked in).
    """

    def __init__(
        self,
        tool_parser: VllmNativeToolParser,
        role_templates: dict[str, str],
        stop_sequences: list[str] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
        output_prefix: str = "",
        output_postfix: str = "<|im_start|>assistant\n",
    ):
        self.tool_parser = tool_parser
        self._role_templates = role_templates
        self.stop_sequences = stop_sequences or []
        self._tool_definitions = tool_definitions
        self._output_prefix = output_prefix
        self._output_postfix = output_postfix

    def _make_request(self) -> Any:
        """Create a dummy ChatCompletionRequest for vLLM parsers.

        Usually these only need the list of tools.
        """
        return ChatCompletionRequest(model="dummy", messages=[], tools=self._tool_definitions)

    def get_tool_calls(self, text: str) -> list[EnvCall]:
        """Extract tool calls from model output.

        Args:
            text: The model output text to parse.

        Returns:
            List of EnvCall objects extracted from the text.
        """
        request = self._make_request()
        result = self.tool_parser.extract_tool_calls(model_output=text, request=request)

        if not result.tools_called:
            return []

        tool_calls = []
        for call in result.tool_calls:
            try:
                tool_calls.append(
                    EnvCall(id=call.id or "", name=call.function.name, args=json.loads(call.function.arguments))
                )
            except json.JSONDecodeError as e:
                # the model may have mungled the tool call somehow, catch the error here.
                logger.warning(
                    f"VllmToolParser: Failed to parse tool arguments: {e}\nArguments: {call.function.arguments!r}"
                )
                continue
        return tool_calls

    def _format_tool_output(self, tool_output: str, role: str = "tool") -> str:
        template = self._role_templates[role]
        return template.format(output=tool_output)

    def format_tool_outputs(self, tool_outputs: list[str], role: str = "tool") -> str:
        return f"{self._output_prefix}{''.join(self._format_tool_output(output, role) for output in tool_outputs)}{self._output_postfix}"


@dataclass
class VllmParserConfig:
    """Configuration for a vLLM tool parser.

    Each config pairs a vLLM native parser with per-role templates that wrap
    individual outputs.  Each template value contains an ``{output}``
    placeholder with the role name baked in.
    """

    import_path: str
    """Dotted import path for the vLLM native parser class
    (e.g. ``"vllm.tool_parsers.hermes_tool_parser:Hermes2ProToolParser"``)."""

    role_templates: dict[str, str]
    """Per-role templates. Keys are role names (e.g. ``"tool"``, ``"user"``),
    values are templates with an ``{output}`` placeholder."""

    output_postfix: str
    """String appended after all formatted outputs (typically starts the
    assistant turn, e.g. ``"<|im_start|>assistant\\n"``)."""

    stop_sequences: list[str] = field(default_factory=list)
    """Stop sequences for this parser."""

    output_prefix: str = ""
    """String prepended before all formatted outputs."""


# Registry of available vLLM tool parsers
VLLM_PARSERS: dict[str, VllmParserConfig] = {
    # Hermes-style / ChatML (also works for Qwen2.5/3)
    "vllm_hermes": VllmParserConfig(
        import_path="vllm.tool_parsers.hermes_tool_parser:Hermes2ProToolParser",
        role_templates={
            "tool": "<|im_start|>tool\n<tool_response>\n{output}\n</tool_response>\n<|im_end|>\n",
            "user": "<|im_start|>user\n{output}<|im_end|>\n",
        },
        output_postfix="<|im_start|>assistant\n",
    ),
    # Llama 3.x JSON style
    "vllm_llama3_json": VllmParserConfig(
        import_path="vllm.tool_parsers.llama_tool_parser:Llama3JsonToolParser",
        role_templates={
            "tool": "<|start_header_id|>ipython<|end_header_id|>\n\n{output}<|eot_id|>",
            "user": "<|start_header_id|>user<|end_header_id|>\n\n{output}<|eot_id|>",
        },
        output_postfix="<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    # Olmo 3
    "vllm_olmo3": VllmParserConfig(
        import_path="vllm.tool_parsers.olmo3_tool_parser:Olmo3PythonicToolParser",
        role_templates={
            "tool": "<|im_start|>environment\n{output}<|im_end|>\n",
            "user": "<|im_start|>user\n{output}<|im_end|>\n",
        },
        output_postfix="<|im_start|>assistant\n",
    ),
}


def create_vllm_parser(
    parser_name: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> VllmToolParser:
    """Create a VllmToolParser by name.

    Args:
        parser_name: Name of the parser (e.g., "vllm_hermes", "vllm_llama3_json").
        tokenizer: The tokenizer for the model.
        tool_definitions: Optional list of tool definitions in OpenAI format.

    Returns:
        VllmToolParser configured for the specified model family.
    """
    if parser_name not in VLLM_PARSERS:
        available = list(VLLM_PARSERS.keys())
        raise ValueError(f"Unknown parser: {parser_name}. Available: {available}")

    config = VLLM_PARSERS[parser_name]
    parser_cls = import_class_from_string(config.import_path)
    native_parser = parser_cls(tokenizer)

    return VllmToolParser(
        tool_parser=native_parser,
        role_templates=config.role_templates,
        stop_sequences=config.stop_sequences,
        tool_definitions=tool_definitions,
        output_prefix=config.output_prefix,
        output_postfix=config.output_postfix,
    )


class DRTuluToolParser(ToolParser):
    """
    Parser for DR Tulu style tool calls. Delegates actual parsing to the tool itself.
    Only detects that a tool call occurred (via stop strings) and passes text to the tool.

    Requires exactly one tool (dr_agent_mcp) in tool_definitions.
    """

    def __init__(self, tool_definitions: list[dict[str, Any]], stop_sequences: list[str]):
        if len(tool_definitions) != 1:
            raise ValueError(f"DRTuluToolParser requires exactly one tool (dr_agent_mcp), got {len(tool_definitions)}")

        self.tool_call_name = tool_definitions[0]["function"]["name"]

        if self.tool_call_name != "dr_agent_mcp":
            raise ValueError(f"DRTuluToolParser requires dr_agent_mcp tool, got {self.tool_call_name}")

        self.stop_sequences = list(dict.fromkeys(stop_sequences)) if stop_sequences else []

        if not self.stop_sequences:
            logger.warning("DRTuluToolParser initialized with no stop sequences — tool calls will never be detected")

    def get_tool_calls(self, text: str) -> list[EnvCall]:
        for stop in self.stop_sequences:
            if stop in text:
                return [EnvCall(id="", name=self.tool_call_name, args={"text": text})]
        return []

    def _format_tool_output(self, tool_output: str) -> str:
        return f"<tool_output>\n{tool_output}\n</tool_output>\n"

    def format_tool_outputs(self, tool_outputs: list[str], role: str = "tool") -> str:
        return "\n".join(self._format_tool_output(output) for output in tool_outputs)


def get_available_parsers() -> list[str]:
    """Return list of available parser types."""
    return ["legacy", "dr_tulu"] + list(VLLM_PARSERS.keys())


def create_tool_parser(
    parser_type: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    tool_definitions: list[dict[str, Any]] | None = None,
    stop_sequences: list[str] | None = None,
) -> ToolParser:
    """Create a tool parser instance by type.

    Args:
        parser_type: Type of parser to create. Options:
            - "legacy": OpenInstructLegacyToolParser for <tool_name>content</tool_name> format
            - "dr_tulu": DRTuluToolParser for <call_tool name="...">content</call_tool> format
            - "vllm_*": VllmToolParser variants (vllm_hermes, vllm_llama3_json, vllm_olmo3)
        tokenizer: Tokenizer for the model (required for all parser types).
        tool_definitions: OpenAI-format tool definitions.
        stop_sequences: a list of stop sequences to use for stopping generations.

    Returns:
        A ToolParser instance configured for the specified type.

    Raises:
        ValueError: If parser_type is unknown.
    """
    if parser_type == "legacy":
        return OpenInstructLegacyToolParser(tool_definitions=tool_definitions, output_wrap_name="output")

    if parser_type == "dr_tulu":
        if tool_definitions is None or stop_sequences is None:
            raise ValueError("dr_tulu parser requires both tool_definitions and stop_sequences")
        return DRTuluToolParser(tool_definitions=tool_definitions, stop_sequences=stop_sequences)

    if parser_type in VLLM_PARSERS:
        return create_vllm_parser(parser_type, tokenizer, tool_definitions=tool_definitions)

    raise ValueError(f"Unknown parser type: '{parser_type}'. Available: {get_available_parsers()}")
