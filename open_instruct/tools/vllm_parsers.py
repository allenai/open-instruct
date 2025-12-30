"""
vLLM Tool Parser Factory

Simple factory for creating VllmToolParser instances that wrap vLLM's native tool parsers.
See: https://docs.vllm.ai/en/latest/features/tool_calling/

Usage:
    parser = create_vllm_parser("hermes", tokenizer)
    tool_calls = parser.get_tool_calls(model_output)
    formatted = parser.format_tool_calls(tool_result)

To add a new parser, add an entry to VLLM_PARSERS dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from open_instruct.tools.base import VllmToolParser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class VllmParserConfig:
    """Configuration for a vLLM tool parser."""

    import_path: str  # e.g., "vllm.entrypoints.openai.tool_parsers.hermes_tool_parser:Hermes2ProToolParser"
    output_template: str  # Template for formatting tool output, uses {} as placeholder
    stop_sequences: list[str]  # Stop sequences to use for this parser


# =============================================================================
# Parser Registry
# =============================================================================

VLLM_PARSERS: dict[str, VllmParserConfig] = {
    # Hermes-style (also works for Qwen2.5)
    "hermes": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.hermes_tool_parser:Hermes2ProToolParser",
        output_template="<tool_response>\n{}\n</tool_response>",
        stop_sequences=["</tool_call>"],
    ),
    # Llama 3.x JSON style
    "llama3_json": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.llama_tool_parser:Llama3JsonToolParser",
        output_template="<|python_tag|>{}",
        stop_sequences=["<|eom_id|>"],
    ),
    # Qwen3 XML style
    "qwen3_xml": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser:Qwen3XMLToolParser",
        output_template="<|im_start|>observation\n{}\n<|im_end|>",
        stop_sequences=["</tool_call>"],
    ),
    # Qwen3 Coder
    "qwen3_coder": VllmParserConfig(
        import_path="vllm.entrypoints.openai.tool_parsers.qwen3coder_tool_parser:Qwen3CoderToolParser",
        output_template="<|im_start|>observation\n{}\n<|im_end|>",
        stop_sequences=["</tool_call>"],
    ),
}


def _import_parser_class(import_path: str) -> type:
    """Import a parser class from module:ClassName path."""
    import importlib

    module_name, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_available_vllm_parsers() -> list[str]:
    """Return list of available vLLM parser names."""
    return list(VLLM_PARSERS.keys())


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
        stop_sequences=config.stop_sequences,
        tool_definitions=tool_definitions,
    )
