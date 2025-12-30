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

from typing import TYPE_CHECKING, Any

from open_instruct.tools.base import VllmToolParser

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# =============================================================================
# Parser Registry
# =============================================================================
# Each entry: (vllm_parser_import_path, output_format_template)
# The template uses {} as placeholder for tool output.

VLLM_PARSERS: dict[str, tuple[str, str]] = {
    # Hermes-style (also works for Qwen2.5)
    "hermes": (
        "vllm.entrypoints.openai.tool_parsers.hermes_tool_parser:Hermes2ProToolParser",
        "<tool_response>\n{}\n</tool_response>",
    ),
    # Llama 3.x JSON style
    "llama3_json": ("vllm.entrypoints.openai.tool_parsers.llama_tool_parser:Llama3JsonToolParser", "<|python_tag|>{}"),
    # Qwen3 XML style
    "qwen3_xml": (
        "vllm.entrypoints.openai.tool_parsers.qwen3xml_tool_parser:Qwen3XMLToolParser",
        "<|im_start|>observation\n{}\n<|im_end|>",
    ),
    # Qwen3 Coder
    "qwen3_coder": (
        "vllm.entrypoints.openai.tool_parsers.qwen3coder_tool_parser:Qwen3CoderToolParser",
        "<|im_start|>observation\n{}\n<|im_end|>",
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
) -> VllmToolParser:
    """Create a VllmToolParser by name.

    Args:
        parser_name: Name of the parser (e.g., "hermes", "llama3_json").
        tokenizer: The tokenizer for the model.
        output_template: Optional custom output template. Uses {} for the tool output.
                        If None, uses the default for this parser.

    Returns:
        VllmToolParser configured for the specified model family.

    Example:
        >>> parser = create_vllm_parser("hermes", tokenizer)
        >>> # Or with custom output format:
        >>> parser = create_vllm_parser("hermes", tokenizer, output_template="<result>{}</result>")
    """
    if parser_name not in VLLM_PARSERS:
        available = get_available_vllm_parsers()
        raise ValueError(f"Unknown parser: {parser_name}. Available: {available}")

    import_path, default_template = VLLM_PARSERS[parser_name]
    template = output_template or default_template

    parser_cls = _import_parser_class(import_path)
    native_parser = parser_cls(tokenizer)

    return VllmToolParser(tool_parser=native_parser, output_formatter=lambda x, t=template: t.format(x))
