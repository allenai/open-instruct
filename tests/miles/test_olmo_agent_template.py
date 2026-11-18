"""Olmo TITO appends tool turns without rewriting sampled prefixes."""

from pathlib import Path

import pytest
from miles.utils.chat_template_utils import get_tito_tokenizer, resolve_reasoning_and_tool_call_parser
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    # A local character BPE keeps this a template test, with no Hub downloads.
    tokens = list(dict.fromkeys([chr(i) for i in range(128)] + ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]))
    backend = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=[]))
    backend.add_special_tokens(tokens[-3:])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")
    tokenizer.chat_template = (
        Path(__file__).parents[2] / "open_instruct/test_data/olmo35_chat_template.jinja"
    ).read_text()
    return tokenizer


@pytest.mark.parametrize("reasoning", ["reason carefully", "  keep whitespace\n"])
def test_tool_continuation_matches_native_template_and_preserves_sampled_tokens(tokenizer, reasoning):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Inspect the task"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}
            ],
        },
    ]
    tito = get_tito_tokenizer(tokenizer, "olmo35", chat_template_kwargs={"tools": tools})
    rendered = tito.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    assert rendered[-1] == tokenizer.encode("\n", add_special_tokens=False)[0]
    sampled = rendered[:-1]  # The model stops at im_end, before the template's newline.
    continued = messages + [{"role": "tool", "tool_call_id": "call1", "content": "/sandbox"}]
    merged = tito.merge_tokens(messages, continued, sampled)
    expected = tito.apply_chat_template(continued, add_generation_prompt=True, tokenize=True)
    assert merged == expected
    assert merged[: len(sampled)] == sampled
    assert tokenizer.chat_template.endswith("{%- endfor -%}\n")
    assert resolve_reasoning_and_tool_call_parser("olmo35") == ("qwen3", "qwen3_coder")
