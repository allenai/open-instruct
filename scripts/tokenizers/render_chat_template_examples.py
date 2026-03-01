#!/usr/bin/env python
"""
Render chat template outputs for debugging.

This script loads a tokenizer, overrides its chat template, and runs
`tokenizer.apply_chat_template(..., tokenize=False)` on a few sample
message sequences that include interleaved reasoning content.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable, List, Mapping, Sequence

from transformers import AutoTokenizer  # type: ignore

from open_instruct.dataset_transformation import CHAT_TEMPLATES


DEFAULT_SAMPLES: Mapping[str, Sequence[Mapping[str, str]]] = {
    "single_turn": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "<think>Recall greeting etiquette</think>Hello!"},
    ],
    "multi_turn_interleaved": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "<think>Calculate 2 + 2</think>It's 4."},
        {"role": "user", "content": "Now double that and explain."},
        {
            "role": "assistant",
            "content": "It's 8.",
            "reasoning_content": "Re-use previous result, multiply by 2, then explain clearly.",
        },
    ],
}


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect how a chat template renders sample conversations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer-name-or-path",
        required=True,
        help="Tokenizer identifier or path for AutoTokenizer.from_pretrained.",
    )
    parser.add_argument(
        "--chat-template",
        required=True,
        help=(
            "Chat template key from CHAT_TEMPLATES or a path to a file containing the template text."
        ),
    )
    parser.add_argument(
        "--messages-json",
        help=(
            "Optional path to a JSON file containing a list of chat messages. "
            "See DEFAULT_SAMPLES in this script for the expected structure."
        ),
    )
    parser.add_argument(
        "--sample",
        choices=sorted(DEFAULT_SAMPLES),
        default="multi_turn_interleaved",
        help="Which built-in sample conversation to render when --messages-json is not provided.",
    )
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Pass add_generation_prompt=True to tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Set local_files_only=True when loading the tokenizer.",
    )
    return parser.parse_args(argv)


def load_messages(args: argparse.Namespace) -> List[Mapping[str, str]]:
    if args.messages_json:
        with open(args.messages_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("messages JSON must be a list of message dicts")
        return data  # type: ignore[return-value]
    return list(DEFAULT_SAMPLES[args.sample])


def load_chat_template(template_arg: str) -> str:
    if template_arg in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[template_arg]
    with open(template_arg, "r", encoding="utf-8") as handle:
        return handle.read()


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        local_files_only=args.local_files_only,
    )
    tokenizer.chat_template = load_chat_template(args.chat_template)

    messages = load_messages(args)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=args.add_generation_prompt,
    )
    print("=== Chat Template Output ===")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
