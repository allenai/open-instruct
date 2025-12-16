#!/usr/bin/env python
"""Render canned chat examples with a chosen template to inspect formatting."""

import argparse
from collections.abc import Iterable
from pathlib import Path

from transformers import AutoTokenizer

from open_instruct.dataset_transformation import CHAT_TEMPLATES

MODEL_NAME = "allenai/dolma2-tokenizer"
MODEL_REVISION = "main"

SINGLE_TURN_REASONING = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "The prompt was asdlkasd"},
    {
        "role": "assistant",
        "content": (
            '<think>Okay... user sent "asdlkasd"—probably a test. Stay friendly, invite clarification...</think>\n'
            "It looks like your message might be a bit jumbled! Could you clarify what you're asking about? "
            "I'm here to help with AI, language models, writing, coding—whatever you need."
        ),
    },
]

MULTI_TURN_REASONING = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "The prompt was asdlkasd"},
    {
        "role": "assistant",
        "content": (
            "<think>First turn... just restate the prompt and ask what they need.</think>\n"
            'The prompt you shared was "asdlkasd". Did you want me to expand on it or help craft a new one?'
        ),
    },
    {"role": "user", "content": "Please restate it politely so I can show templating."},
    {
        "role": "assistant",
        "content": (
            "<think>Second turn... reassure them and keep the tone upbeat...</think>\n"
            "Absolutely! It looked like your message might have been a little jumbled—just let me know what you'd "
            "like to explore and I'm happy to dive in."
        ),
    },
]

BASIC_CHAT_TRANSCRIPT = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

EXAMPLES = {
    "single_reasoning": {
        "messages": SINGLE_TURN_REASONING,
        "description": "Single assistant turn with <think> reasoning.",
    },
    "multi_reasoning": {
        "messages": MULTI_TURN_REASONING,
        "description": "Two assistant turns, both containing <think> traces.",
    },
    "basic_chat": {"messages": BASIC_CHAT_TRANSCRIPT, "description": "Simple chat without reasoning tags."},
}

DEFAULT_EXAMPLES = ("single_reasoning", "multi_reasoning")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name", default=MODEL_NAME, help="Tokenizer identifier on Hugging Face (default: %(default)s)."
    )
    parser.add_argument(
        "--revision", default=MODEL_REVISION, help="Tokenizer revision, tag, or commit (default: %(default)s)."
    )
    parser.add_argument(
        "--template",
        default="olmo_thinker_remove_intermediate_thinking",
        help=(
            "Either the key of a template in open_instruct.dataset_transformation.CHAT_TEMPLATES "
            "or a filesystem path to a Jinja template."
        ),
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=list(DEFAULT_EXAMPLES),
        choices=list(EXAMPLES.keys()) + ["all"],
        help="Which canned message sets to render (use 'all' for everything).",
    )
    parser.add_argument(
        "--show-tokens", action="store_true", help="Also print token ids and counts to compare serialized lengths."
    )
    parser.add_argument(
        "--snippet-len",
        type=int,
        default=160,
        help="Character cap for message previews before printing token ids (0 to disable truncation).",
    )
    return parser.parse_args()


def resolve_examples(selection: Iterable[str]) -> list[str]:
    ordered = list(dict.fromkeys(selection))  # preserve order, drop duplicates
    if "all" in ordered:
        return list(EXAMPLES.keys())
    return ordered


def load_template(template_arg: str) -> str:
    template_path = Path(template_arg)
    if template_path.exists():
        return template_path.read_text()
    if template_arg in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[template_arg]
    raise ValueError(
        f"Template '{template_arg}' is neither a file nor a key in CHAT_TEMPLATES. "
        f"Available keys: {', '.join(sorted(CHAT_TEMPLATES.keys()))}"
    )


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision, use_fast=True)
    tokenizer.chat_template = load_template(args.template)

    had_error = False
    for example_name in resolve_examples(args.examples):
        example = EXAMPLES[example_name]
        print("\n" + "=" * 80)
        print(f"{example_name} :: {example['description']}")
        print("-" * 80)

        print("Messages:")
        for idx, message in enumerate(example["messages"], start=1):
            snippet = message["content"].replace("\n", " ")
            if args.snippet_len and len(snippet) > args.snippet_len:
                snippet = snippet[: args.snippet_len - 3] + "..."
            print(f"  {idx}. {message['role']}: {snippet}")

        print("\nFormatted:")
        try:
            rendered = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        except Exception as exc:  # pragma: no cover - helpful for manual debugging
            had_error = True
            print(f"[ERROR] {exc}")
        else:
            print(rendered)
            if args.show_tokens:
                print("\nTokenized:")
                token_data = tokenizer.apply_chat_template(
                    example["messages"], tokenize=True, add_generation_prompt=False
                )
                if isinstance(token_data, list):
                    ids = token_data
                elif hasattr(token_data, "tolist"):
                    ids = token_data.tolist()
                    if ids and isinstance(ids[0], list):
                        ids = ids[0]
                elif hasattr(token_data, "__iter__"):
                    ids = list(token_data)
                else:
                    ids = [int(token_data)]
                print(f"\nToken count: {len(ids)}")
                print(ids)
        print("=" * 80)
    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
