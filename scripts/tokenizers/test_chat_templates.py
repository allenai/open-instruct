#!/usr/bin/env python
"""Render chat template output for canned examples or real HuggingFace dataset rows.

Can load from:
  - Built-in canned examples (--examples)
  - Any HuggingFace dataset row (--dataset + --row-idx)

Supports SFT datasets (with "messages" key) and DPO/preference datasets
(with "chosen"/"rejected" keys), rendering each conversation separately.

Usage:
    # Canned examples
    uv run python scripts/tokenizers/test_chat_templates.py --template tulu --examples all

    # Real SFT dataset row
    uv run python scripts/tokenizers/test_chat_templates.py \\
        --model-name allenai/OLMo-3.2-Hybrid-7B-Instruct-SFT \\
        --dataset allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated-200K-thinking-id-fixed

    # Real DPO dataset row
    uv run python scripts/tokenizers/test_chat_templates.py \\
        --model-name allenai/OLMo-3.2-Hybrid-7B-Instruct-SFT \\
        --dataset allenai/Dolci-Instruct-DPO --row-idx 0
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

from transformers import AutoTokenizer

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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-name", default=MODEL_NAME, help="Tokenizer identifier on Hugging Face (default: %(default)s)."
    )
    parser.add_argument(
        "--revision", default=MODEL_REVISION, help="Tokenizer revision, tag, or commit (default: %(default)s)."
    )
    parser.add_argument(
        "--template",
        default=None,
        help=(
            "Either the key of a template in open_instruct.dataset_transformation.CHAT_TEMPLATES "
            "or a filesystem path to a Jinja template. If not set, uses the tokenizer's built-in template."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="HuggingFace dataset name. Supports SFT (messages key) and DPO (chosen/rejected keys).",
    )
    parser.add_argument(
        "--row-idx",
        type=int,
        default=0,
        help="Row index to load from the dataset (default: 0).",
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=None,
        choices=list(EXAMPLES.keys()) + ["all"],
        help="Which canned message sets to render (use 'all' for everything). Ignored when --dataset is used.",
    )
    parser.add_argument(
        "--show-tokens", action="store_true", help="Also print token ids and counts to compare serialized lengths."
    )
    parser.add_argument(
        "--snippet-len",
        type=int,
        default=160,
        help="Character cap for message previews (0 to disable truncation).",
    )
    return parser.parse_args()


def resolve_examples(selection: Iterable[str] | None) -> list[str]:
    if selection is None:
        return list(DEFAULT_EXAMPLES)
    ordered = list(dict.fromkeys(selection))  # preserve order, drop duplicates
    if "all" in ordered:
        return list(EXAMPLES.keys())
    return ordered


def load_template(template_arg: str) -> str:
    template_path = Path(template_arg)
    if template_path.exists():
        return template_path.read_text()
    # Lazy import to avoid pulling in torch (which can segfault on shutdown)
    from open_instruct.dataset_transformation import CHAT_TEMPLATES

    if template_arg in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[template_arg]
    raise ValueError(
        f"Template '{template_arg}' is neither a file nor a key in CHAT_TEMPLATES. "
        f"Available keys: {', '.join(sorted(CHAT_TEMPLATES.keys()))}"
    )


def load_dataset_row(dataset_name: str, row_idx: int) -> dict:
    """Load a single row from a HuggingFace dataset using streaming (avoids downloading the full dataset)."""
    import itertools

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    rows = list(itertools.islice(dataset, row_idx, row_idx + 1))
    if not rows:
        raise ValueError(f"Row index {row_idx} not found in dataset {dataset_name}")
    return rows[0]


def extract_conversations(row: dict, dataset_name: str, row_idx: int) -> list[dict]:
    """Extract named conversations from a dataset row.

    Returns a list of {"name": str, "messages": list, "description": str}.
    Handles SFT (messages key) and DPO (chosen/rejected keys).
    """
    conversations = []
    if "messages" in row:
        conversations.append({
            "name": f"dataset row {row_idx}",
            "messages": row["messages"],
            "description": f"{dataset_name} row {row_idx} (SFT)",
        })
    if "chosen" in row:
        conversations.append({
            "name": f"dataset row {row_idx} [chosen]",
            "messages": row["chosen"],
            "description": f"{dataset_name} row {row_idx} (DPO chosen)",
        })
    if "rejected" in row:
        conversations.append({
            "name": f"dataset row {row_idx} [rejected]",
            "messages": row["rejected"],
            "description": f"{dataset_name} row {row_idx} (DPO rejected)",
        })
    if not conversations:
        raise ValueError(
            f"Row has no recognized message keys. Found keys: {list(row.keys())}. "
            "Expected 'messages' (SFT) or 'chosen'/'rejected' (DPO)."
        )
    return conversations


def render_conversation(tokenizer, messages, name, description, snippet_len, show_tokens):
    """Render and print a single conversation."""
    print("\n" + "=" * 80)
    print(f"{name} :: {description}")
    print("-" * 80)

    print("Messages:")
    for idx, message in enumerate(messages, start=1):
        content = message.get("content", "")
        if content is None:
            content = ""
        snippet = content.replace("\n", " ")
        if snippet_len and len(snippet) > snippet_len:
            snippet = snippet[: snippet_len - 3] + "..."
        print(f"  {idx}. {message['role']}: {snippet}")

    print("\nFormatted:")
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        print("=" * 80)
        return True  # had_error

    print(rendered)
    if show_tokens:
        print("\nTokenized:")
        token_data = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
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
    return False  # no error


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision, use_fast=True)
    if args.template is not None:
        tokenizer.chat_template = load_template(args.template)
        print(f"Template: {args.template}")
    else:
        print(f"Using built-in chat template from {args.model_name}")

    had_error = False

    if args.dataset:
        print(f"Loading row {args.row_idx} from {args.dataset}...")
        row = load_dataset_row(args.dataset, args.row_idx)
        conversations = extract_conversations(row, args.dataset, args.row_idx)
        for conv in conversations:
            err = render_conversation(
                tokenizer, conv["messages"], conv["name"], conv["description"],
                args.snippet_len, args.show_tokens,
            )
            if err:
                had_error = True
    else:
        for example_name in resolve_examples(args.examples):
            example = EXAMPLES[example_name]
            err = render_conversation(
                tokenizer, example["messages"], example_name, example["description"],
                args.snippet_len, args.show_tokens,
            )
            if err:
                had_error = True

    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    import os
    import sys

    rc = 0
    try:
        main()
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    # The HuggingFace tokenizers library spawns background threads that can
    # segfault during normal Python interpreter shutdown. Flush and force-exit
    # to avoid this. Same workaround used in visualize_tokenization.py.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
