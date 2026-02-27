"""
Visualize SFT tokenization masking for debugging and inspection.

This script applies sft_tulu_tokenize_and_truncate_v1 to a dataset row and displays
the tokens with color-coding to show which tokens are masked (no loss computed) vs
unmasked (loss computed during training).

Usage:
    uv run python scripts/visualize_tokenization.py -d DATASET -t TOKENIZER [OPTIONS]

Arguments:
    -d, --dataset       HuggingFace dataset name (required)
                        Example: allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated-200K-thinking-id-fixed

    -t, --tokenizer     Tokenizer source - either a HuggingFace model name or a key from
                        CHAT_TEMPLATES in open_instruct/dataset_transformation.py (required)
                        Example HuggingFace: allenai/Olmo-3.1-32B-Think
                        Example CHAT_TEMPLATES key: tulu, olmo, olmo_thinker, etc.

    -i, --row-idx       Row index to process (default: 0)

    -m, --max-length    Maximum number of characters to display. By default, uses
                        approximately half the terminal area (height * width / 2)

Examples:
    # Basic usage with HuggingFace tokenizer
    uv run python scripts/viz_sft_tokenization.py \\
        -d allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated-200K-thinking-id-fixed \\
        -t allenai/Olmo-3.1-32B-Think

    # Using a CHAT_TEMPLATES key
    uv run python scripts/viz_sft_tokenization.py \\
        -d allenai/Dolci-Think-SFT-32B \\
        -t tulu_thinker

    # Process a specific row with custom max length
    uv run python scripts/viz_sft_tokenization.py \\
        -d allenai/Dolci-Think-SFT-32B \\
        -t allenai/Olmo-3.1-32B-Think \\
        -i 42 \\
        -m 5000

Available CHAT_TEMPLATES keys:
    simple_concat_with_space, simple_concat_with_new_line, simple_chat,
    assistant_message_only, zephyr, olmo, tulu, tulu_thinker, tulu_thinker_r1_style,
    olmo_old, olmo_thinker, olmo_thinker_no_think_7b,
    olmo_thinker_remove_intermediate_thinking, olmo_thinker_no_think_sft_tokenization,
    olmo_thinker_rlzero, olmo_thinker_code_rlzero, r1_simple_chat,
    r1_simple_chat_postpend_think, r1_simple_chat_postpend_think_orz_style,
    r1_simple_chat_postpend_think_tool_vllm

Output:
    RED   = masked tokens (label=-100, no loss computed)
    GREEN = unmasked tokens (loss computed during training)
"""

import argparse
import gc
import itertools
import os
import shutil
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

from open_instruct.dataset_transformation import CHAT_TEMPLATES, sft_tulu_tokenize_and_truncate_v1


def get_terminal_size():
    """Get terminal size, with fallback defaults."""
    size = shutil.get_terminal_size(fallback=(120, 40))
    return size.columns, size.lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize SFT tokenization masking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., allenai/Dolci-Think-SFT-32B)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer: HuggingFace model name or CHAT_TEMPLATES key (e.g., allenai/Olmo-3.1-32B-Think or tulu)",
    )
    parser.add_argument(
        "-i",
        "--row-idx",
        type=int,
        default=0,
        help="Row index to process (default: 0)",
    )
    parser.add_argument(
        "-m",
        "--max-length",
        type=int,
        default=None,
        help="Max characters to display (default: terminal height * width / 2)",
    )
    return parser.parse_args()


def load_tokenizer(tokenizer_arg: str):
    """Load tokenizer from HuggingFace or apply CHAT_TEMPLATES."""
    # First, always load from HuggingFace to get the base tokenizer
    if tokenizer_arg in CHAT_TEMPLATES:
        # Use a default tokenizer and override chat template
        # Try common tokenizers that work with most templates
        tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")
        tokenizer.chat_template = CHAT_TEMPLATES[tokenizer_arg]
        print(f"Using CHAT_TEMPLATES['{tokenizer_arg}'] with base tokenizer allenai/OLMo-2-1124-7B-Instruct")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_arg)
        print(f"Loaded tokenizer from HuggingFace: {tokenizer_arg}")
    return tokenizer


def get_row_by_index(dataset_name: str, row_idx: int) -> dict:
    """Get a specific row from a streaming dataset."""
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    # Use islice to avoid keeping the iterator open
    rows = list(itertools.islice(dataset, row_idx, row_idx + 1))
    if not rows:
        raise ValueError(f"Row index {row_idx} not found in dataset")
    # Force cleanup of the dataset iterator to avoid PyGILState_Release error
    del dataset
    gc.collect()
    return rows[0]


def visualize_tokens(input_ids, labels, tokenizer, max_chars: int):
    """Print tokens with color coding based on masking."""
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    print("\n" + "=" * 80)
    print("Token visualization (RED = masked/no loss, GREEN = unmasked/loss computed):")
    print("=" * 80 + "\n")

    char_count = 0
    truncated = False

    for token_id, label in zip(input_ids, labels):
        token_str = tokenizer.decode([token_id.item()])
        # Escape newlines and tabs for display
        display_str = token_str.replace("\n", "\\n").replace("\t", "\\t")

        if char_count + len(display_str) > max_chars:
            truncated = True
            break

        if label == -100:
            print(f"{RED}{display_str}{RESET}", end="")
        else:
            print(f"{GREEN}{display_str}{RESET}", end="")

        char_count += len(display_str)

    print("\n")
    if truncated:
        print(f"[Output truncated at {max_chars} characters. Use -m to adjust.]")
    print("=" * 80)
    print(f"Legend: {RED}RED = masked (no loss){RESET}, {GREEN}GREEN = unmasked (loss computed){RESET}")


def main():
    args = parse_args()

    # Determine max_length
    if args.max_length is None:
        term_width, term_height = get_terminal_size()
        max_chars = (term_width * term_height) // 2
    else:
        max_chars = args.max_length

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Load dataset row
    print(f"Loading row {args.row_idx} from {args.dataset}...")
    row = get_row_by_index(args.dataset, args.row_idx)

    # Print message info
    print(f"\nMessages in row {args.row_idx}:")
    for msg in row["messages"]:
        content_preview = msg["content"][:100].replace("\n", "\\n")
        print(f"  {msg['role']}: {content_preview}...")

    # Apply tokenization
    result = sft_tulu_tokenize_and_truncate_v1(
        row=row,
        tokenizer=tokenizer,
        max_seq_length=4096,
    )

    # Print stats
    print(f"\nTokenization stats:")
    print(f"  input_ids shape: {result['input_ids'].shape}")
    print(f"  labels shape: {result['labels'].shape}")
    print(f"  Masked tokens (label=-100): {(result['labels'] == -100).sum().item()}")
    print(f"  Unmasked tokens: {(result['labels'] != -100).sum().item()}")

    # Visualize
    visualize_tokens(result["input_ids"], result["labels"], tokenizer, max_chars)


if __name__ == "__main__":
    main()
    # Flush output before force exit
    sys.stdout.flush()
    sys.stderr.flush()
    # Force exit to avoid PyGILState_Release error from background threads
    # in tokenizers/datasets libraries during normal Python shutdown
    os._exit(0)
