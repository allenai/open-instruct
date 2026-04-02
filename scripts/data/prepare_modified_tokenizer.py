"""
Prepare a modified tokenizer for use with olmo chat templates.

Olmo chat templates use <|im_end|> as text markers throughout conversations, but
reserve {{ eos_token }} for only the final token (used by olmo-core for document
boundary detection in bin-packing). If a tokenizer's eos_token IS <|im_end|>,
every occurrence would be treated as a document boundary, fragmenting conversations.

This script changes eos_token to a different token (e.g., <|endoftext|>) so that
<|im_end|> becomes a regular token and only the new eos_token marks boundaries.

Usage:
    python scripts/data/prepare_modified_tokenizer.py \
        --model Qwen/Qwen3-1.7B \
        --save-dir /path/to/save \
        --eos-token '<|endoftext|>'
"""

import argparse
import os

from transformers import AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Modify a tokenizer's eos_token for olmo template compatibility.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path (e.g., Qwen/Qwen3-1.7B)")
    parser.add_argument("--save-dir", required=True, help="Directory to save the modified tokenizer + config")
    parser.add_argument("--eos-token", default="<|endoftext|>", help="New eos_token to set (default: <|endoftext|>)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load and modify tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    old_eos = tokenizer.eos_token
    old_eos_id = tokenizer.eos_token_id
    tokenizer.eos_token = args.eos_token
    tokenizer.save_pretrained(args.save_dir)

    # Save model config (needed by get_tokenizer_tulu_v2_2 which calls AutoConfig.from_pretrained)
    config = AutoConfig.from_pretrained(args.model)
    config.save_pretrained(args.save_dir)

    print(f"Modified tokenizer saved to {args.save_dir}")
    print(f"  eos_token: {old_eos} (id: {old_eos_id}) -> {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    print(f"  pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"  vocab_size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
