"""
This script converts SFT datasets to OLMoCore format. OLMoCore has a more efficient
implementation of the OLMo models (espeically for MoE), and so it can be preferable
to use it for training on next-token prediction tasks (e.g. SFT).

OLMoCore accepts data in numpy mmap format. One file is for the input tokens, one for the labels, and one for the attention mask.

Usage:
    python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path meta-llama/Llama-3.1-8B \
        --chat_template_name tulu \
        --dataset_mixer_list allenai/tulu-3-sft-personas-algebra 1.0 \
        --max_seq_length 1024 \
        --output_prefix oi
"""

import argparse
from collections.abc import Mapping
from typing import Any

import numpy as np

from open_instruct.dataset_transformation import (
    ATTENTION_MASK_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data for OLMoCore format")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Tokenizer name or path",
    )
    parser.add_argument(
        "--chat_template_name", type=str, default="tulu", help="Chat template name"
    )
    parser.add_argument(
        "--add_bos", action="store_true", default=True, help="Add BOS token"
    )
    parser.add_argument(
        "--dataset_mixer_list",
        type=str,
        nargs=2,
        default=["allenai/tulu-3-sft-personas-algebra", "1.0"],
        help="Dataset mixer list [dataset_name, weight]",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=0,
        help="Number of examples to process for debugging. 0 means process all examples.",
    )
    parser.add_argument(
        "--output_prefix", type=str, default="oi", help="Output file prefix"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize first token sequence"
    )

    args = parser.parse_args()

    tc = TokenizerConfig(
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        chat_template_name=args.chat_template_name,
        add_bos=args.add_bos,
    )

    transform_functions_and_args = [
        ("sft_tulu_tokenize_and_truncate_v1", {"max_seq_length": args.max_seq_length}),
        ("sft_tulu_filter_v1", {}),
    ]

    transform_functions = [func for func, _ in transform_functions_and_args]
    transform_function_args = [args for _, args in transform_functions_and_args]

    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=[args.dataset_split],
        tc=tc,
        dataset_transform_fn=transform_functions,
        transform_fn_args=transform_function_args,
        target_columns=TOKENIZED_SFT_DATASET_KEYS,
    )

    if args.visualize:
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tc.tokenizer)
        print("Labels:", train_dataset[0][LABELS_KEY])

    if args.num_examples > 0:
        print(f"Selecting {args.num_examples} examples for debugging")
        train_dataset = train_dataset.select(range(args.num_examples))

    print("Collecting tokens from dataset...")
    token_ids = []
    labels = []
    attention_mask = []
    sample: Mapping[str, Any]
    for sample in train_dataset:
        token_ids.extend(sample[INPUT_IDS_KEY])
        labels.extend(sample[LABELS_KEY])
        attention_mask.extend(sample[ATTENTION_MASK_KEY])

    print(f"Total tokens: {len(token_ids)}")
    print(f"Maximum token ID: {max(token_ids)}")

    print("Writing data to numpy files...")

    def write_memmap(filename, data, dtype):
        mmap = np.memmap(filename, mode="w+", dtype=dtype, shape=(len(data),))
        mmap[:] = data
        mmap.flush()
        return mmap

    write_memmap(f"{args.output_prefix}_token_ids.npy", token_ids, np.uint32)
    write_memmap(f"{args.output_prefix}_labels.npy", labels, np.int32)
    write_memmap(f"{args.output_prefix}_attention_mask.npy", attention_mask, np.int32)
    print("Data conversion completed successfully")


if __name__ == "__main__":
    main()
