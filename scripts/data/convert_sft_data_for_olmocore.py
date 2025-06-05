"""
This script converts SFT datasets to OLMoCore format. OLMoCore has a more efficient
implementation of the OLMo models (espeically for MoE), and so it can be preferable
to use it for training on next-token prediction tasks (e.g. SFT).

OLMoCore accepts data in numpy mmap format. One file is for the input tokens, one for the labels, and one for the attention mask.

Usage:
    python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
        --chat_template_name tulu \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
        --max_seq_length 4096 \
        --output_dir /weka/oe-adapt-default/tylerr/tulu-3-sft-olmo-2-mixture-0225-olmocore
"""

import argparse
import os
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
        default="allenai/OLMo-2-1124-7B",
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
        default=["allenai/tulu-3-sft-olmo-2-mixture-0225", "1.0"],
        help="Dataset mixer list [dataset_name, weight]",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=4096, help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=0,
        help="Number of examples to process for debugging. 0 means process all examples.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
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

    # TODO: improve configurability of transform factory
    transform_functions_and_args = [
        ("sft_tulu_tokenize_and_truncate_v1", {"max_seq_length": args.max_seq_length}),
        ("sft_tulu_filter_v1", {}),
    ]

    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=[args.dataset_split],
        tc=tc,
        dataset_transform_fn=[func for func, _ in transform_functions_and_args],
        transform_fn_args=[args for _, args in transform_functions_and_args],
        target_columns=TOKENIZED_SFT_DATASET_KEYS,
    )

    if args.visualize:
        print("Visualizing first example...")
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tc.tokenizer)
        print("Labels:", train_dataset[0][LABELS_KEY])
        print("Attention mask:", train_dataset[0][ATTENTION_MASK_KEY])

    if args.num_examples > 0:
        print(f"Selecting {args.num_examples} examples for debugging")
        train_dataset = train_dataset.select(range(args.num_examples))

    print("Collecting tokens from dataset...")
    token_ids = []
    labels = []
    attention_mask = []
    sample: Mapping[str, Any]
    for i, sample in enumerate(train_dataset):
        if i % 5000 == 0:
            print(f"Processing sample {i}/{len(train_dataset)}")
        token_ids.extend(sample[INPUT_IDS_KEY])
        labels.extend(sample[LABELS_KEY])
        attention_mask.extend(sample[ATTENTION_MASK_KEY])

    print(f"Total tokens: {len(token_ids)}")
    print(f"Maximum token ID: {max(token_ids)}")
    print("Writing data to numpy files...")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    def write_memmap_chunked(base_filename, data, dtype, max_size_gb=2):
        """Write data to multiple memmap files if size exceeds max_size_gb."""
        # Calculate size in bytes
        item_size = np.dtype(dtype).itemsize
        total_size_bytes = len(data) * item_size
        max_size_bytes = max_size_gb * 1024**3

        if total_size_bytes <= max_size_bytes:  # record in single file
            mmap = np.memmap(
                f"{base_filename}.npy", mode="w+", dtype=dtype, shape=(len(data),)
            )
            mmap[:] = data
            mmap.flush()
            print(f"Written {base_filename}.npy ({total_size_bytes / 1024**3:.2f} GB)")
            return [mmap]
        else:  # record in multiple files (if too large)
            chunk_size = max_size_bytes // item_size
            chunks = []
            for i in range(0, len(data), chunk_size):
                chunk_data = data[i : i + chunk_size]
                filename = f"{base_filename}_part_{i // chunk_size:04d}.npy"
                mmap = np.memmap(
                    filename, mode="w+", dtype=dtype, shape=(len(chunk_data),)
                )
                mmap[:] = chunk_data
                mmap.flush()
                chunks.append(mmap)
                print(
                    f"Written {filename} ({len(chunk_data) * item_size / 1024**3:.2f} GB)"
                )
            return chunks

    write_memmap_chunked(f"{args.output_dir}/token_ids", token_ids, np.uint32)
    write_memmap_chunked(f"{args.output_dir}/labels", labels, np.int32)
    write_memmap_chunked(f"{args.output_dir}/attention_mask", attention_mask, np.int32)
    print("Data conversion completed successfully")


if __name__ == "__main__":
    main()
