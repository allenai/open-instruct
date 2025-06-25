"""
This script converts SFT datasets to OLMoCore format. OLMoCore has a more efficient
implementation of the OLMo models (espeically for MoE), and so it can be preferable
to use it for training on next-token prediction tasks (e.g. SFT).

OLMoCore accepts data in numpy mmap format. One file is for the input tokens, one for the labels, and one for the attention mask.

Usage:
    python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
        --add_bos \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
        --output_dir ./data/tulu-3-sft-olmo-2-mixture-0225-olmocore

Ai2 Internal Usage:
    gantry run --cluster ai2/phobos-cirrascale --timeout -1 -y --budget ai2/oe-training \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        -- \
        /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
        --add_bos \
        --output_dir /weka/oe-training-default/ai2-llm/tylerr/data/sft/tulu-3-sft-olmo-2-mixture-0225-olmocore

NOTE: allenai/OLMo-2-1124-7B tokenizer is the same as allenai/dolma2-tokenizer, but allenai/OLMo-2-1124-7B
has additional metadata required for this script.

Recommendations:
 * Don't use max-seq-length, keep full sequences and allow Olmo-core to truncate if needed.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

import numpy as np
from tqdm import tqdm

from open_instruct.dataset_transformation import (
    ATTENTION_MASK_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.utils import ArgumentParserPlus, is_beaker_job


@dataclass
class ConvertSFTDataArguments:
    """
    Arguments for converting SFT data to OLMoCore format.
    """

    """Output directory"""
    output_dir: str = field()

    """The name of the dataset to use (via the datasets library)."""
    dataset_name: Optional[str] = field(default=None)

    """A dictionary of datasets (local or HF) to sample from."""
    dataset_mixer: Optional[dict] = field(default=None)

    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-olmo-2-mixture-0225", "1.0"])

    """The dataset splits to use for training"""
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])

    """The list of transform functions to apply to the dataset."""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: [
            "sft_tulu_tokenize_and_truncate_v1",
            "sft_tulu_filter_v1",
        ]
    )

    """The columns to use for the dataset."""
    dataset_target_columns: List[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)

    """The mode to use for caching the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"

    """The directory to save the local dataset cache to."""
    dataset_local_cache_dir: str = "local_dataset_cache"

    """The hash of the dataset configuration."""
    dataset_config_hash: Optional[str] = None

    """Whether to skip the cache."""
    dataset_skip_cache: bool = False

    """Maximum sequence length. If not provided, no truncation will be performed."""
    max_seq_length: Optional[int] = field(default=None)

    """Number of examples to process for debugging. 0 means process all examples."""
    num_examples: int = field(default=0)

    """Visualize first token sequence"""
    visualize: bool = field(default=False)


def main(args: ConvertSFTDataArguments, tc: TokenizerConfig):
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.dataset_local_cache_dir = beaker_cache_dir

    # TODO: improve configurability of transform factory
    transform_functions_and_args = [
        (
            "sft_tulu_tokenize_and_truncate_v1",
            {"max_seq_length": args.max_seq_length},
        ),
        ("sft_tulu_filter_v1", {}),  # remove examples that don't have any labels
    ]

    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=[func for func, _ in transform_functions_and_args],
        transform_fn_args=[args for _, args in transform_functions_and_args],
        target_columns=args.dataset_target_columns,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
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
    for sample in tqdm(train_dataset, desc="Collecting tokens"):  # type: ignore
        token_ids.extend(sample[INPUT_IDS_KEY])
        labels.extend(sample[LABELS_KEY])
        attention_mask.extend(sample[ATTENTION_MASK_KEY])

    print(f"Total sequences: {len(train_dataset)}")
    print(f"Total tokens: {len(token_ids)}")
    print(f"Maximum token ID: {max(token_ids)}")
    print("Writing data to numpy files...")

    # Create output directory with tokenizer name
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    def write_memmap_chunked(base_filename, data, dtype, max_size_gb=1):
        """Write data to multiple memmap files if size exceeds max_size_gb."""
        # Calculate size in bytes
        item_size = np.dtype(dtype).itemsize
        total_size_bytes = len(data) * item_size
        max_size_bytes = max_size_gb * 1024**3

        if total_size_bytes <= max_size_bytes:  # record in single file
            mmap = np.memmap(f"{base_filename}.npy", mode="w+", dtype=dtype, shape=(len(data),))
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
                mmap = np.memmap(filename, mode="w+", dtype=dtype, shape=(len(chunk_data),))
                mmap[:] = chunk_data
                mmap.flush()
                chunks.append(mmap)
                print(f"Written {filename} ({len(chunk_data) * item_size / 1024**3:.2f} GB)")
            return chunks

    print(f"Writing converted data to {output_dir}")
    write_memmap_chunked(f"{output_dir}/token_ids", token_ids, np.uint32)
    write_memmap_chunked(f"{output_dir}/labels", labels, np.int32)
    write_memmap_chunked(f"{output_dir}/attention_mask", attention_mask, np.int32)
    print("Data conversion completed successfully!")

    tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    print(f"Saving tokenizer to {tokenizer_output_dir}...")
    tc.tokenizer.save_pretrained(tokenizer_output_dir)
    print("Tokenizer saved successfully!")

    print("Verify these values match the tokenizer config used in Olmo-core:")
    print(f"Tokenizer vocab_size: {tc.tokenizer.vocab_size}")
    print(f"Tokenizer bos_token_id: {tc.tokenizer.bos_token_id}")
    print(f"Tokenizer pad_token_id: {tc.tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token_id: {tc.tokenizer.eos_token_id}")


if __name__ == "__main__":
    parser = ArgumentParserPlus((ConvertSFTDataArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
