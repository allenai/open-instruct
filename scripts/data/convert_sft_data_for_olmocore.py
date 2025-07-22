"""
This script converts SFT datasets to OLMoCore format. OLMoCore has a more efficient
implementation of the OLMo models (espeically for MoE), and so it can be preferable
to use it for training on next-token prediction tasks (e.g. SFT).

OLMoCore accepts data in numpy mmap format. One file is for the input tokens and one for the labels mask.

Usage:
    python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
        --output_dir ./data/tulu-3-sft-olmo-2-mixture-0225-olmocore \
        --chat_template_name olmo

Ai2 Internal Usage:
    gantry run --cluster ai2/neptune-cirrascale --timeout -1 -y --budget ai2/oe-training --workspace ai2/jacobm \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
        --weka=oe-training-default:/weka/oe-training-default \
        --env-secret HF_TOKEN=HF_TOKEN \
        --gpus 1 \
        --priority high \
        -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
        --output_dir /weka/oe-training-default/ai2-llm/tylerr/data/sft/tulu-3-sft-olmo-2-mixture-0225-olmocore \
        --visualize True \
        --chat_template_name olmo \
        --max_seq_length 16384

NOTE: allenai/OLMo-2-1124-7B tokenizer is the same as allenai/dolma2-tokenizer, but allenai/OLMo-2-1124-7B
has additional metadata required for this script.

Recommendations:
  * Set max_seq_length, and use the same length you use during SFT
"""

import gzip
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from tqdm import tqdm

from open_instruct.dataset_transformation import (
    ATTENTION_MASK_KEY,
    DATASET_ORIGIN_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE,
    TokenizerConfig,
    get_cached_dataset_tulu_with_statistics,
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
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )

    """The columns to use for the dataset."""
    dataset_target_columns: List[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE)

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

    """Only write the tokenizer config to the output directory"""
    tokenizer_config_only: bool = field(default=False)


def main(args: ConvertSFTDataArguments, tc: TokenizerConfig):
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.dataset_local_cache_dir = beaker_cache_dir

    print("Verify these values match the tokenizer config used in Olmo-core:")
    print(f"Tokenizer vocab_size: {tc.tokenizer.vocab_size}")
    print(f"Tokenizer bos_token_id: {tc.tokenizer.bos_token_id}")
    print(f"Tokenizer pad_token_id: {tc.tokenizer.pad_token_id}")
    print(f"Tokenizer eos_token_id: {tc.tokenizer.eos_token_id}")
    print(f"Tokenizer chat_template: {tc.tokenizer.chat_template}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    print(f"Saving tokenizer to {tokenizer_output_dir}...")
    tc.tokenizer.save_pretrained(tokenizer_output_dir)

    # Check if chat_template.jinja exists and add it to tokenizer_config.json
    chat_template_path = os.path.join(tokenizer_output_dir, "chat_template.jinja")
    tokenizer_config_path = os.path.join(tokenizer_output_dir, "tokenizer_config.json")
    if os.path.exists(chat_template_path) and os.path.exists(tokenizer_config_path):
        with open(chat_template_path, "r") as f:
            chat_template_content = f.read()
        with open(tokenizer_config_path, "r") as f:
            tokenizer_config = json.load(f)
        if "chat_template" not in tokenizer_config:
            tokenizer_config["chat_template"] = chat_template_content
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            print(f"Added chat_template from {chat_template_path} to tokenizer_config.json")

    print("Tokenizer saved successfully!")

    if args.tokenizer_config_only:
        return

    # TODO: improve configurability of transform factory
    transform_functions_and_args = [
        ("sft_tulu_tokenize_and_truncate_v1", {"max_seq_length": args.max_seq_length}),
        ("sft_tulu_filter_v1", {}),  # remove examples that don't have any labels
    ]

    result = get_cached_dataset_tulu_with_statistics(
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
        return_statistics=True,
    )
    
    # Unpack the result
    train_dataset, dataset_statistics = result

    train_dataset = train_dataset.shuffle()

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
    labels_mask = []
    sample: Mapping[str, Any]
    num_samples_skipped = 0
    document_boundaries = []
    current_position = 0
    
    # Track per-dataset statistics using dataset_source field
    per_dataset_counts = {}
    per_dataset_tokens = {}
    per_dataset_trainable_tokens = {}
    per_dataset_filtered = {}

    for idx, sample in enumerate(tqdm(  # type: ignore
        train_dataset,
        desc="Collecting tokens",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}\n",  # better printing in beaker
        mininterval=10.0,
    )):
        sample_length = len(sample[INPUT_IDS_KEY])
        sample_tokens = sample[INPUT_IDS_KEY]
        sample_labels = sample[LABELS_KEY]
        dataset_source = sample.get(DATASET_ORIGIN_KEY, "unknown")
        
        # Initialize counters for new datasets
        if dataset_source not in per_dataset_counts:
            per_dataset_counts[dataset_source] = 0
            per_dataset_tokens[dataset_source] = 0
            per_dataset_trainable_tokens[dataset_source] = 0
            per_dataset_filtered[dataset_source] = 0
        
        # Update per-dataset statistics
        per_dataset_counts[dataset_source] += 1
        per_dataset_tokens[dataset_source] += sample_length
        trainable_tokens_in_sample = sum(1 for label in sample_labels if label != -100)
        per_dataset_trainable_tokens[dataset_source] += trainable_tokens_in_sample
        
        token_ids.extend(sample_tokens)
        labels_mask.extend([1 if label != -100 else 0 for label in sample_labels])

        # Record document boundary (start, end)
        document_boundaries.append((current_position, current_position + sample_length))
        current_position += sample_length

        if all(label == -100 for label in sample_labels):
            num_samples_skipped += 1
            per_dataset_filtered[dataset_source] += 1

        # Assert that attention mask is all 1s
        assert all(mask == 1 for mask in sample[ATTENTION_MASK_KEY]), (
            f"Expected all attention mask values to be 1, but found: {sample[ATTENTION_MASK_KEY]}"
        )

    # Calculate final statistics
    total_instances = len(train_dataset)
    total_tokens = len(token_ids)
    total_trainable_tokens = sum(labels_mask)
    
    print(f"Total sequences: {total_instances}")
    print(f"Total tokens: {total_tokens}")
    print(f"Maximum token ID: {max(token_ids)}")
    print(f"Labels mask sum (trainable tokens): {total_trainable_tokens}")
    print("Writing data to numpy files...")
    print(f"Number of samples that should be skipped: {num_samples_skipped}")

    def write_memmap_chunked(base_filename, data, dtype, max_size_gb=1):
        """Write data to multiple memmap files if size exceeds max_size_gb."""
        # Calculate size in bytes
        item_size = np.dtype(dtype).itemsize
        max_size_bytes = max_size_gb * 1024**3

        chunk_size = max_size_bytes // item_size
        chunks = []
        chunk_boundaries = []

        for i in range(0, len(data), chunk_size):
            chunk_data = data[i : i + chunk_size]
            filename = f"{base_filename}_part_{i // chunk_size:04d}.npy"
            mmap = np.memmap(filename, mode="w+", dtype=dtype, shape=(len(chunk_data),))
            mmap[:] = chunk_data
            mmap.flush()
            chunks.append(mmap)
            chunk_boundaries.append((i, i + len(chunk_data)))
            print(f"Written {filename} ({len(chunk_data) * item_size / 1024**3:.2f} GB)")

        return chunks, chunk_boundaries

    def write_metadata_for_chunks(base_filename, document_boundaries, chunk_boundaries):
        """Write metadata files for each chunk with document boundaries."""

        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            metadata_filename = f"{base_filename}_part_{chunk_idx:04d}.csv.gz"

            with gzip.open(metadata_filename, "wt") as f:
                # Find all documents that overlap with this chunk
                for doc_start, doc_end in document_boundaries:
                    # Check if document overlaps with chunk
                    if doc_end > chunk_start and doc_start < chunk_end:
                        # Adjust boundaries relative to chunk start
                        adjusted_start = max(0, doc_start - chunk_start)
                        adjusted_end = min(chunk_end - chunk_start, doc_end - chunk_start)

                        # Only write if there's actual content in this chunk
                        if adjusted_end > adjusted_start:
                            f.write(f"{adjusted_start},{adjusted_end}\n")

            print(f"Written metadata {metadata_filename}")

    # Choose dtype based on vocab size - Olmo-core does the
    # same operation to infer the dtype of the token_ids array.
    vocab_size = tc.tokenizer.vocab_size
    token_dtype = None
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dtype).max:
            token_dtype = dtype
            print(f"Using dtype '{dtype}' for token_ids based on vocab size {vocab_size}")
            break
    if token_dtype is None:
        raise ValueError(f"Vocab size {vocab_size} is too big for any numpy integer dtype!")

    print(f"Writing converted data to {output_dir}")
    _, token_chunk_boundaries = write_memmap_chunked(f"{output_dir}/token_ids", token_ids, token_dtype)
    write_metadata_for_chunks(f"{output_dir}/token_ids", document_boundaries, token_chunk_boundaries)

    # Write labels_mask using the same chunk boundaries as token_ids
    for i, (start, end) in enumerate(token_chunk_boundaries):
        chunk_data = labels_mask[start:end]
        filename = f"{output_dir}/labels_mask_part_{i:04d}.npy"
        mmap = np.memmap(filename, mode="w+", dtype=np.bool_, shape=(len(chunk_data),))
        mmap[:] = chunk_data
        mmap.flush()
        print(f"Written {filename} ({len(chunk_data) * np.dtype(np.bool_).itemsize / 1024**3:.2f} GB)")

    print("Data conversion completed successfully!")
    
    # Write dataset statistics
    write_dataset_statistics(
        output_dir=output_dir,
        dataset_statistics=dataset_statistics,
        total_instances=total_instances,
        total_tokens=total_tokens,
        total_trainable_tokens=total_trainable_tokens,
        num_samples_skipped=num_samples_skipped,
        tokenizer_name=tc.tokenizer_name_or_path,
        max_seq_length=args.max_seq_length,
        chat_template_name=tc.chat_template_name,
        per_dataset_counts=per_dataset_counts,
        per_dataset_tokens=per_dataset_tokens,
        per_dataset_trainable_tokens=per_dataset_trainable_tokens,
        per_dataset_filtered=per_dataset_filtered,
    )


def write_dataset_statistics(
    output_dir: str,
    dataset_statistics: Dict[str, Any],
    total_instances: int,
    total_tokens: int,
    total_trainable_tokens: int,
    num_samples_skipped: int,
    tokenizer_name: str,
    max_seq_length: Optional[int],
    chat_template_name: Optional[str],
    per_dataset_counts: Dict[str, int],
    per_dataset_tokens: Dict[str, int],
    per_dataset_trainable_tokens: Dict[str, int],
    per_dataset_filtered: Dict[str, int],
):
    """Write dataset statistics to both text and JSON files."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Merge pre-transformation stats with post-shuffle actual counts
    merged_stats = []
    pre_transform_stats = {stat["dataset_name"]: stat for stat in dataset_statistics.get("per_dataset_stats", [])}
    
    for dataset_name in per_dataset_counts:
        pre_stat = pre_transform_stats.get(dataset_name, {})
        merged_stat = {
            "dataset_name": dataset_name,
            "dataset_split": pre_stat.get("dataset_split", "unknown"),
            "initial_instances": pre_stat.get("initial_instances", "N/A"),
            "instances_after_transformation": pre_stat.get("final_instances", "N/A"),
            "instances_filtered_during_transformation": pre_stat.get("instances_filtered", "N/A"),
            "frac_or_num_samples": pre_stat.get("frac_or_num_samples"),
            # Upsampling information
            "original_dataset_size": pre_stat.get("original_dataset_size"),
            "is_upsampled": pre_stat.get("is_upsampled", False),
            "upsampling_factor": pre_stat.get("upsampling_factor", 1.0),
            # Post-shuffle actual statistics
            "final_instances_in_output": per_dataset_counts[dataset_name],
            "final_tokens_in_output": per_dataset_tokens[dataset_name],
            "final_trainable_tokens_in_output": per_dataset_trainable_tokens[dataset_name],
            "instances_filtered_after_tokenization": per_dataset_filtered[dataset_name],
            "avg_tokens_per_instance": per_dataset_tokens[dataset_name] / per_dataset_counts[dataset_name] if per_dataset_counts[dataset_name] > 0 else 0,
            "percentage_of_total_tokens": (per_dataset_tokens[dataset_name] / total_tokens * 100) if total_tokens > 0 else 0,
            "percentage_of_total_instances": (per_dataset_counts[dataset_name] / total_instances * 100) if total_instances > 0 else 0,
        }
        merged_stats.append(merged_stat)
    
    # Prepare statistics data
    stats_data = {
        "timestamp": timestamp,
        "output_directory": output_dir,
        "configuration": {
            "tokenizer": tokenizer_name,
            "max_sequence_length": max_seq_length,
            "chat_template": chat_template_name,
        },
        "per_dataset_statistics": merged_stats,
        "overall_statistics": {
            "total_datasets": len(per_dataset_counts),
            "total_instances": total_instances,
            "total_tokens": total_tokens,
            "trainable_tokens": total_trainable_tokens,
            "trainable_percentage": (total_trainable_tokens / total_tokens * 100) if total_tokens > 0 else 0,
            "instances_filtered": num_samples_skipped,
            "average_sequence_length": total_tokens / total_instances if total_instances > 0 else 0,
        }
    }
    
    # Write JSON file
    json_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"Written dataset statistics to {json_path}")
    
    # Write human-readable text file
    text_path = os.path.join(output_dir, "dataset_statistics.txt")
    with open(text_path, "w") as f:
        f.write("Dataset Statistics Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 40 + "\n")
        f.write(f"- Tokenizer: {tokenizer_name}\n")
        f.write(f"- Max Sequence Length: {max_seq_length}\n")
        f.write(f"- Chat Template: {chat_template_name}\n\n")
        
        f.write("Per-Dataset Statistics:\n")
        f.write("=" * 80 + "\n")
        
        for stat in stats_data["per_dataset_statistics"]:
            f.write(f"\nDataset: {stat['dataset_name']}\n")
            f.write(f"- Split: {stat['dataset_split']}\n")
            
            # Pre-transformation statistics
            f.write("\nPre-transformation:\n")
            f.write(f"  - Instances loaded: {stat.get('initial_instances', 'N/A')}\n")
            f.write(f"  - Instances after transformation: {stat.get('instances_after_transformation', 'N/A')}\n")
            f.write(f"  - Instances filtered during transformation: {stat.get('instances_filtered_during_transformation', 'N/A')}\n")
            
            if stat.get('frac_or_num_samples') is not None:
                if isinstance(stat['frac_or_num_samples'], float):
                    f.write(f"  - Sampling fraction: {stat['frac_or_num_samples']}\n")
                else:
                    f.write(f"  - Sample count: {stat['frac_or_num_samples']}\n")
            
            # Show upsampling information if applicable
            if stat.get('is_upsampled', False):
                f.write(f"  - Original dataset size: {stat.get('original_dataset_size', 'N/A')}\n")
                f.write(f"  - Upsampling factor: {stat.get('upsampling_factor', 1.0):.2f}x\n")
                f.write(f"  - Upsampled to: {stat.get('instances_after_transformation', 'N/A')} instances\n")
            
            # Post-shuffle statistics (actual output)
            f.write("\nFinal output statistics (after shuffling):\n")
            f.write(f"  - Instances in output: {stat['final_instances_in_output']:,}\n")
            f.write(f"  - Total tokens: {stat['final_tokens_in_output']:,}\n")
            f.write(f"  - Trainable tokens: {stat['final_trainable_tokens_in_output']:,}\n")
            f.write(f"  - Instances with no labels: {stat['instances_filtered_after_tokenization']}\n")
            f.write(f"  - Average tokens per instance: {stat['avg_tokens_per_instance']:.1f}\n")
            f.write(f"  - Percentage of total tokens: {stat['percentage_of_total_tokens']:.1f}%\n")
            f.write(f"  - Percentage of total instances: {stat['percentage_of_total_instances']:.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Overall Statistics:\n")
        f.write("=" * 80 + "\n")
        f.write(f"- Total datasets: {stats_data['overall_statistics']['total_datasets']}\n")
        f.write(f"- Total instances: {stats_data['overall_statistics']['total_instances']:,}\n")
        f.write(f"- Total tokens: {stats_data['overall_statistics']['total_tokens']:,}\n")
        f.write(f"- Trainable tokens: {stats_data['overall_statistics']['trainable_tokens']:,} ")
        f.write(f"({stats_data['overall_statistics']['trainable_percentage']:.1f}%)\n")
        f.write(f"- Instances filtered out: {stats_data['overall_statistics']['instances_filtered']}\n")
        f.write(f"- Average sequence length: {stats_data['overall_statistics']['average_sequence_length']:.1f}\n")
    
    print(f"Written human-readable statistics to {text_path}")


if __name__ == "__main__":
    parser = ArgumentParserPlus((ConvertSFTDataArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
