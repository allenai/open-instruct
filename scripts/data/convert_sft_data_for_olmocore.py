# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "beaker-py>=1.32.2,<2.0",
#     "datasets>=4.0.0",
#     "numpy<2",
#     "ray[default]>=2.44.1",
#     "rich>=13.7.0",
#     "tqdm",
#     "transformers>=4.52.4",
#     "torch>=2.7.0,<2.8",
# ]
# ///

"""
This script converts SFT datasets to OLMoCore format. OLMoCore has a more efficient
implementation of the OLMo models (espeically for MoE), and so it can be preferable
to use it for training on next-token prediction tasks (e.g. SFT).

OLMoCore accepts data in numpy mmap format. One file is for the input tokens and one for the labels mask.

## Usage:
    ```bash
    python scripts/data/convert_sft_data_for_olmocore.py \
        --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture-0225 1.0 \
        --output_dir ./data/tulu-3-sft-olmo-2-mixture-0225-olmocore \
        --chat_template_name olmo
    ```

## Ai2 Internal Usage (requires `gantry>=3`):
    ```bash
    gantry run \
        --workspace ai2/jacobm \
        --budget ai2/oe-base \
        --priority normal \
        --cluster ai2/neptune --gpus 1 \
        --weka=oe-training-default:/weka/oe-training-default \
        --task-name convert-sft-data-for-olmocore --yes \
        --env-secret HF_TOKEN=HF_TOKEN \
        --install "echo 'do nothing'" \
        -- uv run --script scripts/data/convert_sft_data_for_olmocore.py \
            --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
            --tokenizer_name_or_path allenai/OLMo-2-1124-7B \
            --output_dir /weka/oe-training-default/ai2-llm/tylerr/data/sft/tulu-3-sft-olmo-2-mixture-0225-olmocore-test1 \
            --visualize True \
            --chat_template_name olmo \
            --max_seq_length 16384
    ```

    Dependencies for this script when run with `uv` are declared at the top of the file. `uv` will
    automatically install them *and not the project dependencies*.

    Add `--show-logs` to stream the logs to the terminal. By default `gantry run` will detach the job.

NOTE: allenai/OLMo-2-1124-7B tokenizer is the same as allenai/dolma2-tokenizer, but allenai/OLMo-2-1124-7B
has additional metadata required for this script.

Recommendations:
  * Set max_seq_length, and use the same length you use during SFT
"""

import gzip
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np
from tqdm import tqdm


def save_checkpoint_metadata(output_dir: str, checkpoint_data: dict[str, Any]) -> None:
    """Save checkpoint metadata to disk atomically (excludes token data)."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoint_data, f)
    os.rename(tmp_path, checkpoint_path)  # Atomic on POSIX


def load_checkpoint(output_dir: str) -> Optional[dict[str, Any]]:
    """Load checkpoint metadata from disk if it exists."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return None


def remove_checkpoint(output_dir: str) -> None:
    """Remove checkpoint files after successful completion."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"Removed checkpoint file: {checkpoint_path}")
    for partial_file in ["_partial_tokens.bin", "_partial_labels.bin", "_partial_boundaries.bin"]:
        partial_path = os.path.join(output_dir, partial_file)
        if os.path.exists(partial_path):
            os.remove(partial_path)
            logger.info(f"Removed partial file: {partial_path}")


def append_tokens_to_disk(
    output_dir: str,
    token_ids: list[int],
    labels_mask: list[int],
    document_boundaries: list[tuple[int, int]],
    token_dtype: np.dtype,
) -> None:
    """Append tokens to binary files (append-only, no loading required)."""
    with open(os.path.join(output_dir, "_partial_tokens.bin"), "ab") as f:
        f.write(np.array(token_ids, dtype=token_dtype).tobytes())
    with open(os.path.join(output_dir, "_partial_labels.bin"), "ab") as f:
        f.write(np.array(labels_mask, dtype=np.bool_).tobytes())
    with open(os.path.join(output_dir, "_partial_boundaries.bin"), "ab") as f:
        f.write(np.array(document_boundaries, dtype=np.int64).tobytes())


def load_progress(output_dir: str, token_dtype: np.dtype) -> tuple[int, list[tuple[int, int]]]:
    """Load progress info from partial files (token count and boundaries, not full data)."""
    tokens_path = os.path.join(output_dir, "_partial_tokens.bin")
    boundaries_path = os.path.join(output_dir, "_partial_boundaries.bin")

    if os.path.exists(tokens_path):
        token_count = os.path.getsize(tokens_path) // np.dtype(token_dtype).itemsize
        boundaries_data = np.fromfile(boundaries_path, dtype=np.int64).reshape(-1, 2)
        boundaries = [tuple(b) for b in boundaries_data.tolist()]
        return token_count, boundaries
    return 0, []

from open_instruct.dataset_transformation import (
    ATTENTION_MASK_KEY,
    DATASET_ORIGIN_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE,
    TokenizerConfig,
    get_cached_dataset_tulu_with_statistics,
    remove_dataset_source_field,
    visualize_token,
)
from open_instruct import logger_utils, utils
from open_instruct.utils import ArgumentParserPlus, is_beaker_job

logger = logger_utils.setup_logger(__name__)


@dataclass
class ConvertSFTDataArguments:
    """
    Arguments for converting SFT data to OLMoCore format.
    """

    """Output directory"""
    output_dir: str = field()

    """The name of the dataset to use (via the datasets library)."""
    dataset_name: str | None = field(default=None)

    """A dictionary of datasets (local or HF) to sample from."""
    dataset_mixer: dict | None = field(default=None)

    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-olmo-2-mixture-0225", "1.0"])

    """The dataset splits to use for training"""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])

    """The list of transform functions to apply to the dataset."""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )

    """The columns to use for the dataset."""
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE)

    """The mode to use for caching the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"

    """The directory to save the local dataset cache to."""
    dataset_local_cache_dir: str = "local_dataset_cache"

    """The hash of the dataset configuration."""
    dataset_config_hash: str | None = None

    """Whether to skip the cache."""
    dataset_skip_cache: bool = False

    """Maximum sequence length. If not provided, no truncation will be performed."""
    max_seq_length: int | None = field(default=None)

    """Number of examples to process for debugging. 0 means process all examples."""
    num_examples: int = field(default=0)

    """Visualize first token sequence"""
    visualize: bool = field(default=False)

    """Only write the tokenizer config to the output directory"""
    tokenizer_config_only: bool = field(default=False)

    """Resume from checkpoint if available"""
    resume: bool = field(default=False)

    """Checkpoint save interval (number of samples)"""
    checkpoint_interval: int = field(default=100_000)

    """Shuffle seed for reproducible dataset ordering"""
    shuffle_seed: int = field(default=42)


def main(args: ConvertSFTDataArguments, tc: TokenizerConfig):
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.dataset_local_cache_dir = beaker_cache_dir

    logger.info("Verify these values match the tokenizer config used in Olmo-core:")
    logger.info(f"Tokenizer vocab_size: {tc.tokenizer.vocab_size}")
    logger.info(f"Tokenizer bos_token_id: {tc.tokenizer.bos_token_id}")
    logger.info(f"Tokenizer pad_token_id: {tc.tokenizer.pad_token_id}")
    logger.info(f"Tokenizer eos_token_id: {tc.tokenizer.eos_token_id}")
    logger.info(f"Tokenizer chat_template: {tc.tokenizer.chat_template}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    logger.info(f"Saving tokenizer to {tokenizer_output_dir}...")
    tc.tokenizer.save_pretrained(tokenizer_output_dir)

    # Check if chat_template.jinja exists and add it to tokenizer_config.json
    chat_template_path = os.path.join(tokenizer_output_dir, "chat_template.jinja")
    tokenizer_config_path = os.path.join(tokenizer_output_dir, "tokenizer_config.json")
    if os.path.exists(chat_template_path) and os.path.exists(tokenizer_config_path):
        with open(chat_template_path) as f:
            chat_template_content = f.read()
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        if "chat_template" not in tokenizer_config:
            tokenizer_config["chat_template"] = chat_template_content
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            logger.info(f"Added chat_template from {chat_template_path} to tokenizer_config.json")

    logger.info("Tokenizer saved successfully!")

    if args.tokenizer_config_only:
        return

    # TODO: improve configurability of transform factory
    transform_functions_and_args = [
        ("sft_tulu_tokenize_and_truncate_v1", {"max_seq_length": args.max_seq_length}),
        ("sft_tulu_filter_v1", {}),  # remove examples that don't have any labels
    ]

    train_dataset, dataset_statistics = get_cached_dataset_tulu_with_statistics(
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
        drop_dataset_source=False,
    )

    # Use fixed seed for reproducible shuffling (required for resume)
    train_dataset = train_dataset.shuffle(seed=args.shuffle_seed)

    if args.visualize:
        logger.info("Visualizing first example...")
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tc.tokenizer)
        logger.info("Labels:", train_dataset[0][LABELS_KEY])
        logger.info("Attention mask:", train_dataset[0][ATTENTION_MASK_KEY])

    if args.num_examples > 0:
        logger.info(f"Selecting {args.num_examples} examples for debugging")
        train_dataset = train_dataset.select(range(args.num_examples))

    # Determine token dtype early (needed for incremental checkpointing)
    vocab_size = tc.tokenizer.vocab_size
    token_dtype = None
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dtype).max:
            token_dtype = dtype
            logger.info(f"Using dtype '{dtype}' for token_ids based on vocab size {vocab_size}")
            break
    if token_dtype is None:
        raise ValueError(f"Vocab size {vocab_size} is too big for any numpy integer dtype!")

    # Initialize state
    start_idx = 0
    current_position = 0
    num_samples_skipped = 0
    per_dataset_counts: dict[str, int] = {}
    per_dataset_tokens: dict[str, int] = {}
    per_dataset_trainable_tokens: dict[str, int] = {}
    per_dataset_filtered: dict[str, int] = {}

    # In-memory buffers (flushed to disk at each checkpoint)
    token_ids: list[int] = []
    labels_mask: list[int] = []
    document_boundaries: list[tuple[int, int]] = []

    # Check for existing checkpoint to resume from
    checkpoint = load_checkpoint(output_dir) if args.resume else None
    if checkpoint:
        start_idx = checkpoint["samples_processed"]
        current_position, _ = load_progress(output_dir, token_dtype)
        num_samples_skipped = checkpoint["num_samples_skipped"]
        per_dataset_counts = checkpoint["per_dataset_counts"]
        per_dataset_tokens = checkpoint["per_dataset_tokens"]
        per_dataset_trainable_tokens = checkpoint["per_dataset_trainable_tokens"]
        per_dataset_filtered = checkpoint["per_dataset_filtered"]
        logger.info(f"=== RESUMING from checkpoint ===")
        logger.info(f"  Samples already processed: {start_idx:,}")
        logger.info(f"  Tokens on disk: {current_position:,}")
        logger.info(f"  Remaining samples: {len(train_dataset) - start_idx:,}")
        logger.info(f"================================")
    elif args.resume:
        logger.info("No checkpoint found, starting from beginning...")

    logger.info("Collecting tokens from dataset...")
    sample: Mapping[str, Any]
    total_samples = len(train_dataset)
    processing_start_time = time.time()
    utils.maybe_update_beaker_description()

    # Skip to resume point efficiently using dataset.select()
    if start_idx > 0:
        train_dataset_iter = train_dataset.select(range(start_idx, total_samples))
    else:
        train_dataset_iter = train_dataset

    for idx, sample in enumerate(
        tqdm(  # type: ignore
            train_dataset_iter,
            desc="Collecting tokens",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}\n",  # better printing in beaker
            mininterval=10.0,
            initial=start_idx,  # Start progress bar from resume point
            total=total_samples,
        ),
        start=start_idx,  # Enumerate from resume point
    ):
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
        assert all(
            mask == 1 for mask in sample[ATTENTION_MASK_KEY]
        ), f"Expected all attention mask values to be 1, but found: {sample[ATTENTION_MASK_KEY]}"

        # Save checkpoint periodically (flush to disk and save metadata)
        if (idx + 1) % args.checkpoint_interval == 0 and idx > start_idx:
            append_tokens_to_disk(output_dir, token_ids, labels_mask, document_boundaries, token_dtype)
            token_ids.clear()
            labels_mask.clear()
            document_boundaries.clear()
            save_checkpoint_metadata(output_dir, {
                "samples_processed": idx + 1,
                "current_position": current_position,
                "num_samples_skipped": num_samples_skipped,
                "per_dataset_counts": per_dataset_counts,
                "per_dataset_tokens": per_dataset_tokens,
                "per_dataset_trainable_tokens": per_dataset_trainable_tokens,
                "per_dataset_filtered": per_dataset_filtered,
            })
            logger.info(f"\nCheckpoint saved at sample {idx + 1:,} ({current_position:,} tokens)")
            utils.maybe_update_beaker_description(
                current_step=idx + 1,
                total_steps=total_samples,
                start_time=processing_start_time,
            )

    train_dataset = remove_dataset_source_field(train_dataset)

    # Flush any remaining tokens in memory to disk
    if token_ids:
        append_tokens_to_disk(output_dir, token_ids, labels_mask, document_boundaries, token_dtype)

    # Load all data from partial files
    tokens_path = os.path.join(output_dir, "_partial_tokens.bin")
    labels_path = os.path.join(output_dir, "_partial_labels.bin")
    boundaries_path = os.path.join(output_dir, "_partial_boundaries.bin")

    all_token_ids = np.fromfile(tokens_path, dtype=token_dtype)
    all_labels_mask = np.fromfile(labels_path, dtype=np.bool_)
    all_boundaries = np.fromfile(boundaries_path, dtype=np.int64).reshape(-1, 2)
    all_document_boundaries = [tuple(b) for b in all_boundaries.tolist()]

    # Calculate final statistics
    total_instances = len(train_dataset)
    total_tokens = len(all_token_ids)
    total_trainable_tokens = int(all_labels_mask.sum())

    logger.info(f"Total sequences: {total_instances}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Maximum token ID: {all_token_ids.max()}")
    logger.info(f"Labels mask sum (trainable tokens): {total_trainable_tokens}")
    logger.info("Writing data to numpy files...")
    logger.info(f"Number of samples that should be skipped: {num_samples_skipped}")

    def write_memmap_chunked(base_filename, data, dtype, max_size_gb=1):
        """Write data to multiple memmap files if size exceeds max_size_gb."""
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
            logger.info(f"Written {filename} ({len(chunk_data) * item_size / 1024**3:.2f} GB)")

        return chunks, chunk_boundaries

    def write_metadata_for_chunks(base_filename, doc_boundaries, chunk_boundaries):
        """Write metadata files for each chunk with document boundaries."""
        for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
            metadata_filename = f"{base_filename}_part_{chunk_idx:04d}.csv.gz"

            with gzip.open(metadata_filename, "wt") as f:
                for doc_start, doc_end in doc_boundaries:
                    if doc_end > chunk_start and doc_start < chunk_end:
                        adjusted_start = max(0, doc_start - chunk_start)
                        adjusted_end = min(chunk_end - chunk_start, doc_end - chunk_start)
                        if adjusted_end > adjusted_start:
                            f.write(f"{adjusted_start},{adjusted_end}\n")

            logger.info(f"Written metadata {metadata_filename}")

    logger.info(f"Writing converted data to {output_dir}")
    _, token_chunk_boundaries = write_memmap_chunked(f"{output_dir}/token_ids", all_token_ids, token_dtype)
    write_metadata_for_chunks(f"{output_dir}/token_ids", all_document_boundaries, token_chunk_boundaries)

    # Write labels_mask using the same chunk boundaries as token_ids
    for i, (start, end) in enumerate(token_chunk_boundaries):
        chunk_data = all_labels_mask[start:end]
        filename = f"{output_dir}/labels_mask_part_{i:04d}.npy"
        mmap = np.memmap(filename, mode="w+", dtype=np.bool_, shape=(len(chunk_data),))
        mmap[:] = chunk_data
        mmap.flush()
        logger.info(f"Written {filename} ({len(chunk_data) * np.dtype(np.bool_).itemsize / 1024**3:.2f} GB)")

    logger.info("Data conversion completed successfully!")

    # Remove checkpoint file on successful completion
    remove_checkpoint(output_dir)

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
    dataset_statistics: dict[str, Any],
    total_instances: int,
    total_tokens: int,
    total_trainable_tokens: int,
    num_samples_skipped: int,
    tokenizer_name: str,
    max_seq_length: int | None,
    chat_template_name: str | None,
    per_dataset_counts: dict[str, int],
    per_dataset_tokens: dict[str, int],
    per_dataset_trainable_tokens: dict[str, int],
    per_dataset_filtered: dict[str, int],
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
            "avg_tokens_per_instance": per_dataset_tokens[dataset_name] / per_dataset_counts[dataset_name]
            if per_dataset_counts[dataset_name] > 0
            else 0,
            "percentage_of_total_tokens": (per_dataset_tokens[dataset_name] / total_tokens * 100)
            if total_tokens > 0
            else 0,
            "percentage_of_total_instances": (per_dataset_counts[dataset_name] / total_instances * 100)
            if total_instances > 0
            else 0,
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
        },
    }

    # Write JSON file
    json_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    logger.info(f"Written dataset statistics to {json_path}")

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
            f.write(
                f"  - Instances filtered during transformation: {stat.get('instances_filtered_during_transformation', 'N/A')}\n"
            )

            if stat.get("frac_or_num_samples") is not None:
                if isinstance(stat["frac_or_num_samples"], float):
                    f.write(f"  - Sampling fraction: {stat['frac_or_num_samples']}\n")
                else:
                    f.write(f"  - Sample count: {stat['frac_or_num_samples']}\n")

            # Show upsampling information if applicable
            if stat.get("is_upsampled", False):
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

    logger.info(f"Written human-readable statistics to {text_path}")


if __name__ == "__main__":
    parser = ArgumentParserPlus((ConvertSFTDataArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
