"""Convert HuggingFace SFT datasets to OLMo-core numpy mmap format.

The output layout for each `output_dir` is:
    token_ids_part_XXXX.npy        # token ids, uint{8,16,32,64}
    labels_mask_part_XXXX.npy      # bool, 1 = trainable, 0 = masked
    token_ids_part_XXXX.csv.gz     # document boundaries (start,end) per chunk
    tokenizer/                     # HF tokenizer snapshot
    dataset_statistics.{json,txt}  # per-dataset stats
"""

import gzip
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from open_instruct import dataset_transformation, logger_utils

logger = logger_utils.setup_logger(__name__)


_CHECKPOINT_FILENAME = "_checkpoint.json"


def save_checkpoint(output_dir: str, checkpoint_data: dict[str, Any]) -> None:
    checkpoint_path = os.path.join(output_dir, _CHECKPOINT_FILENAME)
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoint_data, f)
    os.rename(tmp_path, checkpoint_path)


def load_checkpoint(output_dir: str) -> dict[str, Any] | None:
    checkpoint_path = os.path.join(output_dir, _CHECKPOINT_FILENAME)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return None


def remove_checkpoint(output_dir: str) -> None:
    checkpoint_path = os.path.join(output_dir, _CHECKPOINT_FILENAME)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info(f"Removed checkpoint file: {checkpoint_path}")


def _select_token_dtype(vocab_size: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"Vocab size {vocab_size} is too big for any numpy integer dtype!")


def _write_memmap_chunked(base_filename: str, data: list[int], dtype, max_size_gb: int = 1) -> list[tuple[int, int]]:
    item_size = np.dtype(dtype).itemsize
    max_size_bytes = max_size_gb * 1024**3
    chunk_size = max_size_bytes // item_size
    chunk_boundaries: list[tuple[int, int]] = []

    for i in range(0, len(data), chunk_size):
        chunk_data = data[i : i + chunk_size]
        filename = f"{base_filename}_part_{i // chunk_size:04d}.npy"
        mmap = np.memmap(filename, mode="w+", dtype=dtype, shape=(len(chunk_data),))
        mmap[:] = chunk_data
        mmap.flush()
        chunk_boundaries.append((i, i + len(chunk_data)))
        logger.info(f"Written {filename} ({len(chunk_data) * item_size / 1024**3:.2f} GB)")

    return chunk_boundaries


def _write_metadata_for_chunks(
    base_filename: str, document_boundaries: list[tuple[int, int]], chunk_boundaries: list[tuple[int, int]]
) -> None:
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        metadata_filename = f"{base_filename}_part_{chunk_idx:04d}.csv.gz"
        with gzip.open(metadata_filename, "wt") as f:
            for doc_start, doc_end in document_boundaries:
                if doc_end > chunk_start and doc_start < chunk_end:
                    adjusted_start = max(0, doc_start - chunk_start)
                    adjusted_end = min(chunk_end - chunk_start, doc_end - chunk_start)
                    if adjusted_end > adjusted_start:
                        f.write(f"{adjusted_start},{adjusted_end}\n")
        logger.info(f"Written metadata {metadata_filename}")


def _save_tokenizer(tc: dataset_transformation.TokenizerConfig, output_dir: str) -> None:
    tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    logger.info(f"Saving tokenizer to {tokenizer_output_dir}...")
    tc.tokenizer.save_pretrained(tokenizer_output_dir)

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


def convert_hf_to_numpy_sft(
    output_dir: str,
    dataset_mixer_list: list[str],
    dataset_mixer_list_splits: list[str],
    tc: dataset_transformation.TokenizerConfig,
    dataset_transform_fn: list[str],
    transform_fn_args: list[dict[str, Any]],
    dataset_target_columns: list[str],
    max_seq_length: int | None = None,
    dataset_cache_mode: Literal["hf", "local"] = "local",
    dataset_local_cache_dir: str = "local_dataset_cache",
    dataset_skip_cache: bool = False,
    dataset_config_hash: str | None = None,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    checkpoint_interval: int = 100_000,
    resume: bool = True,
    visualize: bool = False,
    tokenizer_config_only: bool = False,
    num_examples: int = 0,
) -> None:
    """Tokenize `dataset_mixer_list` and write OLMo-core numpy SFT files to `output_dir`.

    Safe to call again after interruption when `resume=True`: an on-disk checkpoint
    lets it pick up where it left off.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Verify these values match the tokenizer config used in Olmo-core:")
    logger.info(f"Tokenizer vocab_size: {tc.tokenizer.vocab_size}")
    logger.info(f"Tokenizer bos_token_id: {tc.tokenizer.bos_token_id}")
    logger.info(f"Tokenizer pad_token_id: {tc.tokenizer.pad_token_id}")
    logger.info(f"Tokenizer eos_token_id: {tc.tokenizer.eos_token_id}")

    _save_tokenizer(tc, output_dir)
    logger.info("Tokenizer saved successfully!")

    if tokenizer_config_only:
        return

    train_dataset, dataset_statistics = dataset_transformation.get_cached_dataset_tulu_with_statistics(
        dataset_mixer_list=dataset_mixer_list,
        dataset_mixer_list_splits=dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        target_columns=dataset_target_columns,
        dataset_cache_mode=dataset_cache_mode,
        dataset_config_hash=dataset_config_hash,
        dataset_local_cache_dir=dataset_local_cache_dir,
        dataset_skip_cache=dataset_skip_cache,
        drop_dataset_source=False,
    )

    if shuffle:
        train_dataset = train_dataset.shuffle(seed=shuffle_seed)

    if visualize:
        logger.info("Visualizing first example...")
        dataset_transformation.visualize_token(train_dataset[0][dataset_transformation.INPUT_IDS_KEY], tc.tokenizer)
        logger.info(f"Labels: {train_dataset[0][dataset_transformation.LABELS_KEY]}")
        logger.info(f"Attention mask: {train_dataset[0][dataset_transformation.ATTENTION_MASK_KEY]}")

    if num_examples > 0:
        logger.info(f"Selecting {num_examples} examples for debugging")
        train_dataset = train_dataset.select(range(num_examples))

    state: dict[str, Any] = {
        "samples_processed": 0,
        "token_ids": [],
        "labels_mask": [],
        "document_boundaries": [],
        "current_position": 0,
        "num_samples_skipped": 0,
        "per_dataset_counts": {},
        "per_dataset_tokens": {},
        "per_dataset_trainable_tokens": {},
        "per_dataset_filtered": {},
    }

    checkpoint = load_checkpoint(output_dir) if resume else None
    if checkpoint:
        state.update(checkpoint)
        state["document_boundaries"] = [tuple(b) for b in state["document_boundaries"]]
        logger.info("=== RESUMING from checkpoint ===")
        logger.info(f"  Samples already processed: {state['samples_processed']:,}")
        logger.info(f"  Tokens collected: {len(state['token_ids']):,}")
        logger.info(f"  Remaining samples: {len(train_dataset) - state['samples_processed']:,}")
    elif resume:
        logger.info("No checkpoint found, starting from beginning...")

    start_idx = state["samples_processed"]
    token_ids = state["token_ids"]
    labels_mask = state["labels_mask"]
    document_boundaries = state["document_boundaries"]
    current_position = state["current_position"]
    num_samples_skipped = state["num_samples_skipped"]
    per_dataset_counts = state["per_dataset_counts"]
    per_dataset_tokens = state["per_dataset_tokens"]
    per_dataset_trainable_tokens = state["per_dataset_trainable_tokens"]
    per_dataset_filtered = state["per_dataset_filtered"]

    logger.info("Collecting tokens from dataset...")
    sample: Mapping[str, Any]
    total_samples = len(train_dataset)

    train_dataset_iter = train_dataset.select(range(start_idx, total_samples)) if start_idx > 0 else train_dataset

    for idx, sample in enumerate(
        tqdm(
            train_dataset_iter,
            desc="Collecting tokens",
            file=sys.stdout,
            bar_format="{l_bar}{bar}{r_bar}\n",
            mininterval=10.0,
            initial=start_idx,
            total=total_samples,
        ),
        start=start_idx,
    ):
        sample_length = len(sample[dataset_transformation.INPUT_IDS_KEY])
        sample_tokens = sample[dataset_transformation.INPUT_IDS_KEY]
        sample_labels = sample[dataset_transformation.LABELS_KEY]
        dataset_source = sample.get(dataset_transformation.DATASET_ORIGIN_KEY, "unknown")

        if dataset_source not in per_dataset_counts:
            per_dataset_counts[dataset_source] = 0
            per_dataset_tokens[dataset_source] = 0
            per_dataset_trainable_tokens[dataset_source] = 0
            per_dataset_filtered[dataset_source] = 0

        per_dataset_counts[dataset_source] += 1
        per_dataset_tokens[dataset_source] += sample_length
        trainable_tokens_in_sample = sum(1 for label in sample_labels if label != -100)
        per_dataset_trainable_tokens[dataset_source] += trainable_tokens_in_sample

        token_ids.extend(sample_tokens)
        labels_mask.extend([1 if label != -100 else 0 for label in sample_labels])
        document_boundaries.append((current_position, current_position + sample_length))
        current_position += sample_length

        if all(label == -100 for label in sample_labels):
            num_samples_skipped += 1
            per_dataset_filtered[dataset_source] += 1

        assert all(mask == 1 for mask in sample[dataset_transformation.ATTENTION_MASK_KEY]), (
            f"Expected all attention mask values to be 1, but found: {sample[dataset_transformation.ATTENTION_MASK_KEY]}"
        )

        if (idx + 1) % checkpoint_interval == 0 and idx > start_idx:
            save_checkpoint(
                output_dir,
                {
                    "samples_processed": idx + 1,
                    "token_ids": token_ids,
                    "labels_mask": labels_mask,
                    "document_boundaries": document_boundaries,
                    "current_position": current_position,
                    "num_samples_skipped": num_samples_skipped,
                    "per_dataset_counts": per_dataset_counts,
                    "per_dataset_tokens": per_dataset_tokens,
                    "per_dataset_trainable_tokens": per_dataset_trainable_tokens,
                    "per_dataset_filtered": per_dataset_filtered,
                },
            )
            logger.info(f"Checkpoint saved at sample {idx + 1:,} ({len(token_ids):,} tokens)")

    train_dataset = dataset_transformation.remove_dataset_source_field(train_dataset)

    total_instances = len(train_dataset)
    total_tokens = len(token_ids)
    total_trainable_tokens = sum(labels_mask)

    logger.info(f"Total sequences: {total_instances}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Maximum token ID: {max(token_ids) if token_ids else 0}")
    logger.info(f"Labels mask sum (trainable tokens): {total_trainable_tokens}")
    logger.info("Writing data to numpy files...")
    logger.info(f"Number of samples that should be skipped: {num_samples_skipped}")

    vocab_size = tc.tokenizer.vocab_size
    token_dtype = _select_token_dtype(vocab_size)
    logger.info(f"Using dtype '{token_dtype}' for token_ids based on vocab size {vocab_size}")

    logger.info(f"Writing converted data to {output_dir}")
    token_chunk_boundaries = _write_memmap_chunked(f"{output_dir}/token_ids", token_ids, token_dtype)
    _write_metadata_for_chunks(f"{output_dir}/token_ids", document_boundaries, token_chunk_boundaries)

    for i, (start, end) in enumerate(token_chunk_boundaries):
        chunk_data = labels_mask[start:end]
        filename = f"{output_dir}/labels_mask_part_{i:04d}.npy"
        mmap = np.memmap(filename, mode="w+", dtype=np.bool_, shape=(len(chunk_data),))
        mmap[:] = chunk_data
        mmap.flush()
        logger.info(f"Written {filename} ({len(chunk_data) * np.dtype(np.bool_).itemsize / 1024**3:.2f} GB)")

    logger.info("Data conversion completed successfully!")
    remove_checkpoint(output_dir)

    write_dataset_statistics(
        output_dir=output_dir,
        dataset_statistics=dataset_statistics,
        total_instances=total_instances,
        total_tokens=total_tokens,
        total_trainable_tokens=total_trainable_tokens,
        num_samples_skipped=num_samples_skipped,
        tokenizer_name=tc.tokenizer_name_or_path,
        max_seq_length=max_seq_length,
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
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
            "original_dataset_size": pre_stat.get("original_dataset_size"),
            "is_upsampled": pre_stat.get("is_upsampled", False),
            "upsampling_factor": pre_stat.get("upsampling_factor", 1.0),
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

    json_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    logger.info(f"Written dataset statistics to {json_path}")

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

            if stat.get("is_upsampled", False):
                f.write(f"  - Original dataset size: {stat.get('original_dataset_size', 'N/A')}\n")
                f.write(f"  - Upsampling factor: {stat.get('upsampling_factor', 1.0):.2f}x\n")
                f.write(f"  - Upsampled to: {stat.get('instances_after_transformation', 'N/A')} instances\n")

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
