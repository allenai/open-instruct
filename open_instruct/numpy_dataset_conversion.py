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
import pathlib
import sys
import time
from datetime import datetime
from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from open_instruct import dataset_transformation, logger_utils, utils

logger = logger_utils.setup_logger(__name__)


_BOUNDARIES_DTYPE = np.int64


def _select_token_dtype(vocab_size: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"Vocab size {vocab_size} is too big for any numpy integer dtype!")


def _flush_partial_files(tokens_fh, labels_fh, boundaries_fh) -> None:
    for fh in (tokens_fh, labels_fh, boundaries_fh):
        fh.flush()
        os.fsync(fh.fileno())


def _write_memmap_chunked_from_file(
    base_filename: pathlib.Path, source_path: pathlib.Path, total_items: int, dtype, max_size_gb: int = 1
) -> list[tuple[int, int]]:
    item_size = np.dtype(dtype).itemsize
    chunk_size = int((max_size_gb * 1024**3) // item_size)
    chunk_boundaries: list[tuple[int, int]] = []

    if total_items == 0:
        return chunk_boundaries

    src = np.memmap(source_path, mode="r", dtype=dtype, shape=(total_items,))
    for chunk_idx, i in enumerate(range(0, total_items, chunk_size)):
        end = min(i + chunk_size, total_items)
        filename = base_filename.with_name(f"{base_filename.name}_part_{chunk_idx:04d}.npy")
        dst = np.memmap(filename, mode="w+", dtype=dtype, shape=(end - i,))
        dst[:] = src[i:end]
        dst.flush()
        chunk_boundaries.append((i, end))
        logger.info(f"Written {filename} ({(end - i) * item_size / 1024**3:.2f} GB)")

    return chunk_boundaries


def _write_metadata_for_chunks(
    base_filename: pathlib.Path, document_boundaries: np.ndarray, chunk_boundaries: list[tuple[int, int]]
) -> None:
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        metadata_filename = base_filename.with_name(f"{base_filename.name}_part_{chunk_idx:04d}.csv.gz")
        with gzip.open(metadata_filename, "wt", encoding="utf-8") as f:
            for doc_start, doc_end in document_boundaries:
                if doc_end > chunk_start and doc_start < chunk_end:
                    adjusted_start = max(0, int(doc_start) - chunk_start)
                    adjusted_end = min(chunk_end - chunk_start, int(doc_end) - chunk_start)
                    if adjusted_end > adjusted_start:
                        f.write(f"{adjusted_start},{adjusted_end}\n")
        logger.info(f"Written metadata {metadata_filename}")


def _save_tokenizer(tc: dataset_transformation.TokenizerConfig, output_dir: pathlib.Path) -> None:
    tokenizer_output_dir = output_dir / "tokenizer"
    tokenizer_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving tokenizer to {tokenizer_output_dir}...")
    tc.tokenizer.save_pretrained(str(tokenizer_output_dir))

    chat_template_path = tokenizer_output_dir / "chat_template.jinja"
    tokenizer_config_path = tokenizer_output_dir / "tokenizer_config.json"
    if chat_template_path.exists() and tokenizer_config_path.exists():
        chat_template_content = chat_template_path.read_text()
        with tokenizer_config_path.open() as f:
            tokenizer_config = json.load(f)
        if "chat_template" not in tokenizer_config:
            tokenizer_config["chat_template"] = chat_template_content
            with tokenizer_config_path.open("w") as f:
                json.dump(tokenizer_config, f, indent=2)
            logger.info(f"Added chat_template from {chat_template_path} to tokenizer_config.json")


def _recover_partial_state(
    tokens_path: pathlib.Path, labels_path: pathlib.Path, boundaries_path: pathlib.Path, token_item_size: int
) -> tuple[int, int]:
    # Boundaries file is the commit log: its last complete (start,end) pair determines how many
    # bytes of tokens/labels are valid; trailing bytes from an incomplete write get truncated.
    paths = (tokens_path, labels_path, boundaries_path)
    if not all(p.exists() for p in paths):
        for p in paths:
            p.unlink(missing_ok=True)
        return 0, 0

    pair_bytes = 2 * np.dtype(_BOUNDARIES_DTYPE).itemsize
    boundaries_size = boundaries_path.stat().st_size
    valid_bytes = (boundaries_size // pair_bytes) * pair_bytes
    if valid_bytes != boundaries_size:
        os.truncate(boundaries_path, valid_bytes)

    num_samples = valid_bytes // pair_bytes
    if num_samples == 0:
        os.truncate(tokens_path, 0)
        os.truncate(labels_path, 0)
        return 0, 0

    boundaries = np.memmap(boundaries_path, mode="r", dtype=_BOUNDARIES_DTYPE, shape=(num_samples, 2))
    current_position = int(boundaries[-1, 1])
    del boundaries

    os.truncate(tokens_path, current_position * token_item_size)
    os.truncate(labels_path, current_position)
    return num_samples, current_position


def _aggregate_stats(
    train_dataset,
    boundaries_path: pathlib.Path,
    labels_path: pathlib.Path,
    tokens_path: pathlib.Path,
    num_samples: int,
    total_tokens: int,
    token_dtype,
) -> dict[str, Any]:
    per_dataset_counts: dict[str, int] = {}
    per_dataset_tokens: dict[str, int] = {}
    per_dataset_trainable_tokens: dict[str, int] = {}
    per_dataset_filtered: dict[str, int] = {}

    if num_samples == 0:
        return {
            "per_dataset_counts": per_dataset_counts,
            "per_dataset_tokens": per_dataset_tokens,
            "per_dataset_trainable_tokens": per_dataset_trainable_tokens,
            "per_dataset_filtered": per_dataset_filtered,
            "total_trainable_tokens": 0,
            "num_samples_skipped": 0,
            "max_token_id": 0,
        }

    boundaries = np.memmap(boundaries_path, mode="r", dtype=_BOUNDARIES_DTYPE, shape=(num_samples, 2))
    sample_lengths = (boundaries[:, 1] - boundaries[:, 0]).astype(np.int64)

    labels_mm = np.memmap(labels_path, mode="r", dtype=np.uint8, shape=(total_tokens,))
    starts = boundaries[:, 0].astype(np.intp)
    trainable_counts = np.add.reduceat(labels_mm, starts, dtype=np.int64)
    zero_length_mask = sample_lengths == 0
    if zero_length_mask.any():
        trainable_counts = np.where(zero_length_mask, 0, trainable_counts)

    num_samples_skipped = int((trainable_counts == 0).sum())
    total_trainable_tokens = int(trainable_counts.sum())

    sources = train_dataset[dataset_transformation.DATASET_ORIGIN_KEY]
    if len(sources) != num_samples:
        raise ValueError(
            f"Dataset source column length {len(sources)} does not match number of processed "
            f"samples {num_samples}. Partial files may have been produced with a different dataset."
        )

    for source, length, trainable in zip(sources, sample_lengths.tolist(), trainable_counts.tolist()):
        if source not in per_dataset_counts:
            per_dataset_counts[source] = 0
            per_dataset_tokens[source] = 0
            per_dataset_trainable_tokens[source] = 0
            per_dataset_filtered[source] = 0
        per_dataset_counts[source] += 1
        per_dataset_tokens[source] += int(length)
        per_dataset_trainable_tokens[source] += int(trainable)
        if trainable == 0:
            per_dataset_filtered[source] += 1

    tokens_mm = np.memmap(tokens_path, mode="r", dtype=token_dtype, shape=(total_tokens,))
    max_token_id = int(tokens_mm.max()) if total_tokens > 0 else 0

    return {
        "per_dataset_counts": per_dataset_counts,
        "per_dataset_tokens": per_dataset_tokens,
        "per_dataset_trainable_tokens": per_dataset_trainable_tokens,
        "per_dataset_filtered": per_dataset_filtered,
        "total_trainable_tokens": total_trainable_tokens,
        "num_samples_skipped": num_samples_skipped,
        "max_token_id": max_token_id,
    }


def convert_hf_to_numpy_sft(
    output_dir: pathlib.Path,
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
    shuffle_seed: int = 42,
    resume: bool = False,
    visualize: bool = False,
    tokenizer_config_only: bool = False,
    num_examples: int = 0,
    batch_size: int = 1000,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_dataset = train_dataset.shuffle(seed=shuffle_seed)

    if visualize:
        logger.info("Visualizing first example...")
        dataset_transformation.visualize_token(train_dataset[0][dataset_transformation.INPUT_IDS_KEY], tc.tokenizer)
        logger.info(f"Labels: {train_dataset[0][dataset_transformation.LABELS_KEY]}")
        logger.info(f"Attention mask: {train_dataset[0][dataset_transformation.ATTENTION_MASK_KEY]}")

    if num_examples > 0:
        logger.info(f"Selecting {num_examples} examples for debugging")
        train_dataset = train_dataset.select(range(num_examples))

    vocab_size = tc.tokenizer.vocab_size
    token_dtype = _select_token_dtype(vocab_size)
    token_item_size = np.dtype(token_dtype).itemsize
    logger.info(f"Using dtype '{np.dtype(token_dtype).name}' for token_ids based on vocab size {vocab_size}")

    tokens_path = output_dir / "_tokens.partial.bin"
    labels_path = output_dir / "_labels.partial.bin"
    boundaries_path = output_dir / "_boundaries.partial.bin"

    if resume:
        start_idx, current_position = _recover_partial_state(
            tokens_path, labels_path, boundaries_path, token_item_size
        )
        if start_idx > 0:
            logger.info(f"=== RESUMING from partial files: {start_idx:,} samples / {current_position:,} tokens ===")
        else:
            logger.info("No recoverable partial files found, starting from beginning...")
    else:
        for path in (tokens_path, labels_path, boundaries_path):
            path.unlink(missing_ok=True)
        start_idx = 0
        current_position = 0

    total_samples = len(train_dataset)
    logger.info("Collecting tokens from dataset...")

    if start_idx > 0:
        train_dataset_iter = train_dataset.select(range(min(start_idx, total_samples), total_samples))
    else:
        train_dataset_iter = train_dataset

    train_dataset_iter = train_dataset_iter.with_format("numpy")

    input_ids_key = dataset_transformation.INPUT_IDS_KEY
    labels_key = dataset_transformation.LABELS_KEY
    attention_mask_key = dataset_transformation.ATTENTION_MASK_KEY

    progress = tqdm(
        desc="Collecting tokens",
        file=sys.stdout,
        bar_format="{l_bar}{bar}{r_bar}\n",
        mininterval=10.0,
        initial=start_idx,
        total=total_samples,
    )

    collect_start = time.perf_counter()
    utils.maybe_update_beaker_description(current_step=start_idx, total_steps=total_samples, start_time=collect_start)
    last_description_update = collect_start
    idx = start_idx - 1
    with (
        tokens_path.open("ab") as tokens_fh,
        labels_path.open("ab") as labels_fh,
        boundaries_path.open("ab") as boundaries_fh,
    ):
        for batch in train_dataset_iter.iter(batch_size=batch_size):
            batch_input_ids = batch[input_ids_key]
            batch_labels = batch[labels_key]
            batch_attention = batch[attention_mask_key]

            for sample_tokens, sample_labels, sample_attention in zip(batch_input_ids, batch_labels, batch_attention):
                idx += 1
                sample_length = len(sample_tokens)

                tokens_arr = np.asarray(sample_tokens, dtype=token_dtype)
                labels_arr = np.asarray(sample_labels)
                labels_mask = (labels_arr != -100).astype(np.uint8)

                tokens_arr.tofile(tokens_fh)
                labels_mask.tofile(labels_fh)
                np.array([current_position, current_position + sample_length], dtype=_BOUNDARIES_DTYPE).tofile(
                    boundaries_fh
                )
                current_position += sample_length

                assert (sample_attention == 1).all(), (
                    f"Expected all attention mask values to be 1, but found: {sample_attention}"
                )

            progress.update(len(batch_input_ids))
            _flush_partial_files(tokens_fh, labels_fh, boundaries_fh)

            now = time.perf_counter()
            if now - last_description_update >= 30.0:
                utils.maybe_update_beaker_description(
                    current_step=idx + 1, total_steps=total_samples, start_time=collect_start
                )
                last_description_update = now

    progress.close()
    utils.maybe_update_beaker_description(
        current_step=total_samples, total_steps=total_samples, start_time=collect_start
    )
    collect_elapsed = time.perf_counter() - collect_start
    samples_processed = max(0, total_samples - start_idx)
    rate = samples_processed / collect_elapsed if collect_elapsed > 0 else 0.0
    logger.info(f"Collect loop: {samples_processed:,} samples in {collect_elapsed:.2f}s ({rate:.1f} samples/s)")

    total_tokens = current_position
    logger.info("Aggregating per-dataset statistics from disk...")
    stats_start = time.perf_counter()
    stats = _aggregate_stats(
        train_dataset=train_dataset,
        boundaries_path=boundaries_path,
        labels_path=labels_path,
        tokens_path=tokens_path,
        num_samples=total_samples,
        total_tokens=total_tokens,
        token_dtype=token_dtype,
    )
    logger.info(f"Stats aggregation: {time.perf_counter() - stats_start:.2f}s")

    logger.info(f"Total sequences: {total_samples}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Maximum token ID: {stats['max_token_id']}")
    logger.info(f"Labels mask sum (trainable tokens): {stats['total_trainable_tokens']}")
    logger.info(f"Number of samples that should be skipped: {stats['num_samples_skipped']}")

    logger.info(f"Writing converted data to {output_dir}")
    token_ids_base = output_dir / "token_ids"
    token_chunk_boundaries = _write_memmap_chunked_from_file(token_ids_base, tokens_path, total_tokens, token_dtype)

    document_boundaries = (
        np.memmap(boundaries_path, mode="r", dtype=_BOUNDARIES_DTYPE, shape=(total_samples, 2))
        if total_samples > 0
        else np.zeros((0, 2), dtype=_BOUNDARIES_DTYPE)
    )
    _write_metadata_for_chunks(token_ids_base, document_boundaries, token_chunk_boundaries)
    del document_boundaries

    labels_src = np.memmap(labels_path, mode="r", dtype=np.uint8, shape=(total_tokens,)) if total_tokens else None
    for i, (start, end) in enumerate(token_chunk_boundaries):
        filename = output_dir / f"labels_mask_part_{i:04d}.npy"
        mmap = np.memmap(filename, mode="w+", dtype=np.bool_, shape=(end - start,))
        mmap[:] = labels_src[start:end].astype(np.bool_)
        mmap.flush()
        logger.info(f"Written {filename} ({(end - start) * np.dtype(np.bool_).itemsize / 1024**3:.2f} GB)")
    del labels_src

    for path in (tokens_path, labels_path, boundaries_path):
        path.unlink(missing_ok=True)

    logger.info("Data conversion completed successfully!")

    write_dataset_statistics(
        output_dir=output_dir,
        dataset_statistics=dataset_statistics,
        total_instances=total_samples,
        total_tokens=total_tokens,
        total_trainable_tokens=stats["total_trainable_tokens"],
        num_samples_skipped=stats["num_samples_skipped"],
        tokenizer_name=tc.tokenizer_name_or_path,
        max_seq_length=max_seq_length,
        chat_template_name=tc.chat_template_name,
        per_dataset_counts=stats["per_dataset_counts"],
        per_dataset_tokens=stats["per_dataset_tokens"],
        per_dataset_trainable_tokens=stats["per_dataset_trainable_tokens"],
        per_dataset_filtered=stats["per_dataset_filtered"],
    )


def write_dataset_statistics(
    output_dir: pathlib.Path,
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
        "output_directory": str(output_dir),
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

    json_path = output_dir / "dataset_statistics.json"
    with json_path.open("w") as f:
        json.dump(stats_data, f, indent=2)
    logger.info(f"Written dataset statistics to {json_path}")

    text_path = output_dir / "dataset_statistics.txt"
    with text_path.open("w") as f:
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
