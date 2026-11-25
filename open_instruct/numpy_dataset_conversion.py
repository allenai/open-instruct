"""Convert HuggingFace SFT datasets to OLMo-core numpy mmap format.

The output layout for each `output_dir` is:
    token_ids_part_XXXX.npy        # token ids, uint{8,16,32,64}
    labels_mask_part_XXXX.npy      # bool, 1 = trainable, 0 = masked
    token_ids_part_XXXX.csv.gz     # document boundaries (start,end) per chunk
    tokenizer/                     # HF tokenizer snapshot
    dataset_statistics.{json,txt}  # per-dataset stats
"""

import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Literal

import numpy as np
from tqdm import tqdm

from open_instruct import dataset_transformation, logger_utils, utils

logger = logger_utils.setup_logger(__name__)


_BOUNDARIES_DTYPE = np.int64

TOKEN_IDS_NPY_GLOB = "token_ids_part_*.npy"
LABELS_MASK_NPY_GLOB = "labels_mask_part_*.npy"
TOKEN_IDS_METADATA_GLOB = "token_ids_part_*.csv.gz"
# Set in dataset_statistics.json's configuration when the parts were cut only between rows.
ROW_ALIGNED_PARTS_KEY = "row_aligned_parts"


def _select_token_dtype(vocab_size: int):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"Vocab size {vocab_size} is too big for any numpy integer dtype!")


def _flush_partial_files(tokens_fh, labels_fh, boundaries_fh) -> None:
    for fh in (tokens_fh, labels_fh, boundaries_fh):
        fh.flush()
        os.fsync(fh.fileno())


def _row_aligned_chunk_ends(document_ends: np.ndarray, total_items: int, chunk_size: int) -> list[int]:
    """End offsets of chunks of at most `chunk_size` items that only end where a row ends.

    A row longer than `chunk_size` gets a chunk of its own, over the size limit, rather than being cut.
    """
    ends: list[int] = []
    start = 0
    while start < total_items:
        # The last row end within the size limit, or failing that the first row end after it.
        idx = int(np.searchsorted(document_ends, start + chunk_size, side="right")) - 1
        if idx < 0 or document_ends[idx] <= start:
            idx = int(np.searchsorted(document_ends, start, side="right"))
        ends.append(int(document_ends[idx]))
        start = ends[-1]
    return ends


def _write_memmap_chunked_from_file(
    base_filename: pathlib.Path,
    source_path: pathlib.Path,
    total_items: int,
    dtype,
    max_size_gb: int = 1,
    document_ends: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Copy `source_path` into `_part_XXXX.npy` files of at most `max_size_gb` each.

    Without `document_ends`, files are cut every `max_size_gb`, mid-row. With the sorted row end
    offsets, files are cut only where a row ends, so no row spans two files.
    """
    item_size = np.dtype(dtype).itemsize
    chunk_size = int((max_size_gb * 1024**3) // item_size)
    chunk_boundaries: list[tuple[int, int]] = []

    if total_items == 0:
        return chunk_boundaries

    if document_ends is None:
        chunk_ends = [min(i + chunk_size, total_items) for i in range(0, total_items, chunk_size)]
    else:
        if len(document_ends) == 0 or int(document_ends[-1]) != total_items:
            raise ValueError(f"Row ends must finish at the last token ({total_items:,}).")
        chunk_ends = _row_aligned_chunk_ends(document_ends, total_items, chunk_size)

    src = np.memmap(source_path, mode="r", dtype=dtype, shape=(total_items,))
    i = 0
    for chunk_idx, end in enumerate(chunk_ends):
        filename = base_filename.with_name(f"{base_filename.name}_part_{chunk_idx:04d}.npy")
        dst = np.memmap(filename, mode="w+", dtype=dtype, shape=(end - i,))
        dst[:] = src[i:end]
        dst.flush()
        chunk_boundaries.append((i, end))
        logger.info(f"Written {filename} ({(end - i) * item_size / 1024**3:.2f} GB)")
        i = end

    return chunk_boundaries


def _write_metadata_for_chunks(
    base_filename: pathlib.Path, document_boundaries: np.ndarray, chunk_boundaries: list[tuple[int, int]]
) -> None:
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        metadata_filename = base_filename.with_name(f"{base_filename.name}_part_{chunk_idx:04d}.csv.gz")
        # mtime=0 keeps the file byte-identical across rewrites of the same rows. OLMo-core
        # fingerprints metadata-backed datasets by these files.
        with gzip.GzipFile(metadata_filename, mode="wb", mtime=0) as gz, io.TextIOWrapper(gz, encoding="utf-8") as f:
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
    tokens_path: pathlib.Path,
    labels_path: pathlib.Path,
    boundaries_path: pathlib.Path,
    token_item_size: int,
    reject_empty_rows: bool = False,
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
    if reject_empty_rows and np.any(boundaries[:, 0] == boundaries[:, 1]):
        raise ValueError(
            "Cannot resume --row_aligned_parts: the partial cache contains an empty row. "
            "Remove empty rows from the input and restart without --resume."
        )
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
    row_aligned_parts: bool = False,
) -> None:
    """Tokenize the mix and write it to `output_dir` in OLMo-core's numpy layout.

    With `row_aligned_parts`, the token and label files are cut only at row boundaries, so every
    row is one `(start,end)` line of one metadata file. Otherwise they are cut every 1 GiB, mid-row,
    and the row at each cut appears as two lines, one in each file.
    """
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
            tokens_path, labels_path, boundaries_path, token_item_size, reject_empty_rows=row_aligned_parts
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

    if start_idx > total_samples:
        raise ValueError(
            f"Partial files in {output_dir} hold {start_idx:,} rows but the dataset has {total_samples:,}; "
            "they belong to another dataset. Delete them or rerun without --resume."
        )
    if start_idx > 0:
        # A run preempted after collecting every row resumes with nothing left to collect; `select` rejects
        # the empty range that starts at the dataset's length, so take an empty slice instead.
        remaining = range(start_idx, total_samples) if start_idx < total_samples else []
        train_dataset_iter = train_dataset.select(remaining)
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
                if row_aligned_parts and sample_length == 0:
                    raise ValueError(
                        f"--row_aligned_parts cannot write empty row {idx}: "
                        "each row must contain at least one token to have a metadata boundary."
                    )

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
    document_boundaries = (
        np.memmap(boundaries_path, mode="r", dtype=_BOUNDARIES_DTYPE, shape=(total_samples, 2))
        if total_samples > 0
        else np.zeros((0, 2), dtype=_BOUNDARIES_DTYPE)
    )
    token_chunk_boundaries = _write_memmap_chunked_from_file(
        token_ids_base,
        tokens_path,
        total_tokens,
        token_dtype,
        document_ends=document_boundaries[:, 1] if row_aligned_parts else None,
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
        chat_template_name=dataset_statistics.get("chat_template_name"),
        chat_template_source=dataset_statistics.get("chat_template_source"),
        chat_template_hash=dataset_statistics.get("chat_template_hash"),
        per_dataset_counts=stats["per_dataset_counts"],
        per_dataset_tokens=stats["per_dataset_tokens"],
        per_dataset_trainable_tokens=stats["per_dataset_trainable_tokens"],
        per_dataset_filtered=stats["per_dataset_filtered"],
        row_aligned_parts=row_aligned_parts,
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
    chat_template_source: str | None = None,
    chat_template_hash: str | None = None,
    row_aligned_parts: bool = False,
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
            "chat_template_source": chat_template_source,
            "chat_template_hash": chat_template_hash,
            # Only when set, so default caches keep their existing statistics layout.
            **({ROW_ALIGNED_PARTS_KEY: True} if row_aligned_parts else {}),
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
        f.write(f"- Chat Template: {chat_template_name}\n")
        f.write(f"- Chat Template Source: {chat_template_source}\n")
        f.write(f"- Chat Template Hash: {chat_template_hash}\n\n")

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


# allenai/dolma2-tokenizer-olmo35: `<|endoftext|>` (EOS, which also ends every assistant turn) and `<|im_start|>`.
OLMO35_EOS_TOKEN_ID = 100257
OLMO35_IM_START_TOKEN_ID = 100264
# The replay build's record (the ledger's experiments/c11/c11_replay.py build): one JSON line per document with its
# `part`, `start`, `end` and `length_with_eos`.
REPLAY_MANIFEST = "manifest.jsonl"
# sha256 of H182's replay manifest (/weka/olmo-3p5-post-checkpoints/abhishekr/h182/replay-b1c12r), computed
# 2026-11-24; its ids, rows and tokens match the ledger's experiments/h182/receipts/r2_b1c12r.json.
H182_REPLAY_MANIFEST_SHA256 = "2683f40911e79e95bae4d2202ada1b7f677b2b8daacf649b25ee929cd8dcc412"
_PART_RE = {
    kind: re.compile(re.escape(glob).replace(r"\*", r"(\d+)") + "$")
    for kind, glob in (
        ("tokens", TOKEN_IDS_NPY_GLOB),
        ("mask", LABELS_MASK_NPY_GLOB),
        ("metadata", TOKEN_IDS_METADATA_GLOB),
    )
}


def replay_final_eos_positions(
    token_ids: np.ndarray, document_bounds: np.ndarray, eos_token_id: int, im_start_token_id: int
) -> np.ndarray:
    """Label positions to clear: the final token of each replay document, when that token is EOS.

    A replay document (long-context, pretraining-style text) is one whose first token is not
    `<|im_start|>`; every chat row starts with it. A document `[start, end)` of `document_bounds` is
    selected iff it has at least two tokens, does not start with `im_start_token_id` and ends with
    `eos_token_id`; the result holds its `end - 1`. A label mask is indexed by the target token, so
    clearing it there drops exactly the target that predicts that EOS from the token before it, in
    the same document: the rule of ledger H270's TPU patch (`replay_eos_keep`). Chat rows, documents
    that end in another token, interior EOS tokens and one-token documents (whose only target is
    predicted from the previous document) are never selected. `document_bounds` must be whole
    documents: an EOS-scan or a part cut would make an interior EOS look final.
    """
    starts, ends = document_bounds[:, 0], document_bounds[:, 1]
    long_enough = ends - starts >= 2
    starts, ends = starts[long_enough], ends[long_enough]
    selected = (token_ids[starts] != im_start_token_id) & (token_ids[ends - 1] == eos_token_id)
    return ends[selected] - 1


def _numpy_dir_parts(numpy_dir: str) -> dict[str, dict[str, str]]:
    """{kind: {part index string: path}} for the token, mask and metadata parts of a numpy dir."""
    parts: dict[str, dict[str, str]] = {kind: {} for kind in _PART_RE}
    for name in os.listdir(numpy_dir):
        for kind, pattern in _PART_RE.items():
            match = pattern.match(name)
            if match is not None:
                parts[kind][match.group(1)] = os.path.join(numpy_dir, name)
    return parts


def _read_document_bounds(metadata_path: str) -> np.ndarray:
    """Every line of a `.csv.gz` as (start, end). A blank or malformed line is refused: OLMo-core cannot parse
    it, and main's startup check counts it as a row."""
    rows = []
    with gzip.open(metadata_path, "rt") as f:
        for number, line in enumerate(f, 1):
            fields = line.rstrip("\n").split(",")
            if len(fields) < 2 or not (fields[0].isdigit() and fields[1].isdigit()):
                raise ValueError(f"{metadata_path}: line {number} is not a start,end row: {line!r}.")
            rows.append((int(fields[0]), int(fields[1])))
    return np.array(rows, dtype=np.int64).reshape(-1, 2)


def _replay_manifest_bounds(src_dir: str) -> dict[str, np.ndarray]:
    """{part index string: (start, end) rows in manifest order} from the replay build's manifest."""
    path = os.path.join(src_dir, REPLAY_MANIFEST)
    if not os.path.exists(path):
        raise ValueError(
            f"{src_dir} has no {REPLAY_MANIFEST}, the replay build's per-document record; without it the "
            "document boundaries cannot be tied to the build."
        )
    rows: dict[str, list[tuple[int, int]]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            start, end = int(record["start"]), int(record["end"])
            if "length_with_eos" in record and end - start != int(record["length_with_eos"]):
                raise ValueError(
                    f"{path}: a row spans {end - start:,} tokens but records {record['length_with_eos']:,}."
                )
            rows.setdefault(f"{int(record['part']):04d}", []).append((start, end))
    return {index: np.array(r, dtype=np.int64).reshape(-1, 2) for index, r in rows.items()}


def write_replay_eos_masked_dir(
    src_dir: str,
    out_dir: str,
    expected_manifest_sha256: str,
    eos_token_id: int = OLMO35_EOS_TOKEN_ID,
    im_start_token_id: int = OLMO35_IM_START_TOKEN_ID,
) -> dict[str, Any]:
    """Write `out_dir`: the replay dir `src_dir` with the final-EOS label of each replay document cleared.

    `src_dir`'s manifest must have sha256 `expected_manifest_sha256` (H182's: `H182_REPLAY_MANIFEST_SHA256`),
    checked before anything is written, so the boundaries are the build's and not a rewritten record.

    Ledger H270's fix as data, for a long-context replay dir passed to olmo_core_finetune.py's
    `--extra_numpy_dirs` and written by the ledger's replay build, whose `manifest.jsonl` records every
    document. The documents are the manifest's rows. In each part they must tile the part and equal its
    `.csv.gz` rows, the boundaries training reads, so no document is cut between parts and an interior
    EOS is never taken for a final one; any other dir is refused.

    `out_dir` gets new label masks, False at `replay_final_eos_positions` and equal to the source's
    elsewhere, and symlinks to the source's token parts, `.csv.gz` parts and manifest, so tokens,
    boundaries, OLMo-core's packing and the data order are unchanged. Its `dataset_statistics.json` is
    new: loss-token counts from the new masks, the row-alignment marker (true: each document is one
    row inside one part, checked above) and the change made; the source's statistics are not carried
    over. `src_dir` is not modified. OLMo-core packs whole documents and truncates one longer than the
    sequence length, so a truncated document loses its final EOS with the cleared label, and nothing is
    dropped for it. Returns the statistics written.
    """
    src_dir = os.path.realpath(src_dir)
    manifest_path = os.path.join(src_dir, REPLAY_MANIFEST)
    if not os.path.exists(manifest_path):
        raise ValueError(f"{src_dir} has no {REPLAY_MANIFEST}, the replay build's per-document record.")
    digest = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    if digest.hexdigest() != expected_manifest_sha256:
        raise ValueError(
            f"{manifest_path} has sha256 {digest.hexdigest()}, not the pinned {expected_manifest_sha256}; "
            "its boundaries cannot be tied to the replay build."
        )
    parts = _numpy_dir_parts(src_dir)
    manifest = _replay_manifest_bounds(src_dir)
    if not parts["tokens"] or not (
        set(parts["tokens"]) == set(parts["mask"]) == set(parts["metadata"]) == set(manifest)
    ):
        raise ValueError(
            f"{src_dir}: needs token, label-mask, .csv.gz and manifest parts with one shared index set; found "
            f"{ {kind: sorted(found) for kind, found in [*parts.items(), ('manifest', manifest)]} }."
        )
    # Positions first, so nothing is written for a dir that is refused.
    plan, counts = [], {"documents": 0, "tokens": 0, "starting_im_start": 0, "not_ending_eos": 0}
    for index in sorted(parts["tokens"]):
        num_tokens = os.path.getsize(parts["mask"][index])
        itemsize, rest = divmod(os.path.getsize(parts["tokens"][index]), max(num_tokens, 1))
        if num_tokens == 0 or rest or itemsize not in (1, 2, 4, 8):
            raise ValueError(f"{parts['tokens'][index]}: not one uint8/16/32/64 token per one-byte label.")
        bounds = manifest[index]
        expected_starts = np.concatenate([[0], bounds[:-1, 1]])
        if (
            (bounds[:, 0] != expected_starts).any()
            or (bounds[:, 1] <= bounds[:, 0]).any()
            or bounds[-1, 1] != num_tokens
        ):
            raise ValueError(f"{src_dir}: the manifest's rows for part {index} do not tile its {num_tokens:,} tokens.")
        if not np.array_equal(_read_document_bounds(parts["metadata"][index]), bounds):
            raise ValueError(f"{parts['metadata'][index]}: rows differ from the manifest's for part {index}.")
        tokens = np.memmap(parts["tokens"][index], mode="r", dtype=np.dtype(f"uint{8 * itemsize}"))
        counts["documents"] += len(bounds)
        counts["tokens"] += num_tokens
        counts["starting_im_start"] += int((tokens[bounds[:, 0]] == im_start_token_id).sum())
        counts["not_ending_eos"] += int((tokens[bounds[:, 1] - 1] != eos_token_id).sum())
        plan.append((index, replay_final_eos_positions(tokens, bounds, eos_token_id, im_start_token_id)))

    os.makedirs(out_dir)
    receipt_parts, dropped, already_untrained, trainable = [], 0, 0, 0
    for index, positions in plan:
        out_mask = os.path.join(out_dir, os.path.basename(parts["mask"][index]))
        shutil.copyfile(parts["mask"][index], out_mask)
        mask = np.memmap(out_mask, mode="r+", dtype=np.uint8)
        was_trained = mask[positions] != 0
        mask[positions] = 0
        mask.flush()
        trainable += int(np.count_nonzero(mask))
        del mask
        receipt_parts.append({"part": index, "selected": len(positions), "dropped": int(was_trained.sum())})
        dropped += int(was_trained.sum())
        already_untrained += int((~was_trained).sum())
        for kind in ("tokens", "metadata"):
            os.symlink(parts[kind][index], os.path.join(out_dir, os.path.basename(parts[kind][index])))
    os.symlink(os.path.join(src_dir, REPLAY_MANIFEST), os.path.join(out_dir, REPLAY_MANIFEST))
    stats = {
        "output_directory": out_dir,
        "configuration": {
            ROW_ALIGNED_PARTS_KEY: True,
            "replay_eos_mask": {
                "source": src_dir,
                "rule": "label cleared on the final token of each manifest document of >= 2 tokens that does not "
                "start with im_start_token_id and ends with eos_token_id (ledger H270)",
                "eos_token_id": eos_token_id,
                "im_start_token_id": im_start_token_id,
                **{k: v for k, v in counts.items() if k != "tokens"},
                "dropped": dropped,
                "already_untrained": already_untrained,
                "parts": receipt_parts,
            },
        },
        "overall_statistics": {
            "total_instances": counts["documents"],
            "total_tokens": counts["tokens"],
            "trainable_tokens": trainable,
            "trainable_percentage": trainable / counts["tokens"] * 100,
        },
    }
    with open(os.path.join(out_dir, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=1)
    logger.info(
        f"Wrote {out_dir}: {dropped:,} replay final-EOS labels cleared over {counts['documents']:,} documents; "
        f"{already_untrained:,} were already untrained; {trainable:,} of {counts['tokens']:,} tokens trained."
    )
    return stats
