"""Convert HuggingFace SFT datasets to OLMo-core numpy mmap format.

The output layout for each `output_dir` is:
    token_ids_part_XXXX.npy        # token ids, uint{8,16,32,64}
    labels_mask_part_XXXX.npy      # bool, 1 = trainable, 0 = masked
    token_ids_part_XXXX.csv.gz     # document boundaries (start,end) per chunk
    tokenizer/                     # HF tokenizer snapshot
    dataset_statistics.{json,txt}  # per-dataset stats
"""

import collections
import contextlib
import functools
import gzip
import io
import json
import os
import pathlib
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent import futures
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pyarrow as pa
from pyarrow import compute
from tqdm import tqdm

from open_instruct import dataset_transformation, logger_utils, utils

logger = logger_utils.setup_logger(__name__)


_BOUNDARIES_DTYPE = np.int64

TOKEN_IDS_NPY_GLOB = "token_ids_part_*.npy"
LABELS_MASK_NPY_GLOB = "labels_mask_part_*.npy"
TOKEN_IDS_METADATA_GLOB = "token_ids_part_*.csv.gz"
# Set in dataset_statistics.json's configuration when the parts were cut only between rows.
ROW_ALIGNED_PARTS_KEY = "row_aligned_parts"

# The collect loop reads batches on worker threads, up to _COLLECT_PREFETCH_BATCHES ahead of the writer. The shuffled
# rows are scattered over the memory-mapped Arrow cache, so reading them is random access, and pyarrow copies them out
# with the GIL released, so the threads' reads overlap. Memory: each prepared batch waiting to be written holds about
# 5 bytes per token (the token ids at the token dtype and a uint8 label mask), and each worker briefly holds about
# 50 bytes per token of its batch (the three combined int64 columns and the temporaries). With batches of B tokens,
# the peak is about B * (5 * _COLLECT_PREFETCH_BATCHES + 50 * _COLLECT_WORKERS) bytes: about 6.5 GB at the qwen-c12
# mean (1,000 rows of ~6,800 tokens), and about 63 GB if every row of a 1,000-row batch had 65,536 tokens. Lower
# `batch_size` for mixes of very long rows on small machines.
_COLLECT_WORKERS = 16
_COLLECT_PREFETCH_BATCHES = 32
# Elements per chunk when summing the label mask per row: a 1 GiB int64 temporary.
_SEGMENT_SUM_CHUNK = 1 << 27


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


def _segment_sums(values: np.ndarray, starts: np.ndarray, chunk_size: int = _SEGMENT_SUM_CHUNK) -> np.ndarray:
    """int64 sums of `values[starts[i]:starts[i + 1]]`, the last segment running to the end of `values`.

    `starts` must be non-decreasing, so a segment whose start equals the next one's sums to 0. `values` is read
    `chunk_size` elements at a time, so a memory-mapped array is never cast to int64 whole, which is what
    `np.add.reduceat(values, starts, dtype=np.int64)` does.
    """
    n = len(values)
    bounds = np.append(np.asarray(starts, dtype=np.int64), n)
    if bounds[0] < 0 or np.any(np.diff(bounds) < 0):
        raise ValueError("Segment starts must be non-negative, non-decreasing and at most len(values).")
    # prefix[i] = sum of values[: bounds[i]].
    prefix = np.empty(len(bounds), dtype=np.int64)
    running = 0
    done = 0
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        cumsum = np.cumsum(values[chunk_start:chunk_end], dtype=np.int64)
        # The bounds in [chunk_start, chunk_end); every earlier bound is filled in already.
        upto = int(np.searchsorted(bounds, chunk_end, side="left"))
        offsets = bounds[done:upto] - chunk_start
        prefix[done:upto] = running + np.where(offsets > 0, cumsum[np.maximum(offsets - 1, 0)], 0)
        running += int(cumsum[-1])
        done = upto
    prefix[done:] = running
    return np.diff(prefix)


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
    # The rows are contiguous, so each row's segment runs from its start to the next row's, and an empty row sums to 0.
    trainable_counts = _segment_sums(labels_mm, boundaries[:, 0])

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


def _prefetch_in_order(
    items: Iterable[Any],
    fn: Callable[[Any], Any],
    workers: int = _COLLECT_WORKERS,
    depth: int = _COLLECT_PREFETCH_BATCHES,
) -> Iterator[Any]:
    """Yield `fn(item)` for each item in order, computing up to `depth` results ahead on `workers` threads."""
    pending: collections.deque[futures.Future] = collections.deque()
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for item in items:
                pending.append(pool.submit(fn, item))
                if len(pending) >= depth:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


def _list_column_parts(column: pa.ChunkedArray | pa.Array, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The flattened values of an Arrow list column, and its row offsets as int64 starting at 0."""
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if array.null_count:
        raise ValueError(f"{name} has null rows")
    values = compute.list_flatten(array)
    if values.null_count:
        raise ValueError(f"{name} has null values")
    offsets = np.zeros(len(array) + 1, dtype=np.int64)
    np.cumsum(compute.list_value_length(array).to_numpy(zero_copy_only=False), out=offsets[1:])
    return values.to_numpy(zero_copy_only=False), offsets


def _prepare_collect_batch(batch: pa.Table, token_dtype) -> dict[str, Any]:
    """Turn one Arrow batch into the arrays the collect loop writes. Runs on a worker thread."""
    tokens, token_offsets = _list_column_parts(batch[dataset_transformation.INPUT_IDS_KEY], "input_ids")
    labels, label_offsets = _list_column_parts(batch[dataset_transformation.LABELS_KEY], "labels")
    attention, attention_offsets = _list_column_parts(
        batch[dataset_transformation.ATTENTION_MASK_KEY], "attention_mask"
    )
    empty_rows = np.flatnonzero(token_offsets[1:] == token_offsets[:-1])
    bad_attention = np.flatnonzero(attention != 1)
    bad_attention_row = (
        int(np.searchsorted(attention_offsets, bad_attention[0], side="right")) - 1 if bad_attention.size else None
    )
    return {
        "tokens": tokens.astype(token_dtype),
        "token_offsets": token_offsets,
        "labels_mask": (labels != -100).astype(np.uint8),
        "label_offsets": label_offsets,
        "first_empty_row": int(empty_rows[0]) if empty_rows.size else None,
        "bad_attention_row": bad_attention_row,
        "bad_attention": None
        if bad_attention_row is None
        else attention[attention_offsets[bad_attention_row] : attention_offsets[bad_attention_row + 1]],
    }


def _write_collect_batch(
    prepared: dict[str, Any],
    current_position: int,
    tokens_fh,
    labels_fh,
    boundaries_fh,
    first_row: int,
    reject_empty_rows: bool,
) -> tuple[int, int]:
    """Append one prepared batch to the partial files and return (rows written, new position).

    The files get the bytes a row-by-row loop would write, in the same order. That loop checks each row for emptiness
    before writing it and checks its attention mask after, so on either failure the rows before the failing one, and
    for a bad attention mask the failing row too, are written before the error is raised.
    """
    token_offsets = prepared["token_offsets"]
    num_rows = len(token_offsets) - 1
    empty_row = prepared["first_empty_row"] if reject_empty_rows else None
    bad_row = prepared["bad_attention_row"]
    error: Exception | None = None
    if empty_row is not None and (bad_row is None or empty_row <= bad_row):
        num_rows = empty_row
        error = ValueError(
            f"--row_aligned_parts cannot write empty row {first_row + empty_row}: "
            "each row must contain at least one token to have a metadata boundary."
        )
    elif bad_row is not None:
        num_rows = bad_row + 1
        error = AssertionError(f"Expected all attention mask values to be 1, but found: {prepared['bad_attention']}")

    prepared["tokens"][: token_offsets[num_rows]].tofile(tokens_fh)
    prepared["labels_mask"][: prepared["label_offsets"][num_rows]].tofile(labels_fh)
    pairs = np.empty((num_rows, 2), dtype=_BOUNDARIES_DTYPE)
    pairs[:, 0] = token_offsets[:num_rows]
    pairs[:, 1] = token_offsets[1 : num_rows + 1]
    pairs += current_position
    pairs.tofile(boundaries_fh)
    current_position += int(token_offsets[num_rows])
    if error is not None:
        raise error
    return num_rows, current_position


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

    if start_idx > 0:
        # datasets rejects select(range(N, N)), so a resume that finds every row collected selects [] instead.
        train_dataset_iter = train_dataset.select(range(start_idx, total_samples) if start_idx < total_samples else [])
    else:
        train_dataset_iter = train_dataset

    train_dataset_iter = train_dataset_iter.select_columns(
        [
            dataset_transformation.INPUT_IDS_KEY,
            dataset_transformation.LABELS_KEY,
            dataset_transformation.ATTENTION_MASK_KEY,
        ]
    ).with_format("arrow")

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
        contextlib.closing(
            _prefetch_in_order(
                train_dataset_iter.iter(batch_size=batch_size),
                functools.partial(_prepare_collect_batch, token_dtype=token_dtype),
            )
        ) as prepared_batches,
    ):
        for prepared in prepared_batches:
            num_rows, current_position = _write_collect_batch(
                prepared,
                current_position,
                tokens_fh,
                labels_fh,
                boundaries_fh,
                first_row=idx + 1,
                reject_empty_rows=row_aligned_parts,
            )
            idx += num_rows
            progress.update(num_rows)
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
