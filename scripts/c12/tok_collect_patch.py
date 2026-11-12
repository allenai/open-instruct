"""Patch open_instruct/numpy_dataset_conversion.py (37a119ffd) in place so convert_hf_to_numpy_sft's collect loop
reads each batch with array operations over Arrow on worker threads, several batches ahead, instead of a Python loop
over rows in numpy format on one thread (250-440 rows/s, 2.5 h for the qwen-c12 mix). The main thread writes each
batch in order: the flattened token ids, the flattened label mask and the (start, end) pairs from the list offsets,
in that order, one tofile per file, then flushes and fsyncs as before. The three partial files come out
byte-identical to the original's, so resume, the statistics and the cache are unchanged
(scripts/c12/test_tok_collect_patch.py checks this). An attention mask that is not all ones fails the job after
writing the rows up to and including the bad one, as the original does. Applied after the sed fix and
scripts/c12/tok_stats_patch.py; it touches neither of their lines."""
import sys

p = sys.argv[1]
s = open(p).read()
old_format = '    train_dataset_iter = train_dataset_iter.with_format("numpy")\n'
new_format = (
    "    train_dataset_iter = train_dataset_iter.select_columns(\n"
    "        [\n"
    "            dataset_transformation.INPUT_IDS_KEY,\n"
    "            dataset_transformation.LABELS_KEY,\n"
    "            dataset_transformation.ATTENTION_MASK_KEY,\n"
    "        ]\n"
    '    ).with_format("arrow")\n'
)
old_loop = '''        for batch in train_dataset_iter.iter(batch_size=batch_size):
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
'''
new_loop = '''        batches = train_dataset_iter.iter(batch_size=_COLLECT_BATCH_ROWS)
        for prepared in _prefetch_in_order(batches, lambda b: _prepare_arrow_batch(b, token_dtype)):
            num_rows, current_position = _write_prepared_batch(
                prepared, current_position, tokens_fh, labels_fh, boundaries_fh
            )
            idx += num_rows
            progress.update(num_rows)
'''
func = '''
# Rows per collect batch, and the batches read ahead by worker threads. The shuffled rows are scattered over the
# memory-mapped Arrow cache, so reading them is random access; pyarrow copies the rows out (concat_arrays) with the
# GIL released, so the workers' page faults overlap. One 1,024-row batch is ~8M tokens at the qwen-c12 mean (~250 MB
# of temporaries); at most 67M tokens if every row were 65,536 long.
_COLLECT_BATCH_ROWS = 1024
_COLLECT_WORKERS = 16
_COLLECT_DEPTH = 32


def _prefetch_in_order(items, fn, workers=None, depth=None):
    \"\"\"Yield fn(item) for each item, in order, computing up to `depth` of them ahead on `workers` threads.\"\"\"
    import collections
    import concurrent.futures

    workers = workers or _COLLECT_WORKERS
    depth = depth or _COLLECT_DEPTH
    pool = concurrent.futures.ThreadPoolExecutor(workers)
    pending = collections.deque()
    try:
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= depth:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _arrow_list_parts(column, name: str) -> tuple[np.ndarray, np.ndarray]:
    \"\"\"The flattened values of an Arrow list column and its row offsets as int64, starting at 0.\"\"\"
    import pyarrow as pa
    import pyarrow.compute as pc

    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if arr.null_count:
        raise ValueError(f"{name} has null rows")
    values = pc.list_flatten(arr)
    if values.null_count:
        raise ValueError(f"{name} has null values")
    lengths = pc.list_value_length(arr).to_numpy(zero_copy_only=False).astype(np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return values.to_numpy(zero_copy_only=False), offsets


def _prepare_arrow_batch(batch, token_dtype) -> dict[str, Any]:
    \"\"\"Read one Arrow batch into the arrays the per-row loop would write. Runs on a worker thread.\"\"\"
    tokens, token_offsets = _arrow_list_parts(batch[dataset_transformation.INPUT_IDS_KEY], "input_ids")
    labels, label_offsets = _arrow_list_parts(batch[dataset_transformation.LABELS_KEY], "labels")
    attention, attention_offsets = _arrow_list_parts(batch[dataset_transformation.ATTENTION_MASK_KEY], "attention_mask")
    bad = np.flatnonzero(attention != 1)
    failed_row = int(np.searchsorted(attention_offsets, bad[0], side="right")) - 1 if bad.size else None
    return {
        "tokens": tokens.astype(token_dtype),
        "token_offsets": token_offsets,
        "labels_mask": (labels != -100).astype(np.uint8),
        "label_offsets": label_offsets,
        "failed_row": failed_row,
        "failed_mask": None
        if failed_row is None
        else attention[attention_offsets[failed_row] : attention_offsets[failed_row + 1]],
    }


def _write_prepared_batch(prepared, current_position: int, tokens_fh, labels_fh, boundaries_fh) -> tuple[int, int]:
    \"\"\"Append one batch with the bytes the per-row loop writes, in its file order; return (rows, position).\"\"\"
    token_offsets = prepared["token_offsets"]
    num_rows = len(token_offsets) - 1
    failed_row = prepared["failed_row"]
    if failed_row is not None:
        # The original writes each row before checking its mask, so the failing row is on disk when it raises.
        num_rows = failed_row + 1
    prepared["tokens"][: token_offsets[num_rows]].tofile(tokens_fh)
    prepared["labels_mask"][: prepared["label_offsets"][num_rows]].tofile(labels_fh)
    pairs = np.empty((num_rows, 2), dtype=_BOUNDARIES_DTYPE)
    pairs[:, 0] = token_offsets[:num_rows]
    pairs[:, 1] = token_offsets[1 : num_rows + 1]
    pairs += current_position
    pairs.tofile(boundaries_fh)
    current_position += int(token_offsets[num_rows])
    if failed_row is not None:
        raise AssertionError(f"Expected all attention mask values to be 1, but found: {prepared['failed_mask']}")
    return num_rows, current_position

'''
anchor = "\ndef convert_hf_to_numpy_sft("
assert s.count(old_format) == 1, "with_format line not found exactly once"
assert s.count(old_loop) == 1, "collect loop does not match 37a119ffd"
assert s.count(anchor) == 1, "convert_hf_to_numpy_sft not found exactly once"
s = s.replace(old_format, new_format).replace(old_loop, new_loop).replace(anchor, func + anchor, 1)
open(p, "w").write(s)
print("COLLECT-PATCHED", p)
