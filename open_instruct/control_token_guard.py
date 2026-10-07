"""Find SFT rows whose raw text spells out one of the tokenizer's control tokens.

A fast tokenizer splits added special tokens out of the text before BPE runs, so a literal
`<|im_end|>` or `<|endoftext|>` inside a message tokenizes to the real control id. In a training
row that plants a spurious turn boundary or EOS the chat template never put there. The scan reads
the raw fields *before* the chat template renders them, so the control tokens the template itself
inserts are never seen.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pyarrow
from datasets import Dataset
from pyarrow import compute
from transformers import PreTrainedTokenizerBase

_SCAN_BATCH_SIZE = 10_000


def control_tokens(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """The tokenizer's special tokens: the named ones plus every added token marked special.

    Added tokens that are not special are left out on purpose. They include the Olmo PII
    placeholders (`|||EMAIL_ADDRESS|||`), which real text contains, and the tags promoted into
    reserved slots by `--reserved_slot_tokens`, which `promote_tokens_into_reserved_slots` keeps
    non-special so that a literal `<think>` in assistant content maps to the promoted id.
    """
    tokens = set(tokenizer.all_special_tokens)
    tokens.update(token.content for token in tokenizer.added_tokens_decoder.values() if token.special)
    return sorted(token for token in tokens if token)


_RE2_METACHARACTERS = frozenset("\\.+*?()|[]{}^$")


def _re2_literal(text: str) -> str:
    return "".join("\\" + char if char in _RE2_METACHARACTERS else char for char in text)


def _rows_with_match(array: pyarrow.Array | pyarrow.ChunkedArray, pattern: str) -> np.ndarray:
    """For each element of `array`, whether any string nested anywhere inside it matches `pattern`."""
    if isinstance(array, pyarrow.ChunkedArray):
        if array.num_chunks == 0:
            return np.zeros(0, dtype=bool)
        return np.concatenate([_rows_with_match(chunk, pattern) for chunk in array.chunks])
    hits = np.zeros(len(array), dtype=bool)
    array_type = array.type
    if pyarrow.types.is_dictionary(array_type):
        return _rows_with_match(array.dictionary_decode(), pattern)
    if pyarrow.types.is_string(array_type) or pyarrow.types.is_large_string(array_type):
        matched = compute.call_function("match_substring_regex", [array], compute.MatchSubstringOptions(pattern))
        return compute.call_function("coalesce", [matched, False]).to_numpy(zero_copy_only=False)
    if pyarrow.types.is_struct(array_type):
        # flatten() applies the struct's offset and null mask to every child.
        for child in array.flatten():
            hits |= _rows_with_match(child, pattern)
        return hits
    if pyarrow.types.is_fixed_size_list(array_type):
        return _rows_with_match(array.cast(pyarrow.list_(array_type.value_field)), pattern)
    if (
        pyarrow.types.is_list(array_type)
        or pyarrow.types.is_large_list(array_type)
        or pyarrow.types.is_map(array_type)
    ):
        # `offsets` already accounts for a slice; `values` is the unsliced child. A null list
        # may still own values, so its row is masked out afterwards.
        offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        child_hits = _rows_with_match(array.values.slice(offsets[0], offsets[-1] - offsets[0]), pattern)
        parents = np.repeat(np.arange(len(array)), np.diff(offsets))
        hits[parents[child_hits]] = True
        if array.null_count:
            hits &= array.is_valid().to_numpy(zero_copy_only=False)
        return hits
    # Numbers, booleans and nulls cannot spell a token.
    return hits


def control_token_row_mask(
    dataset: Dataset, columns: Sequence[str], tokens: Sequence[str], stop_at_first_hit: bool = False
) -> np.ndarray:
    """Whether each row has a control-token literal in any string inside `columns`.

    Every string leaf counts, however deeply nested: message content in any role,
    `reasoning_content`, tool-call names and arguments, and tool schemas, whether stored as
    structs or JSON strings. Columns missing from `dataset` are skipped. With `stop_at_first_hit`
    the scan stops after the first batch with a hit and the remaining rows read as False.
    """
    mask = np.zeros(len(dataset), dtype=bool)
    present = [column for column in columns if column in dataset.column_names]
    if not present or not tokens or len(dataset) == 0:
        return mask
    pattern = "|".join(_re2_literal(token) for token in tokens)
    start = 0
    for batch in dataset.select_columns(present).with_format("arrow").iter(batch_size=_SCAN_BATCH_SIZE):
        for column in batch.columns:
            mask[start : start + batch.num_rows] |= _rows_with_match(column, pattern)
        start += batch.num_rows
        if stop_at_first_hit and mask.any():
            break
    return mask


def control_token_locations(value: Any, tokens: Sequence[str], path: str = "") -> Iterator[tuple[str, str]]:
    """Yield `(path, token)` for every control-token literal inside a decoded row value."""
    if isinstance(value, str):
        for token in tokens:
            if token in value:
                yield path, token
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from control_token_locations(item, tokens, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from control_token_locations(item, tokens, f"{path}[{index}]")
