"""Find SFT rows whose raw text spells out one of the tokenizer's control tokens.

A fast tokenizer splits added special tokens out of the text before BPE runs, so a literal
`<|im_end|>` or `<|endoftext|>` inside a message tokenizes to the real control id. In a training
row that plants a spurious turn boundary or EOS the chat template never put there. The scan reads
the raw fields *before* the chat template renders them, so the control tokens the template itself
inserts are never seen.
"""

from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

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
_JSON_SHORT_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "\b": "b", "\f": "f", "\n": "n", "\r": "r", "\t": "t"}


def _re2_literal(text: str) -> str:
    return "".join("\\" + char if char in _RE2_METACHARACTERS else char for char in text)


def _re2_json_unicode_escape(code_unit: int) -> str:
    # JSON's \uXXXX accepts either case for the hex digits.
    digits = "".join(f"[{d}{d.upper()}]" if d.isalpha() else d for d in f"{code_unit:04x}")
    return "\\\\u" + digits


def _re2_json_char(char: str) -> str:
    """Every way JSON text can spell `char` inside a string."""
    options = [_re2_literal(char)]
    code = ord(char)
    if code <= 0xFFFF:
        options.append(_re2_json_unicode_escape(code))
    else:
        high, low = 0xD800 + ((code - 0x10000) >> 10), 0xDC00 + ((code - 0x10000) & 0x3FF)
        options.append(_re2_json_unicode_escape(high) + _re2_json_unicode_escape(low))
    if char in _JSON_SHORT_ESCAPES:
        options.append("\\\\" + _re2_literal(_JSON_SHORT_ESCAPES[char]))
    return "(?:" + "|".join(options) + ")"


class _Patterns(NamedTuple):
    literal: str
    """Matches a token spelled out in plain text."""
    json: str
    """Matches a token spelled out in JSON text, where any character may be escaped."""


def _patterns(tokens: Sequence[str]) -> _Patterns:
    return _Patterns(
        literal="|".join(_re2_literal(token) for token in tokens),
        json="|".join("".join(_re2_json_char(char) for char in token) for token in tokens),
    )


def _matches(strings: pyarrow.Array, pattern: str) -> np.ndarray:
    matched = compute.call_function("match_substring_regex", [strings], compute.MatchSubstringOptions(pattern))
    return compute.call_function("coalesce", [matched, False]).to_numpy(zero_copy_only=False)


def _joined_text_parts(array: pyarrow.Array) -> pyarrow.Array | None:
    """For a list of `{"type": "text", "text": ...}` parts, each list's texts joined with no separator.

    Chat templates concatenate content parts directly, so a token split across two parts is
    still one token once rendered.
    """
    value_type = array.type.value_type
    if not pyarrow.types.is_struct(value_type) or value_type.get_field_index("text") < 0:
        return None
    text_type = value_type.field("text").type
    if not (pyarrow.types.is_string(text_type) or pyarrow.types.is_large_string(text_type)):
        return None
    texts = compute.call_function("struct_field", [array.values], compute.StructFieldOptions("text"))
    texts = compute.call_function("coalesce", [texts, ""])
    rebuilt = type(array).from_arrays(array.offsets, texts)
    return compute.call_function("binary_join", [rebuilt, ""])


def _rows_with_match(array: pyarrow.Array | pyarrow.ChunkedArray, patterns: _Patterns, json_text: bool) -> np.ndarray:
    """For each element of `array`, whether any string nested anywhere inside it spells a token.

    Strings that are JSON text (`json_text`, or the storage of an Arrow extension type such as
    datasets' `Json` feature) are matched with escapes allowed, since they are decoded before
    rendering. Struct field names count too, for every row where the struct itself is set:
    chat templates render the keys of tool-call arguments and tool schemas, and a decoded row
    carries every field of the struct type, unset ones as None.
    """
    if isinstance(array, pyarrow.ChunkedArray):
        if array.num_chunks == 0:
            return np.zeros(0, dtype=bool)
        return np.concatenate([_rows_with_match(chunk, patterns, json_text) for chunk in array.chunks])
    hits = np.zeros(len(array), dtype=bool)
    array_type = array.type
    if isinstance(array_type, pyarrow.BaseExtensionType):
        return _rows_with_match(array.storage, patterns, json_text=True)
    if pyarrow.types.is_dictionary(array_type):
        return _rows_with_match(array.dictionary_decode(), patterns, json_text)
    if pyarrow.types.is_string(array_type) or pyarrow.types.is_large_string(array_type):
        return _matches(array, patterns.json if json_text else patterns.literal)
    if pyarrow.types.is_struct(array_type):
        names = [array_type.field(i).name for i in range(array_type.num_fields)]
        if names and _matches(pyarrow.array(names, type=pyarrow.string()), patterns.literal).any():
            hits |= array.is_valid().to_numpy(zero_copy_only=False)
        # flatten() applies the struct's offset and null mask to every child.
        for child in array.flatten():
            hits |= _rows_with_match(child, patterns, json_text)
        return hits
    if pyarrow.types.is_fixed_size_list(array_type):
        return _rows_with_match(array.cast(pyarrow.list_(array_type.value_field)), patterns, json_text)
    if (
        pyarrow.types.is_list(array_type)
        or pyarrow.types.is_large_list(array_type)
        or pyarrow.types.is_map(array_type)
    ):
        # `offsets` already accounts for a slice; `values` is the unsliced child. A null list
        # may still own values, so its row is masked out afterwards.
        offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        child_hits = _rows_with_match(array.values.slice(offsets[0], offsets[-1] - offsets[0]), patterns, json_text)
        parents = np.repeat(np.arange(len(array)), np.diff(offsets))
        hits[parents[child_hits]] = True
        joined = None if pyarrow.types.is_map(array_type) else _joined_text_parts(array)
        if joined is not None:
            hits |= _matches(joined, patterns.json if json_text else patterns.literal)
        if array.null_count:
            hits &= array.is_valid().to_numpy(zero_copy_only=False)
        return hits
    # Numbers, booleans and nulls cannot spell a token.
    return hits


def control_token_row_mask(
    dataset: Dataset,
    columns: Sequence[str],
    tokens: Sequence[str],
    json_columns: Sequence[str] = (),
    stop_at_first_hit: bool = False,
) -> np.ndarray:
    """Whether each row spells out a control token in any string inside `columns`.

    Every string leaf counts, however deeply nested: message content in any role (text parts
    joined as the template joins them), `reasoning_content`, tool-call names and arguments (keys
    and values), and tool schemas, whether stored as structs, JSON strings or datasets' `Json`
    feature. Strings in `json_columns` are JSON text the pipeline decodes before rendering, so
    escaped spellings count there. This is deliberately a superset of what any one template
    renders: a token in a field the template ignores (a tool-call `id`, say) still drops the row.
    Columns missing from `dataset` are skipped. With `stop_at_first_hit` the scan stops after
    the first batch with a hit and the remaining rows read as False.
    """
    mask = np.zeros(len(dataset), dtype=bool)
    present = [column for column in columns if column in dataset.column_names]
    if not present or not tokens or len(dataset) == 0:
        return mask
    patterns = _patterns(tokens)
    start = 0
    for batch in dataset.select_columns(present).with_format("arrow").iter(batch_size=_SCAN_BATCH_SIZE):
        for name, column in zip(batch.column_names, batch.columns):
            mask[start : start + batch.num_rows] |= _rows_with_match(column, patterns, name in json_columns)
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
            key_path = f"{path}.{key}" if path else str(key)
            for token in tokens:
                if token in str(key):
                    yield f"{key_path} (key)", token
            yield from control_token_locations(item, tokens, key_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from control_token_locations(item, tokens, f"{path}[{index}]")
