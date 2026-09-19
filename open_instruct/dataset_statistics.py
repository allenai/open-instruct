"""Token statistics over logical dataset rows without writing a derived cache."""

from datasets import Dataset
from pyarrow import compute


def count_tokens(dataset: Dataset, input_ids_key: str, labels_key: str, masked_token_value: int) -> tuple[int, int]:
    total_tokens = 0
    trainable_tokens = 0
    columns = dataset.select_columns([input_ids_key, labels_key]).with_format("arrow")
    for batch in columns.iter(batch_size=1024):
        inputs, labels = batch[input_ids_key], batch[labels_key]
        if inputs.null_count or labels.null_count:
            raise ValueError("Token and label sequences must not be null")
        total_tokens += (
            compute.call_function("sum", [compute.call_function("list_value_length", [inputs])]).as_py() or 0
        )
        # Python's original label != mask count includes null scalar labels.
        trainable = compute.fill_null(
            compute.call_function("not_equal", [compute.call_function("list_flatten", [labels]), masked_token_value]),
            True,
        )
        trainable_tokens += compute.call_function("sum", [trainable]).as_py() or 0
    return total_tokens, trainable_tokens
