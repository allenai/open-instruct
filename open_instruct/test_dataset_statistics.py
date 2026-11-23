from unittest import mock

import pytest
from datasets import Dataset

from open_instruct import dataset_statistics


def test_counts_logical_rows_without_map_or_unrelated_columns():
    dataset = Dataset.from_dict(
        {"input_ids": [[1, 2], [], [3, 4, 5]], "labels": [[-100, 2], [], [3, -100, None]], "text": ["a", "b", "c"]}
    ).select([2, 0, 2, 1])
    expected = (
        sum(len(row["input_ids"]) for row in dataset),
        sum(label != -100 for row in dataset for label in row["labels"]),
    )
    with mock.patch.object(Dataset, "map", side_effect=AssertionError("must not create a derived dataset")):
        assert dataset_statistics.count_tokens(dataset, "input_ids", "labels", -100) == expected == (8, 5)
    assert dataset.column_names == ["input_ids", "labels", "text"]


def test_counts_multiple_batches_and_empty_dataset():
    dataset = Dataset.from_dict({"input_ids": [[1, 2]] * 2051, "labels": [[-100, 2]] * 2051})
    assert dataset_statistics.count_tokens(dataset, "input_ids", "labels", -100) == (4102, 2051)
    assert dataset_statistics.count_tokens(dataset.select([]), "input_ids", "labels", -100) == (0, 0)


def test_rejects_null_sequences():
    dataset = Dataset.from_dict({"input_ids": [[1], None], "labels": [[1], [1]]})
    with pytest.raises(ValueError, match="must not be null"):
        dataset_statistics.count_tokens(dataset, "input_ids", "labels", -100)
