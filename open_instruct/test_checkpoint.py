#!/usr/bin/env python
"""Tests for checkpoint functions in open_instruct.numpy_dataset_conversion.

Run from project root:
    pytest open_instruct/test_checkpoint.py -v
"""

import json
import tempfile

import pytest

from open_instruct import numpy_dataset_conversion


class TestCheckpointFunctions:
    def test_save_checkpoint_creates_file(self, tmp_path):
        checkpoint_data = {"samples_processed": 100, "token_ids": [1, 2, 3]}
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), checkpoint_data)

        checkpoint_path = tmp_path / "_checkpoint.json"
        assert checkpoint_path.exists()

    def test_save_checkpoint_content(self, tmp_path):
        checkpoint_data = {
            "samples_processed": 1000,
            "token_ids": [1, 2, 3, 4, 5],
            "labels_mask": [1, 0, 1, 0, 1],
            "current_position": 500,
        }
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), checkpoint_data)

        checkpoint_path = tmp_path / "_checkpoint.json"
        with open(checkpoint_path) as f:
            loaded = json.load(f)

        assert loaded == checkpoint_data

    def test_save_checkpoint_atomic(self, tmp_path):
        checkpoint_data = {"samples_processed": 100}
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), checkpoint_data)

        tmp_file = tmp_path / "_checkpoint.json.tmp"
        assert not tmp_file.exists()

        checkpoint_path = tmp_path / "_checkpoint.json"
        assert checkpoint_path.exists()

    def test_save_checkpoint_overwrites(self, tmp_path):
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), {"samples_processed": 100})
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), {"samples_processed": 200})

        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))
        assert loaded["samples_processed"] == 200

    def test_load_checkpoint_returns_data(self, tmp_path):
        checkpoint_data = {
            "samples_processed": 5000,
            "token_ids": list(range(100)),
            "document_boundaries": [(0, 10), (10, 25), (25, 50)],
        }
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), checkpoint_data)

        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))
        assert loaded["samples_processed"] == 5000
        assert loaded["token_ids"] == list(range(100))
        assert loaded["document_boundaries"] == [[0, 10], [10, 25], [25, 50]]

    def test_load_checkpoint_no_file(self, tmp_path):
        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))
        assert loaded is None

    def test_load_checkpoint_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded = numpy_dataset_conversion.load_checkpoint(tmp_dir)
            assert loaded is None

    def test_remove_checkpoint_deletes_file(self, tmp_path):
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), {"samples_processed": 100})
        checkpoint_path = tmp_path / "_checkpoint.json"
        assert checkpoint_path.exists()

        numpy_dataset_conversion.remove_checkpoint(str(tmp_path))
        assert not checkpoint_path.exists()

    def test_remove_checkpoint_no_file(self, tmp_path):
        numpy_dataset_conversion.remove_checkpoint(str(tmp_path))

    def test_roundtrip_full_checkpoint(self, tmp_path):
        checkpoint_data = {
            "samples_processed": 150000,
            "token_ids": list(range(1000)),
            "labels_mask": [i % 2 for i in range(1000)],
            "document_boundaries": [(i * 10, (i + 1) * 10) for i in range(100)],
            "current_position": 1000,
            "num_samples_skipped": 5,
            "per_dataset_counts": {"dataset_a": 100, "dataset_b": 50},
            "per_dataset_tokens": {"dataset_a": 5000, "dataset_b": 2500},
            "per_dataset_trainable_tokens": {"dataset_a": 4000, "dataset_b": 2000},
            "per_dataset_filtered": {"dataset_a": 2, "dataset_b": 3},
        }

        numpy_dataset_conversion.save_checkpoint(str(tmp_path), checkpoint_data)
        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))

        assert loaded["samples_processed"] == 150000
        assert loaded["token_ids"] == list(range(1000))
        assert loaded["labels_mask"] == [i % 2 for i in range(1000)]
        boundaries = [tuple(b) for b in loaded["document_boundaries"]]
        assert boundaries == [(i * 10, (i + 1) * 10) for i in range(100)]
        assert loaded["current_position"] == 1000
        assert loaded["num_samples_skipped"] == 5
        assert loaded["per_dataset_counts"] == {"dataset_a": 100, "dataset_b": 50}
        assert loaded["per_dataset_tokens"] == {"dataset_a": 5000, "dataset_b": 2500}
        assert loaded["per_dataset_trainable_tokens"] == {"dataset_a": 4000, "dataset_b": 2000}
        assert loaded["per_dataset_filtered"] == {"dataset_a": 2, "dataset_b": 3}

    def test_document_boundaries_tuple_conversion(self, tmp_path):
        checkpoint_data = {"samples_processed": 100, "document_boundaries": [(0, 10), (10, 20), (20, 30)]}
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), checkpoint_data)
        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))

        boundaries = [tuple(b) for b in loaded["document_boundaries"]]
        assert boundaries == [(0, 10), (10, 20), (20, 30)]

    def test_large_checkpoint(self, tmp_path):
        num_tokens = 100_000
        checkpoint_data = {
            "samples_processed": 50000,
            "token_ids": list(range(num_tokens)),
            "labels_mask": [1] * num_tokens,
            "document_boundaries": [(i * 100, (i + 1) * 100) for i in range(1000)],
            "current_position": num_tokens,
            "num_samples_skipped": 10,
            "per_dataset_counts": {"test_dataset": 50000},
            "per_dataset_tokens": {"test_dataset": num_tokens},
            "per_dataset_trainable_tokens": {"test_dataset": num_tokens},
            "per_dataset_filtered": {"test_dataset": 10},
        }

        numpy_dataset_conversion.save_checkpoint(str(tmp_path), checkpoint_data)
        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))

        assert len(loaded["token_ids"]) == num_tokens
        assert loaded["samples_processed"] == 50000


class TestCheckpointIntegration:
    def test_simulate_interrupt_and_resume(self, tmp_path):
        state = {
            "samples_processed": 100,
            "token_ids": [i for i in range(500)],
            "labels_mask": [1, 0, 1, 0, 1] * 100,
            "document_boundaries": [(i * 5, (i + 1) * 5) for i in range(100)],
            "current_position": 500,
            "num_samples_skipped": 2,
            "per_dataset_counts": {"ds": 100},
            "per_dataset_tokens": {"ds": 500},
            "per_dataset_trainable_tokens": {"ds": 300},
            "per_dataset_filtered": {"ds": 2},
        }
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), state)

        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))
        assert loaded is not None

        start_idx = loaded["samples_processed"]
        assert start_idx == 100

        token_ids = loaded["token_ids"]
        for i in range(start_idx, 200):
            token_ids.extend([i * 5 + j for j in range(5)])

        assert len(token_ids) == 1000

    def test_no_checkpoint_fresh_start(self, tmp_path):
        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))

        start_idx = 0 if loaded is None else loaded["samples_processed"]
        assert start_idx == 0

    def test_cleanup_after_success(self, tmp_path):
        numpy_dataset_conversion.save_checkpoint(str(tmp_path), {"samples_processed": 1000})

        checkpoint_path = tmp_path / "_checkpoint.json"
        assert checkpoint_path.exists()

        numpy_dataset_conversion.remove_checkpoint(str(tmp_path))

        assert not checkpoint_path.exists()

        loaded = numpy_dataset_conversion.load_checkpoint(str(tmp_path))
        assert loaded is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
