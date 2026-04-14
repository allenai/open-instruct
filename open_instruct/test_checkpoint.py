#!/usr/bin/env python
"""Tests for checkpoint functions in convert_sft_data_for_olmocore.py

Run from project root:
    pytest open_instruct/test_checkpoint.py -v
"""

import json
import os
import tempfile

# Checkpoint functions copied here to avoid importing from convert_sft_data_for_olmocore.py
# which has heavy dependencies (datasets, transformers, etc.)
from typing import Any

import pytest


def save_checkpoint(output_dir: str, checkpoint_data: dict[str, Any]) -> None:
    """Save checkpoint to disk atomically."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(checkpoint_data, f)
    os.rename(tmp_path, checkpoint_path)  # Atomic on POSIX


def load_checkpoint(output_dir: str) -> dict[str, Any] | None:
    """Load checkpoint from disk if it exists."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            return json.load(f)
    return None


def remove_checkpoint(output_dir: str) -> None:
    """Remove checkpoint file after successful completion."""
    checkpoint_path = os.path.join(output_dir, "_checkpoint.json")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Removed checkpoint file: {checkpoint_path}")


class TestCheckpointFunctions:
    """Test suite for checkpoint save/load/remove functions."""

    def test_save_checkpoint_creates_file(self, tmp_path):
        """Test that save_checkpoint creates a checkpoint file."""
        checkpoint_data = {"samples_processed": 100, "token_ids": [1, 2, 3]}
        save_checkpoint(str(tmp_path), checkpoint_data)

        checkpoint_path = tmp_path / "_checkpoint.json"
        assert checkpoint_path.exists()

    def test_save_checkpoint_content(self, tmp_path):
        """Test that save_checkpoint writes correct content."""
        checkpoint_data = {
            "samples_processed": 1000,
            "token_ids": [1, 2, 3, 4, 5],
            "labels_mask": [1, 0, 1, 0, 1],
            "current_position": 500,
        }
        save_checkpoint(str(tmp_path), checkpoint_data)

        checkpoint_path = tmp_path / "_checkpoint.json"
        with open(checkpoint_path) as f:
            loaded = json.load(f)

        assert loaded == checkpoint_data

    def test_save_checkpoint_atomic(self, tmp_path):
        """Test that save_checkpoint doesn't leave tmp files on success."""
        checkpoint_data = {"samples_processed": 100}
        save_checkpoint(str(tmp_path), checkpoint_data)

        # Should not have .tmp file
        tmp_file = tmp_path / "_checkpoint.json.tmp"
        assert not tmp_file.exists()

        # Should have final file
        checkpoint_path = tmp_path / "_checkpoint.json"
        assert checkpoint_path.exists()

    def test_save_checkpoint_overwrites(self, tmp_path):
        """Test that save_checkpoint overwrites existing checkpoint."""
        # Save first checkpoint
        save_checkpoint(str(tmp_path), {"samples_processed": 100})

        # Save second checkpoint
        save_checkpoint(str(tmp_path), {"samples_processed": 200})

        # Load and verify it's the second one
        loaded = load_checkpoint(str(tmp_path))
        assert loaded["samples_processed"] == 200

    def test_load_checkpoint_returns_data(self, tmp_path):
        """Test that load_checkpoint returns saved data."""
        checkpoint_data = {
            "samples_processed": 5000,
            "token_ids": list(range(100)),
            "document_boundaries": [(0, 10), (10, 25), (25, 50)],
        }
        save_checkpoint(str(tmp_path), checkpoint_data)

        loaded = load_checkpoint(str(tmp_path))
        assert loaded["samples_processed"] == 5000
        assert loaded["token_ids"] == list(range(100))
        # JSON converts tuples to lists
        assert loaded["document_boundaries"] == [[0, 10], [10, 25], [25, 50]]

    def test_load_checkpoint_no_file(self, tmp_path):
        """Test that load_checkpoint returns None when no checkpoint exists."""
        loaded = load_checkpoint(str(tmp_path))
        assert loaded is None

    def test_load_checkpoint_empty_dir(self):
        """Test load_checkpoint with a fresh temp directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded = load_checkpoint(tmp_dir)
            assert loaded is None

    def test_remove_checkpoint_deletes_file(self, tmp_path):
        """Test that remove_checkpoint deletes the checkpoint file."""
        # Create checkpoint
        save_checkpoint(str(tmp_path), {"samples_processed": 100})
        checkpoint_path = tmp_path / "_checkpoint.json"
        assert checkpoint_path.exists()

        # Remove it
        remove_checkpoint(str(tmp_path))
        assert not checkpoint_path.exists()

    def test_remove_checkpoint_no_file(self, tmp_path):
        """Test that remove_checkpoint handles missing file gracefully."""
        # Should not raise an error
        remove_checkpoint(str(tmp_path))

    def test_roundtrip_full_checkpoint(self, tmp_path):
        """Test complete roundtrip with all checkpoint fields."""
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

        # Save
        save_checkpoint(str(tmp_path), checkpoint_data)

        # Load
        loaded = load_checkpoint(str(tmp_path))

        # Verify all fields
        assert loaded["samples_processed"] == 150000
        assert loaded["token_ids"] == list(range(1000))
        assert loaded["labels_mask"] == [i % 2 for i in range(1000)]
        # JSON converts tuples to lists - convert back for comparison
        boundaries = [tuple(b) for b in loaded["document_boundaries"]]
        assert boundaries == [(i * 10, (i + 1) * 10) for i in range(100)]
        assert loaded["current_position"] == 1000
        assert loaded["num_samples_skipped"] == 5
        assert loaded["per_dataset_counts"] == {"dataset_a": 100, "dataset_b": 50}
        assert loaded["per_dataset_tokens"] == {"dataset_a": 5000, "dataset_b": 2500}
        assert loaded["per_dataset_trainable_tokens"] == {"dataset_a": 4000, "dataset_b": 2000}
        assert loaded["per_dataset_filtered"] == {"dataset_a": 2, "dataset_b": 3}

    def test_document_boundaries_tuple_conversion(self, tmp_path):
        """Test that document boundaries are properly serialized (tuples become lists in JSON)."""
        checkpoint_data = {"samples_processed": 100, "document_boundaries": [(0, 10), (10, 20), (20, 30)]}
        save_checkpoint(str(tmp_path), checkpoint_data)
        loaded = load_checkpoint(str(tmp_path))

        # JSON converts tuples to lists, so we need to convert back
        boundaries = [tuple(b) for b in loaded["document_boundaries"]]
        assert boundaries == [(0, 10), (10, 20), (20, 30)]

    def test_large_checkpoint(self, tmp_path):
        """Test checkpoint with larger data sizes."""
        # Simulate a realistic checkpoint size
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

        # Save and load
        save_checkpoint(str(tmp_path), checkpoint_data)
        loaded = load_checkpoint(str(tmp_path))

        assert len(loaded["token_ids"]) == num_tokens
        assert loaded["samples_processed"] == 50000


class TestCheckpointIntegration:
    """Integration tests simulating resume scenarios."""

    def test_simulate_interrupt_and_resume(self, tmp_path):
        """Simulate processing interruption and resume."""
        # Simulate first run - process 100 samples, save checkpoint
        state = {
            "samples_processed": 100,
            "token_ids": [i for i in range(500)],  # 5 tokens per sample avg
            "labels_mask": [1, 0, 1, 0, 1] * 100,
            "document_boundaries": [(i * 5, (i + 1) * 5) for i in range(100)],
            "current_position": 500,
            "num_samples_skipped": 2,
            "per_dataset_counts": {"ds": 100},
            "per_dataset_tokens": {"ds": 500},
            "per_dataset_trainable_tokens": {"ds": 300},
            "per_dataset_filtered": {"ds": 2},
        }
        save_checkpoint(str(tmp_path), state)

        # Simulate resume - load checkpoint
        loaded = load_checkpoint(str(tmp_path))
        assert loaded is not None

        start_idx = loaded["samples_processed"]
        assert start_idx == 100

        # Continue processing from sample 100
        token_ids = loaded["token_ids"]
        for i in range(start_idx, 200):  # Process 100 more
            token_ids.extend([i * 5 + j for j in range(5)])

        # Final state
        assert len(token_ids) == 1000  # 500 + 500 new

    def test_no_checkpoint_fresh_start(self, tmp_path):
        """Test behavior when no checkpoint exists (fresh start)."""
        loaded = load_checkpoint(str(tmp_path))

        # Should start from beginning
        start_idx = 0 if loaded is None else loaded["samples_processed"]
        assert start_idx == 0

    def test_cleanup_after_success(self, tmp_path):
        """Test that checkpoint is cleaned up after successful completion."""
        # Save a checkpoint (simulating in-progress state)
        save_checkpoint(str(tmp_path), {"samples_processed": 1000})

        checkpoint_path = tmp_path / "_checkpoint.json"
        assert checkpoint_path.exists()

        # Simulate successful completion
        remove_checkpoint(str(tmp_path))

        # Checkpoint should be gone
        assert not checkpoint_path.exists()

        # Next run should start fresh
        loaded = load_checkpoint(str(tmp_path))
        assert loaded is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
