"""Frozen routing diagnostics must retain exact source and token alignment."""

import hashlib
import json

import numpy as np
import pytest
from scripts.miles import compare_inference_replay

from open_instruct.miles.configuration import benchmark
from open_instruct.miles.rollout import benchmark_route_reference


def source(tmp_path):
    record = {"index": 7, "group_index": 2, "prompt_tokens": 2, "response_tokens": 2, "tokens": [10, 11, 12, 13]}
    matrix = np.array([[[-1, -1], [0, 1]], [[-1, -1], [1, 2]], [[-1, -1], [2, 3]]], dtype=np.int32)
    np.savez_compressed(tmp_path / "routes.npz", routes=matrix)
    row = {
        **{key: record[key] for key in ("index", "group_index", "prompt_tokens", "response_tokens")},
        "source_index": 7,
        "copy_index": 0,
        "route_path": "routes.npz",
        "route_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
    }
    (tmp_path / "benchmark-replay-0.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "benchmark-summary.json").write_text(json.dumps({"reference_sha256": "captures"}))
    (tmp_path / "workflow.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "run-spec.json").write_text(json.dumps({"model": {"source": "checkpoint"}}))
    model = {"num_hidden_layers": 2, "num_experts_per_tok": 2, "n_routed_experts": 4, "dense_layers_indices": [0]}
    return record, row, matrix, model


def test_route_reference_preserves_all_conditioning_positions(tmp_path):
    record, row, matrix, model = source(tmp_path)
    root, rows, provenance = benchmark_route_reference.reference_routes(
        tmp_path, "captures", [record], 0, {"source": "checkpoint"}
    )
    actual = benchmark_route_reference.load_routes(root, rows[7], record, model)
    np.testing.assert_array_equal(actual, matrix)
    assert len(actual) == len(record["tokens"]) - 1
    assert provenance["repeat"] == 0
    assert rows[7] == row


@pytest.mark.parametrize("mutation", ["checkpoint", "capture", "incomplete", "duplicate", "copy", "length", "missing"])
def test_route_reference_rejects_mismatched_provenance(tmp_path, mutation):
    record, row, _, _ = source(tmp_path)
    model, digest = {"source": "checkpoint"}, "captures"
    rows = [row]
    if mutation == "checkpoint":
        model = {"source": "another-checkpoint"}
    elif mutation == "capture":
        digest = "different-tokens"
    elif mutation == "incomplete":
        (tmp_path / "workflow.json").write_text('{"status":"running"}')
    elif mutation == "duplicate":
        rows.append(row)
    elif mutation == "copy":
        row["copy_index"] = 1
    elif mutation == "length":
        row["prompt_tokens"] += 1
    elif mutation == "missing":
        rows.clear()
    (tmp_path / "benchmark-replay-0.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows))
    with pytest.raises(ValueError):
        benchmark_route_reference.reference_routes(tmp_path, digest, [record], 0, model)


@pytest.mark.parametrize("mutation", ["hash", "dtype", "length", "layers", "topk", "negative", "too_large"])
def test_route_reference_rejects_corrupt_matrices_before_core(tmp_path, mutation):
    record, row, matrix, model = source(tmp_path)
    if mutation == "dtype":
        matrix = matrix.astype(np.int64)
    elif mutation == "length":
        matrix = matrix[:-1]
    elif mutation == "layers":
        matrix = matrix[:, :1]
    elif mutation == "topk":
        matrix = matrix[..., :1]
    elif mutation == "negative":
        matrix[0, 1, 0] = -1
    elif mutation == "too_large":
        matrix[0, 1, 0] = 4
    row["route_sha256"] = "wrong" if mutation == "hash" else hashlib.sha256(matrix.tobytes()).hexdigest()
    np.savez_compressed(tmp_path / "routes.npz", routes=matrix)
    with pytest.raises(ValueError):
        benchmark_route_reference.load_routes(tmp_path, row, record, model)


def test_core_route_configuration_requires_explicit_scoring():
    with pytest.raises(ValueError, match="score_reference"):
        benchmark.parse({"score_route_reference": "/replay"})
    with pytest.raises(ValueError, match="score_route_reference"):
        benchmark.parse({"score_route_repeat": 1})
    with pytest.raises(ValueError, match="absolute"):
        benchmark.parse({"score_reference": True, "score_route_reference": "relative"})
    result = benchmark.parse({"score_reference": True, "score_route_reference": "/replay", "score_route_repeat": 1})
    assert result["score_route_repeat"] == 1


def test_replayed_core_scores_cannot_be_used_as_independent_reference(tmp_path):
    (tmp_path / "workflow.json").write_text('{"status":"complete"}')
    (tmp_path / "benchmark-summary.json").write_text(
        json.dumps({"reference_backend": "core", "reference_sha256": "captures", "routing": "serving_replay"})
    )
    with pytest.raises(ValueError, match="natural-routing"):
        compare_inference_replay.core_comparison(tmp_path, "captures", [], [])
