"""Read exact frozen replay routes for an optional forward-only Core diagnostic."""

import hashlib
import json
from pathlib import Path

import numpy as np


def reference_routes(directory, reference_sha256, records, repeat, model):
    root = Path(directory)
    summary = json.loads((root / "benchmark-summary.json").read_text())
    if (
        json.loads((root / "workflow.json").read_text())["status"] != "complete"
        or summary.get("reference_sha256") != reference_sha256
        or json.loads((root / "run-spec.json").read_text())["model"] != model
    ):
        raise ValueError("Core routing replay requires a completed replay of the same checkpoint and captures")
    path = root / f"benchmark-replay-{repeat}.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    by_index = {row["index"]: row for row in rows}
    if len(by_index) != len(rows) or set(by_index) != {record["index"] for record in records}:
        raise ValueError("Core routing replay requires exactly one route record per original capture")
    for record in records:
        row = by_index[record["index"]]
        if (
            row.get("source_index") != record["index"]
            or row.get("copy_index") != 0
            or any(row[key] != record[key] for key in ("group_index", "prompt_tokens", "response_tokens"))
            or not row.get("route_path")
            or not row.get("route_sha256")
        ):
            raise ValueError("Core routing replay trajectory identity or route metadata differs")
    return (
        root,
        by_index,
        {
            "directory": str(root),
            "repeat": repeat,
            "scores_sha256": hashlib.sha256(raw).hexdigest(),
            "reference_sha256": reference_sha256,
        },
    )


def load_routes(root, row, record, model_config):
    path = (root / row["route_path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Core routing replay artifact must be inside the replay directory")
    with np.load(path, allow_pickle=False) as archive:
        matrix = archive["routes"]
    expected = (len(record["tokens"]) - 1, model_config["num_hidden_layers"], model_config["num_experts_per_tok"])
    if (
        matrix.dtype != np.int32
        or matrix.shape != expected
        or hashlib.sha256(matrix.tobytes()).hexdigest() != row["route_sha256"]
    ):
        raise ValueError("Core routing replay artifact has inconsistent shape, dtype or hash")
    dense = model_config.get("dense_layers_indices") or []
    selected = matrix[:, [layer for layer in range(expected[1]) if layer not in dense]]
    if (selected < 0).any() or (selected >= model_config["n_routed_experts"]).any():
        raise ValueError("Core routing replay artifact contains an out-of-range expert ID")
    # The native trainer pads each document's unscored final token. Retain the
    # full input sequence and delegate that padding to its router_routes helper.
    return matrix
