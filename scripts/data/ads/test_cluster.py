"""Offline CPU clustering preserves exact prompt identities."""

import json
import sys

import numpy as np
import pytest
from scripts.data.ads import cluster


def test_clustering_preserves_prompt_order_and_is_repeatable(tmp_path, monkeypatch):
    vectors = np.random.default_rng(17).normal(size=(48, 16)).astype("float32")
    np.save(tmp_path / "embeddings.npy", vectors)
    rows = [{"row_index": i, "prompt_id": str(i), "difficulty": i / 10} for i in range(48)]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sys, "argv", ["cluster", str(tmp_path), "--clusters", "4", "--components", "8"])
    cluster.main()
    first = (tmp_path / "ads-table.jsonl").read_bytes()
    cluster.main()
    assert (tmp_path / "ads-table.jsonl").read_bytes() == first
    table = [json.loads(line) for line in first.splitlines()]
    assert [r["prompt_id"] for r in table] == [str(i) for i in range(48)]
    assert {r["cluster_id"] for r in table} == {0, 1, 2, 3}
    rows[0]["prompt_id"] = rows[1]["prompt_id"]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="identities"):
        cluster.main()
