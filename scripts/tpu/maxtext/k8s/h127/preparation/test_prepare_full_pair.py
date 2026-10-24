import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
from prepare_full_pair import prepare
from pyarrow import parquet as pq

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for arm in ("a", "b", "bad"):
        (root / arm).mkdir()
    for shard in range(3):
        rows = [
            {
                "id": str(i),
                "source": "source" + str(i % 2),
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": "<think> Okay</think>answer"},
                ],
            }
            for i in range(shard * 5, (shard + 1) * 5)
        ]
        pq.write_table(pa.Table.from_pylist(rows), root / "a" / f"{shard}.parquet")
        for row in rows:
            row["messages"][0]["content"] += " Present your answer."
        pq.write_table(pa.Table.from_pylist(rows), root / "b" / f"{shard}.parquet")
        rows[0]["messages"][1]["content"] += " forbidden change"
        pq.write_table(pa.Table.from_pylist(rows), root / "bad" / f"{shard}.parquet")
    receipt = prepare(root / "a", root / "b", root / "out", shards=4, seed=0, expected_rows=15)
    assert receipt["changed_records"] == 15
    ids = []
    for path in sorted((root / "out/control").glob("*.parquet")):
        ids += pq.read_table(path)["id"].to_pylist()
    assert ids == [str(i) for i in np.random.default_rng(0).permutation(15)]
    try:
        prepare(root / "a", root / "bad", root / "rejected", shards=4, seed=0, expected_rows=15)
    except AssertionError as error:
        assert "Non-user message changed" in str(error)
    else:
        raise AssertionError("Assistant corruption was not rejected")
print("Exact global order, pairing, uneven shards, full coverage and assistant-corruption checks passed")
