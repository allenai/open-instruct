"""Disk and memory selection must agree independently of prefetch completion order."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.datasets import indexed_rows
from open_instruct.miles.rollout import indexed_samples, pool_provider


def fixture(tmp_path, *, cache_rows=8, cache_bytes=8192):
    pools = {"math": {}, "code": {}, "empty": {}}
    sources, samples = [], []
    for source in ("one", "two"):
        path = tmp_path / f"{source}.jsonl"
        rows = [
            {
                "input": f"{source}-{i}",
                "label": str(i),
                "metadata": {
                    "sampling_pool": "math" if i % 3 else "code",
                    "run_prompt_tokens": i + 1,
                    "sample_id": f"{source}-{i}",
                    "verifiers": [{"name": "math", "target": str(i)}],
                    "payload": [{"x": [i] * 16}] * 8,
                },
            }
            for i in range(24)
        ]
        raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
        path.write_bytes(raw)
        sources.append(
            {"name": source, "path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows)}
        )
        samples.extend(
            SimpleNamespace(prompt=r["input"], label=r["label"], metadata=r["metadata"] | {"mixture_source": source})
            for r in rows
        )
    catalog = indexed_samples.Catalog(
        tmp_path, sources, pools, sample_type=SimpleNamespace, cache_rows=cache_rows, cache_bytes=cache_bytes
    )
    return catalog, samples, pools


@pytest.mark.parametrize("policy", ["shuffle", "uniform", "weighted", "exposure"])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_disk_selection_matches_memory_and_resumes(tmp_path, policy, shuffle, replay):
    catalog, samples, pools = fixture(tmp_path)
    sampling = {"policy": policy}
    if policy == "weighted":
        sampling["weights"] = {"math": 3, "code": 1, "empty": 1}
    options = dict(
        pools=pools,
        seed=17,
        shuffle=shuffle,
        identity="same-content",
        sampling=sampling,
        replay={"fraction": 0.1} if replay else None,
    )
    disk, memory = (
        pool_provider.PoolPromptProvider(catalog, **options),
        pool_provider.PoolPromptProvider(samples, **options),
    )
    try:
        for _ in range(5):
            assert disk.next_samples(3) == memory.next_samples(3)
        state = disk.state_dict()
        snapshot = copy.deepcopy(disk.selection_state_dict())
        expected = disk.next_samples(4)
        disk.load_selection_state_dict(snapshot)
        assert disk.next_samples(4) == expected
        disk.load_state_dict(state)
        assert disk.next_samples(33) == memory.next_samples(33)
        assert not disk.available()
        with catalog.lock:
            assert catalog.bytes <= catalog.cache_bytes
            assert len(catalog.cache) <= catalog.cache_rows
            assert len(catalog.pending) <= catalog.cache_rows
        # Generated metadata cannot contaminate an immutable disk template.
        sample = catalog.sample(0)
        sample.metadata["payload"][0]["x"][0] = -1
        assert catalog.sample(0).metadata["payload"][0]["x"][0] == 0
    finally:
        disk.close()


def test_index_integrity_and_prompt_budget(tmp_path):
    catalog, _, _ = fixture(tmp_path)
    source = catalog.sources[0]
    catalog.close()
    path = tmp_path / source["path"]
    with pytest.raises(ValueError, match="length limit"):
        indexed_rows.open_index(path, source["sha256"], source["rows"], 3)
    path.write_bytes(path.read_bytes().replace(b"one-0", b"bad-0"))
    with pytest.raises(ValueError, match="hash changed"):
        indexed_rows.open_index(path, source["sha256"], source["rows"])


def test_prefetch_can_finish_out_of_order_without_selecting(tmp_path):
    catalog, samples, _ = fixture(tmp_path, cache_rows=64, cache_bytes=1 << 20)
    try:
        catalog.prefetch(reversed(range(48)))
        assert [catalog.sample(i).prompt for i in range(48)] == [s.prompt for s in samples]
        catalog.executor.shutdown(wait=True)
        assert catalog.sample(0).prompt == samples[0].prompt
        assert catalog.metrics()["cache_hits"] > 0
    finally:
        catalog.close()


def test_corrupted_index_is_rejected(tmp_path):
    catalog, _, _ = fixture(tmp_path)
    source = catalog.sources[0]
    catalog.close()
    index = next(tmp_path.glob("one.jsonl.*.index.bin"))
    index.write_bytes(index.read_bytes()[:-1])
    with pytest.raises(ValueError, match="incomplete or changed"):
        indexed_rows.open_index(tmp_path / source["path"], source["sha256"], source["rows"])


def test_oversized_rows_bypass_cache_without_changing_delivery(tmp_path):
    catalog, samples, _ = fixture(tmp_path, cache_rows=2, cache_bytes=1)
    try:
        catalog.prefetch(range(48))
        assert catalog.sample(1) == samples[1]
        catalog.executor.shutdown(wait=True)
        assert catalog.metrics()["cache_bytes"] == 0
        assert catalog.metrics()["cache_rows"] == 0
    finally:
        catalog.close()
