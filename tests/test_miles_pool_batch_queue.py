"""Pool packing must survive out-of-order completion without losing groups."""

from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.errors import InputError
from open_instruct.miles.rollout.pool_batch_queue import PoolBatchQueue


def group(pool, identity):
    return SimpleNamespace(group=[SimpleNamespace(metadata={"pool": pool})], identity=identity)


def queue(fraction=1):
    return PoolBatchQueue(
        size=2,
        fraction=fraction,
        field="pool",
        values=["if", "math"],
        seed=17,
        version=lambda: 0,
        stale=lambda entry: entry.identity < 0,
    )


def test_asynchronous_completions_wait_for_full_pool_and_preserve_leftovers():
    q = queue()
    first, second, third, fourth = [group(pool, i) for i, pool in enumerate(["if", "math", "math", "if"])]
    q.append(first)
    q.append(second)
    assert not q
    q.append(third)
    assert q
    assert q.pop(0) is second
    q.append(fourth)
    assert q
    assert q.pop(0) is third
    assert q
    assert q.pop(0) is first
    assert q
    assert q.pop(0) is fourth
    assert len(q) == 0
    assert q.pool_batches == 2


def test_zero_fraction_preserves_fifo():
    q = queue(0)
    entries = [group(pool, i) for i, pool in enumerate(["if", "math", "math", "if"])]
    for entry in entries:
        q.append(entry)
    result = []
    while q:
        result.append(q.pop(0))
    assert result == entries
    assert q.pool_batches == 0


def test_stale_groups_remain_visible_to_native_filter():
    q = queue()
    stale = group("if", -1)
    q.append(stale)
    q.append(group("math", 1))
    assert q
    assert q.pop(0) is stale
    assert not q
    assert len(q) == 1


def test_no_loss_at_pigeonhole_capacity():
    q = queue()
    entries = [group("if" if i % 3 else "math", i) for i in range(600)]
    consumed = []
    for entry in entries:
        q.append(entry)
        while q:
            consumed.append(q.pop(0))
        assert len(q) <= 2  # pools * (batch size - 1)
    assert sorted(entry.identity for entry in consumed + list(q)) == list(range(600))
    for offset in range(0, len(consumed), 2):
        assert q.pool(consumed[offset]) == q.pool(consumed[offset + 1])


def test_pool_labels_are_explicit_and_consistent():
    q = queue()
    with pytest.raises(ValueError, match="declared"):
        q.append(group("unknown", 1))
    assert len(q) == 0


@pytest.mark.parametrize("fraction", [-0.1, 1.1, True, float("nan")])
def test_invalid_fraction(fraction):
    with pytest.raises(InputError):
        CoreConfig(pool_batch_fraction=fraction, pool_batch_values=["if", "math"])


def test_pool_values_required():
    with pytest.raises(InputError, match="explicit"):
        RunConfig(CoreConfig(pool_batch_fraction=1), {}).validate()


def test_dotted_source_metadata():
    q = PoolBatchQueue(
        size=2,
        fraction=1,
        field="workbench.source.candidate_id",
        values=["if"],
        seed=17,
        version=lambda: 0,
        stale=lambda entry: False,
    )
    for i in range(2):
        q.append(SimpleNamespace(group=[SimpleNamespace(metadata={"workbench": {"source": {"candidate_id": "if"}}})]))
    assert q
    assert q.pool(q.pop(0)) == "if"
