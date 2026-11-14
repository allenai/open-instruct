"""Weighted committed exposure, finite priority pools and asynchronous delivery."""

import heapq
import math
import random
from collections import Counter, deque
from types import SimpleNamespace

import pytest

from open_instruct import test_miles_indexed_samples, test_miles_prompt_replay, test_miles_run_spec
from open_instruct.miles.errors import InputError
from open_instruct.miles.rollout import pool_provider, pool_sampling

PRIORITY = "priority"


def provider(weights, sizes=None, seed=17, priority=True, replay=None):
    sizes = sizes or {name: 100 for name in weights}
    samples = [
        SimpleNamespace(prompt=f"{name}:{index}", metadata={"sampling_pool": name, PRIORITY: 1 if index < 4 else 2})
        for name, size in sizes.items()
        for index in range(size)
    ]
    sampling = {"policy": "exposure", "weights": weights}
    if priority:
        sampling["priority"] = {"field": PRIORITY}
    return pool_provider.PoolPromptProvider(
        samples,
        pools={name: {"label": name} for name in weights},
        seed=seed,
        shuffle=True,
        identity="weighted-fixture",
        sampling=sampling,
        replay=replay,
    )


def feedback(p, sample, passed):
    p.feedback(
        [
            {
                "kind": "group",
                "pool_selection_id": sample.metadata["pool_selection_id"],
                "filter_decision": "passed" if passed else "filtered",
                "responses": [
                    {"reward": i % 2 if passed else 0, "validity": {"valid": True}, "response_tokens": 1}
                    for i in range(8)
                ],
            }
        ]
    )


def test_commit_uses_weighted_active_snapshot_and_is_exactly_once():
    p = provider({"dolci": 500, "small": 3, "empty": 2}, {"dolci": 1000, "small": 1, "empty": 0})
    initial = []
    while p.providers["small"].available():
        sample = p.next_samples(1)[0]
        initial.append(sample.metadata["pool_selection_id"])
    assert all(p.pending[identity]["active"] == ["dolci", "small"] for identity in initial)
    p.commit(0, initial)
    assert p.targets["dolci"] == pytest.approx(len(initial) * 500 / 503)
    assert p.targets["small"] == pytest.approx(len(initial) * 3 / 503)
    assert p.targets["empty"] == 0
    before = p.state_dict()
    p.commit(0, initial)
    assert p.state_dict() == before
    sample = p.next_samples(1)[0]
    p.commit(1, [sample.metadata["pool_selection_id"]])
    assert p.targets["dolci"] == pytest.approx(before["targets"]["dolci"] + 1)
    assert p.targets["small"] == before["targets"]["small"]
    assert sum(p.targets.values()) == pytest.approx(sum(p.trained.values()))


def test_priority_boundaries_exhaustion_restore_and_selection_rollback():
    options = ({"a": 1, "b": 0.01}, {"a": 19, "b": 7})
    p = provider(*options)
    first = p.next_samples(5)
    feedback(p, first[0], False)
    p.commit(0, [sample.metadata["pool_selection_id"] for sample in first[1:]])
    restored = provider(*options)
    restored.load_state_dict(p.state_dict())
    snapshot = p.selection_state_dict()
    expected = p.next_samples(3)
    p.load_selection_state_dict(snapshot)
    assert p.next_samples(3) == expected == restored.next_samples(3)
    tail = p.next_samples(18)
    assert tail == restored.next_samples(18)
    all_samples = first + expected + tail
    assert len({sample.prompt for sample in all_samples}) == 26
    for pool in options[0]:
        priorities = [
            sample.metadata["sampling_priority_group"]
            for sample in all_samples
            if sample.metadata["sampling_pool"] == pool
        ]
        assert priorities == sorted(priorities)
        assert set(priorities) == {1, 2}
    assert not p.available()
    with pytest.raises(StopIteration):
        p.next_samples(1)
    changed = provider({"a": 1, "b": 0.02}, options[1])
    with pytest.raises(ValueError, match="identity"):
        changed.load_state_dict(restored.state_dict())


def test_small_baseline_is_not_forced_to_absolute_twenty_to_one_ratio():
    weights = {"dolci": 500, "bulk": 497, "tiny": 3}
    sampling = pool_sampling.validate({"policy": "exposure", "weights": weights}, weights)
    relative = pool_sampling.draw_weights(
        sampling, list(weights), trained=Counter(), targets=Counter(), terminal=Counter(), pending={}
    )
    shares = [weight / sum(relative) for weight in relative]
    assert shares == pytest.approx([0.5, 0.497, 0.003])
    assert relative[0] / relative[2] == pytest.approx(500 / 3)
    # A correctly proportioned pending backlog has no relative deficit.
    pending = {
        i: {"pool": pool}
        for pool, count, start in [("dolci", 500, 0), ("bulk", 497, 500), ("tiny", 3, 997)]
        for i in range(start, start + count)
    }
    backlog = pool_sampling.draw_weights(
        sampling, list(weights), trained=Counter(), targets=Counter(), terminal=Counter(), pending=pending
    )
    assert backlog == pytest.approx(relative)


def test_no_weights_keep_legacy_validation_identity_and_draw_math():
    sampling = {"policy": "exposure", "priority": {"field": PRIORITY}}
    assert pool_sampling.validate(sampling, {"a": {}, "b": {}}) == sampling
    p = provider({"a": 1, "b": 1})
    old = pool_provider.PoolPromptProvider(
        [SimpleNamespace(metadata={"sampling_pool": name, PRIORITY: 1}) for name in ["a", "b"]],
        pools={"a": {"label": "a"}, "b": {"label": "b"}},
        seed=17,
        shuffle=True,
        identity="fixture",
        sampling=sampling,
    )
    assert old.identity == {
        "version": 4,
        "manifest": "fixture",
        "pools": {"a": {"label": "a"}, "b": {"label": "b"}},
        "seed": 17,
        "shuffle": True,
        "replay": None,
        "sampling": sampling,
    }
    trained, targets, terminal = Counter(a=120, b=80), Counter(a=100, b=100), Counter(a=200, b=100)
    pending = {i: {"pool": "a" if i % 2 else "b"} for i in range(100)}
    result = pool_sampling.draw_weights(
        sampling, ["a", "b"], trained=trained, targets=targets, terminal=terminal, pending=pending
    )
    expected = []
    for name in ["a", "b"]:
        delivery = max(0.05, (trained[name] + 1) / (terminal[name] + 2))
        deficit = targets[name] - trained[name] - 50 * delivery
        expected.append(max(0.25, min(5.0, math.exp(max(-3, min(3, deficit / 64))) / delivery)))
    assert result == expected
    assert "weights" in p.identity["sampling"] and "weights" not in old.identity["sampling"]
    old.next_samples(1)
    old.commit(0, [0])
    assert old.targets == {"a": 0.5, "b": 0.5}


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 10**1000])
def test_invalid_exposure_weights_fail_before_sampling(value):
    with pytest.raises(InputError, match="finite positive"):
        pool_sampling.validate({"policy": "exposure", "weights": {"a": value}}, {"a": {}})


def test_weight_schema_and_extreme_finite_numbers():
    for weights in [None, {}, {"other": 1}]:
        with pytest.raises(InputError, match="exactly"):
            pool_sampling.validate({"policy": "exposure", "weights": weights}, {"a": {}})
    for weights in [{"a": 1e308, "b": 1e-308}, {"a": 1, "b": 1e-323}]:
        with pytest.raises(InputError, match="representable|extreme"):
            pool_sampling.validate({"policy": "exposure", "weights": weights}, {"a": {}, "b": {}})
    sampling = pool_sampling.validate({"policy": "exposure", "weights": {"a": 1e308, "b": 1e308}}, {"a": {}, "b": {}})
    assert pool_sampling.target_shares(sampling["weights"], ["a", "b"]) == {"a": 0.5, "b": 0.5}
    assert pool_sampling.validate({"policy": "exposure"}, {"a": {}}) == {"policy": "exposure"}
    # Attempted weighted selection retains its original outcome-independent ratio.
    assert pool_sampling.draw_weights(
        {"policy": "weighted", "weights": {"a": 3, "b": 1}},
        ["a", "b"],
        trained=Counter(a=999),
        targets=Counter(),
        terminal=Counter(),
        pending={},
    ) == [1, 1 / 3]


def test_public_run_spec_prepares_weighted_exposure_with_priority(tmp_path):
    data = {
        "sources": [{"name": "one", "prompt_data": "/p.jsonl", "reward_config": "/r.json"}],
        "pools": {"dolci": {"label": "dolci"}, "tiny": {"label": "tiny"}},
        "sampling": {"policy": "exposure", "weights": {"dolci": 500, "tiny": 3}, "priority": {"field": PRIORITY}},
    }
    run = test_miles_run_spec.spec(
        tmp_path, data=data, records={"enabled": True, "root": "/records"}, **{"async": {"fully_async": True}}
    )
    assert run.compile().core.prompt_provider == "pools"
    assert run.data["sampling"] == data["sampling"]


def test_weighted_exposure_can_credit_replay_without_priority():
    p = provider(
        {"a": 3, "b": 1},
        {"a": 12, "b": 12},
        priority=False,
        replay={"fraction": 0.5, "window": 1, "max_successes": 0, "truncation": "any", "length_drop": 0.1},
    )
    seen = set()
    observed_pools = set()
    replays = 0
    for step in range(100):
        if not p.available():
            break
        sample = p.next_samples(1)[0]
        pool = sample.metadata["sampling_pool"]
        if sample.metadata.get("prompt_replay_review"):
            replays += 1
            assert sample.prompt in seen
            record = test_miles_prompt_replay.record(sample, rewards=(0, 1, 1, 1), tokens=30)
        else:
            assert sample.prompt not in seen
            seen.add(sample.prompt)
            first = pool not in observed_pools
            observed_pools.add(pool)
            record = test_miles_prompt_replay.record(
                sample,
                rewards=(0, 0, 0, 0) if first else (0, 1, 1, 1),
                caps=4 if first else 0,
                tokens=100 if first else 30,
            )
        p.feedback([record])
        if record["filter_decision"] != "filtered":
            p.commit(step, [sample.metadata["pool_selection_id"]])
    assert len(seen) == 24 and replays > 0
    assert not p.available()
    assert sum(p.targets.values()) == pytest.approx(sum(p.trained.values()))


def test_disk_priority_provider_matches_memory_with_weighted_feedback(tmp_path):
    catalog, samples, pools = test_miles_indexed_samples.fixture(tmp_path)
    options = dict(
        pools=pools,
        seed=17,
        shuffle=True,
        identity="weighted-disk",
        sampling={
            "policy": "exposure",
            "weights": {"math": 497, "code": 3, "empty": 500},
            "priority": {"field": PRIORITY},
        },
    )
    disk = pool_provider.PoolPromptProvider(catalog, **options)
    memory = pool_provider.PoolPromptProvider(samples, **options)
    try:
        for update in range(48):
            sample, expected = disk.next_samples(1)[0], memory.next_samples(1)[0]
            assert sample == expected
            passed = update % 5 != 0
            for current in [disk, memory]:
                feedback(current, sample, passed)
                if passed:
                    current.commit(update, [sample.metadata["pool_selection_id"]])
            assert disk.trained == memory.trained and disk.targets == memory.targets
            if update == 11:
                restored = pool_provider.PoolPromptProvider(samples, **options)
                restored.load_state_dict(disk.state_dict())
                memory = restored
        assert not disk.available() and not memory.available()
    finally:
        disk.close()


@pytest.mark.parametrize("seed", [1, 17, 49])
@pytest.mark.parametrize("dolci_latency", [3, 5])
def test_async_unequal_delivery_reaches_backbone_and_point_three_percent(seed, dolci_latency):
    weights = {"dolci": 500, "bulk": 497, "tiny": 3}
    p = provider(weights, {name: 50000 for name in weights}, seed=seed)
    rng = random.Random(seed + 1000)
    running = []
    ready = deque()
    blocked = deque()
    staged = []
    training = None
    train_end = math.inf
    now = 0.0
    update = sequence = 0
    max_pending = 0
    while update < 200:
        while True:
            changed = False
            while ready and len(staged) < 128:
                staged.append(ready.popleft())
                changed = True
            if training is None and len(staged) == 128:
                training, staged = staged, []
                train_end = now + 8
                changed = True
            while blocked and (len(staged) < 128 or len(ready) < 512):
                (staged if len(staged) < 128 else ready).append(blocked.popleft())
                changed = True
            if not changed:
                break
        if not blocked:
            while len(running) < 128:
                sample = p.next_samples(1)[0]
                latency = {"dolci": dolci_latency, "bulk": 1, "tiny": 4}[sample.metadata["sampling_pool"]]
                heapq.heappush(running, (now + latency * rng.uniform(0.9, 1.1), sequence, sample))
                sequence += 1
        max_pending = max(max_pending, len(p.pending))
        assert running or training is not None
        next_completion = running[0][0] if running else math.inf
        if train_end <= next_completion:
            now = train_end
            p.commit(update, training)
            update += 1
            training, train_end = None, math.inf
        else:
            now, _, sample = heapq.heappop(running)
            accepted = rng.random() < {"dolci": 0.55, "bulk": 0.7, "tiny": 0.25}[sample.metadata["sampling_pool"]]
            feedback(p, sample, accepted)
            if accepted:
                blocked.append(sample.metadata["pool_selection_id"])
        assert sum(p.terminal.values()) + len(p.pending) == p.draws
    total = sum(p.trained.values())
    assert total == 200 * 128
    assert abs(p.trained["dolci"] / total - 0.5) < 0.008
    assert abs(p.trained["tiny"] / total - 0.003) < 0.0006
    assert max_pending >= 512
    assert p.providers["dolci"].offset > 4 and p.providers["tiny"].offset > 4
    assert all(p.providers[name].available() for name in weights)
