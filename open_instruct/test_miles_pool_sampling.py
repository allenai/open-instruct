"""The same labeled inventory supports independent finite selection policies."""

import copy
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from open_instruct import test_miles_pool_exposure, test_miles_prompt_replay, test_miles_run_spec
from open_instruct.miles.datasets import run_data
from open_instruct.miles.rollout import pool_provider, pool_sampling

POLICIES = ("shuffle", "uniform", "weighted", "exposure")


def make(policy, sizes=(80, 240), replay=None):
    samples = [
        SimpleNamespace(
            prompt=f"{name}-{i}",
            metadata={
                "sampling_pool": name,
                "pool_id": name,
                "mixture_source": "same-source",
                "verifiers": [{"name": "math", "target": "42"}],
            },
        )
        for name, count in zip(("math", "if"), sizes, strict=True)
        for i in range(count)
    ]
    sampling = {"policy": policy}
    if policy == "weighted":
        sampling["weights"] = {"math": 3, "if": 1, "empty": 9}
    return pool_provider.PoolPromptProvider(
        samples,
        pools={n: {"label": n} for n in ("math", "if", "empty")},
        seed=17,
        shuffle=True,
        identity="fixture",
        sampling=sampling,
        replay=replay,
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_exhausts_each_fresh_row_once_and_restores(policy):
    p = make(policy)
    first = p.next_samples(17)
    p.commit(0, [s.metadata["pool_selection_id"] for s in first[:8]])
    restored = make(policy)
    restored.load_state_dict(p.state_dict())
    tail = p.next_samples(303)
    assert tail == restored.next_samples(303)
    assert len({s.prompt for s in first + tail}) == 320
    assert p.proposed == {"math": 80, "if": 240}
    assert not p.available()
    with pytest.raises(StopIteration):
        p.next_samples(1)
    assert any(k.endswith("/target") for k in p.metrics()) == (policy == "exposure")


@pytest.mark.parametrize("policy,expected", [("shuffle", 0.25), ("uniform", 0.5), ("weighted", 0.75)])
def test_draw_policy_and_outcome_independence(policy, expected):
    p, other = make(policy, (5000, 15000)), make(policy, (5000, 15000))
    count = Counter()
    for i in range(2000):
        a, b = p.next_samples(1)[0], other.next_samples(1)[0]
        assert a == b
        count[a.metadata["sampling_pool"]] += 1
        p.commit(i, [a.metadata["pool_selection_id"]])
        test_miles_pool_exposure.discard(other, b)
    assert abs(count["math"] / 2000 - expected) < 0.04
    assert p.proposed["empty"] == 0


def test_global_shuffle_is_one_permutation_not_round_robin_pools():
    expected = [f"{name}-{i}" for name, count in [("math", 80), ("if", 240)] for i in range(count)]
    random.Random(17).shuffle(expected)
    assert [s.prompt for s in make("shuffle").next_samples(320)] == expected


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_identity_and_atomic_selection_rollback(policy):
    p = make(policy)
    p.next_samples(5)
    snapshot = p.selection_state_dict()
    expected = p.next_samples(10)
    p.load_selection_state_dict(snapshot)
    assert p.next_samples(10) == expected
    other = make("uniform" if policy != "uniform" else "shuffle")
    with pytest.raises(ValueError, match="identity"):
        other.load_state_dict(p.state_dict())


@pytest.mark.parametrize("policy", POLICIES)
def test_selection_rollback_does_not_copy_or_replace_outstanding_metadata(policy):
    class DoNotCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("Selection traversed outstanding prompt metadata")

    p = make(policy)
    p.next_samples(5)
    existing = dict(p.pending)
    for entry in existing.values():
        entry["metadata"]["large_target"] = DoNotCopy()
    # The bridge itself deep-copies the snapshot before attempting delivery.
    snapshot = copy.deepcopy(p.selection_state_dict())
    expected = p.next_samples(10)
    p.load_selection_state_dict(snapshot)
    assert p.pending.keys() == existing.keys()
    assert all(p.pending[key] is value for key, value in existing.items())
    assert p.next_samples(10) == expected


@pytest.mark.parametrize("policy", POLICIES)
def test_replay_is_optional_and_preserves_fresh_inventory(policy):
    replay = dict(fraction=0.5, window=1, max_successes=0, truncation="any", length_drop=0.1)
    p = make(policy, (12, 12), replay)
    baseline = make(policy, (12, 12))
    baseline_order = [s.prompt for s in baseline.next_samples(24)]
    fresh, reviews, candidates = [], [], set()
    for step in range(100):
        if not p.available():
            break
        sample = p.next_samples(1)[0]
        pool = sample.metadata["sampling_pool"]
        if sample.metadata["prompt_replay_review"]:
            reviews.append(sample)
            assert sample.prompt in candidates
            assert sample.metadata["verifiers"] == [{"name": "math", "target": "42"}]
            record = test_miles_prompt_replay.record(sample, rewards=(0, 1, 1, 1), tokens=30)
        else:
            fresh.append(sample.prompt)
            first = not any(prompt.startswith(pool + "-") for prompt in candidates)
            if first:
                candidates.add(sample.prompt)
            record = test_miles_prompt_replay.record(
                sample,
                rewards=(0, 0, 0, 0) if first else (0, 1, 1, 1),
                caps=4 if first else 0,
                tokens=100 if first else 50,
            )
        p.feedback([record])
        if record["filter_decision"] != "filtered":
            p.commit(step, [sample.metadata["pool_selection_id"]])
        if step == 5:
            restored = make(policy, (12, 12), replay)
            restored.load_state_dict(p.state_dict())
            assert restored.state_dict() == p.state_dict()
    assert not p.available()
    assert len(fresh) == len(set(fresh)) == 24
    assert reviews
    assert len(reviews) <= p.draws * 0.5
    if policy == "shuffle":
        assert fresh == baseline_order


@pytest.mark.parametrize(
    "sampling",
    [
        {"policy": "unknown"},
        {"policy": "uniform", "weights": {"math": 1}},
        {"policy": "weighted"},
        {"policy": "weighted", "weights": {"other": 1}},
        *[{"policy": "weighted", "weights": {"math": x}} for x in [0, -1, True, float("nan"), float("inf")]],
        {"policy": "shuffle", "typo": 1},
    ],
)
def test_invalid_sampling_is_rejected(sampling):
    with pytest.raises(ValueError):
        pool_sampling.validate(sampling, {"math": {"label": "math"}})


@pytest.mark.parametrize("policy", POLICIES)
def test_policy_compiles_through_public_run_spec(policy, tmp_path):
    data = {
        "sources": [{"name": "one", "prompt_data": "/p.jsonl", "reward_config": "/r.json"}],
        "pools": {"math": {"label": "math"}},
        "sampling": {"policy": policy},
    }
    if policy == "weighted":
        data["sampling"]["weights"] = {"math": 3}
    spec = test_miles_run_spec.spec(
        tmp_path,
        data=data,
        records={"enabled": True, "root": "/records"},
        core={"prepared_batch_overlap": False},
        **{"async": {"fully_async": True}},
    )
    assert spec.compile().core.prompt_provider == "pools"
    assert spec.data["sampling"] == data["sampling"]
    bad = copy.deepcopy(data)
    bad.pop("pools")
    with pytest.raises(ValueError, match="requires data.pools"):
        run_data.validate_data(bad)
