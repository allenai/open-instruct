"""Priority ordering composes with allocation without changing its accounting."""

import copy
import json
import random
from types import SimpleNamespace

import pytest

from open_instruct import test_miles_pool_exposure, test_miles_run_data, test_miles_run_spec
from open_instruct.miles.rollout import pool_provider, pool_sampling

environment = test_miles_run_data.environment


def provider(policy="exposure", *, size=120, shuffle=True, seed=17, field="priority", replay=None):
    samples = [
        SimpleNamespace(
            prompt=f"{pool}-{i}",
            metadata={
                "sampling_pool": pool,
                "mixture_source": pool,
                "priority": 4 - i % 4,
                "alternate_priority": 4 - i % 4,
            },
        )
        for pool in ("dolci", "component")
        for i in range(size)
    ]
    return pool_provider.PoolPromptProvider(
        samples,
        pools={pool: {} for pool in ("dolci", "component", "empty")},
        seed=seed,
        shuffle=shuffle,
        identity="fixture",
        replay=replay,
        sampling={"policy": policy, "priority": {"field": field}},
    )


@pytest.mark.parametrize("policy", ["uniform", "exposure"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_each_pool_progresses_independently_and_is_finite(policy, shuffle):
    samples = [
        SimpleNamespace(prompt=f"{pool}-{group}-{i}", metadata={"sampling_pool": pool, "priority": group})
        for pool in ("dolci", "component")
        for group in range(1, 5)
        for i in range(1 if pool == "component" and group == 1 else 10)
    ]
    p = pool_provider.PoolPromptProvider(
        samples,
        pools={pool: {} for pool in ("dolci", "component")},
        seed=17,
        shuffle=shuffle,
        identity="unequal-strata",
        sampling={"policy": policy, "priority": {"field": "priority"}},
    )
    rows = p.next_samples(71)
    assert len({row.prompt for row in rows}) == 71
    assert not p.available()
    for name in ("dolci", "component"):
        priorities = [s.metadata["priority"] for s in rows if s.metadata["sampling_pool"] == name]
        assert priorities == sorted(priorities)
    first_component_2 = next(
        i for i, s in enumerate(rows) if s.metadata["sampling_pool"] == "component" and s.metadata["priority"] == 2
    )
    assert any(
        s.metadata["sampling_pool"] == "dolci" and s.metadata["priority"] == 1 for s in rows[first_component_2 + 1 :]
    )
    assert all(s.metadata["sampling_priority_group"] == s.metadata["priority"] for s in rows)
    with pytest.raises(StopIteration):
        p.next_samples(1)


def test_seeded_within_stratum_shuffle_and_resume_with_pending_work():
    p = provider("uniform")
    initial = p.next_samples(17)
    p.commit(0, [row.metadata["pool_selection_id"] for row in initial[:8]])
    restored = provider("uniform")
    restored.load_state_dict(copy.deepcopy(p.state_dict()))
    expected = p.next_samples(60)
    assert restored.next_samples(60) == expected
    assert restored.trained == p.trained
    assert all("sampling_priority_group" in row["metadata"] for row in restored.pending.values())
    snapshot = p.selection_state_dict()
    tail = p.next_samples(12)
    p.load_selection_state_dict(snapshot)
    assert p.next_samples(12) == tail
    a = provider("uniform", seed=1).next_samples(240)
    b = provider("uniform", seed=2).next_samples(240)
    assert [r.prompt for r in a if r.metadata["priority"] == 1] != [r.prompt for r in b if r.metadata["priority"] == 1]
    changed = provider("uniform", field="alternate_priority")
    with pytest.raises(ValueError, match="identity"):
        changed.load_state_dict(p.state_dict())


def test_exposure_corrects_unequal_acceptance_across_priority_transitions():
    p = provider(size=12000)
    rng = random.Random(3)
    priorities = {"dolci": [], "component": []}
    for update in range(120):
        accepted = []
        while len(accepted) < 32:
            row = p.next_samples(1)[0]
            name = row.metadata["sampling_pool"]
            priorities[name].append(row.metadata["priority"])
            if rng.random() < {"dolci": 0.8, "component": 0.2}[name]:
                accepted.append(row.metadata["pool_selection_id"])
            else:
                test_miles_pool_exposure.discard(p, row)
        p.commit(update, accepted)
    assert abs(p.trained["component"] / sum(p.trained.values()) - 0.5) < 0.025
    assert p.proposed["component"] > 3 * p.proposed["dolci"]
    assert max(priorities["component"]) > 1
    for values in priorities.values():
        assert values == sorted(values)


@pytest.mark.parametrize("value", [None, True, 0, 5, "1"])
def test_invalid_priority_labels_fail_before_drawing(value):
    sample = SimpleNamespace(prompt="q", metadata={"sampling_pool": "p", "priority": value})
    with pytest.raises(ValueError, match="integer in 1..4"):
        pool_provider.PoolPromptProvider(
            [sample],
            pools={"p": {}},
            seed=1,
            shuffle=True,
            identity="fixture",
            sampling={"policy": "exposure", "priority": {"field": "priority"}},
        )


@pytest.mark.parametrize("policy", ["shuffle", "weighted"])
def test_priority_rejects_unsupported_allocation_policies(policy):
    sampling = {"policy": policy, "priority": {"field": "priority"}}
    if policy == "weighted":
        sampling["weights"] = {"p": 1}
    with pytest.raises(ValueError, match="uniform or exposure"):
        pool_sampling.validate(sampling, {"p": {}})


@pytest.mark.parametrize("priority", [True, None, {}, {"field": ""}, {"field": 1}, {"field": "p", "extra": 1}])
def test_priority_modifier_rejects_invalid_configuration(priority):
    with pytest.raises(ValueError, match="metadata field"):
        pool_sampling.validate({"policy": "uniform", "priority": priority}, {"p": {}})


@pytest.mark.parametrize("policy", ["uniform", "exposure"])
def test_modifier_reaches_preparation_and_rejects_replay(tmp_path, policy):
    data = {
        "sources": [{"name": "component", "pool": "component", "prompt_data": "/p.jsonl", "reward_config": "/r.json"}],
        "sampling": {"policy": policy, "priority": {"field": "priority"}},
    }
    run = test_miles_run_spec.spec(
        tmp_path, data=data, records={"enabled": True, "root": "/records"}, **{"async": {"fully_async": True}}
    )
    assert run.compile().core.prompt_provider == "pools"
    assert run.to_dict()["data"]["sampling"] == data["sampling"]
    document = run.to_dict()
    document.setdefault("core", {})["prompt_replay_enabled"] = True
    with pytest.raises(ValueError, match="does not support replay"):
        test_miles_run_spec.RunSpec.from_dict(document).compile()
    with pytest.raises(ValueError, match="does not support replay"):
        provider(policy, replay={"fraction": 0.1})


@pytest.mark.parametrize("value", [None, True, 0, 5, "1"])
def test_preparation_rejects_invalid_priority_before_training(environment, tmp_path, value):
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "input": "What is 6 times 7?",
                "label": "42",
                "metadata": {"priority": value, "verifiers": [{"name": "gsm8k", "target": "42"}]},
            }
        )
        + "\n"
    )
    rewards = tmp_path / "rewards.json"
    rewards.write_text(json.dumps({"gsm8k": {"factory": "open_instruct.miles.rewards.verifiers.GSM8KVerifier"}}))
    data = {
        "sources": [
            {"name": "component", "pool": "component", "prompt_data": str(source), "reward_config": str(rewards)}
        ],
        "sampling": {"policy": "exposure", "priority": {"field": "priority"}},
    }
    with pytest.raises(ValueError, match="source-component row 1: Priority metadata.priority"):
        test_miles_run_data.prepare(environment, data)


@pytest.mark.parametrize("policy", ["uniform", "exposure"])
@pytest.mark.parametrize("priority", [False, True])
def test_priority_receipt_alias_is_only_stamped_when_enabled(policy, priority):
    sample = SimpleNamespace(prompt="q", metadata={"sampling_pool": "p", "priority": 1, "sampling_priority_group": 4})
    sampling = {"policy": policy}
    if priority:
        sampling["priority"] = {"field": "priority"}
    p = pool_provider.PoolPromptProvider(
        [sample], pools={"p": {}}, seed=1, shuffle=True, identity="alias-fixture", sampling=sampling
    )
    drawn = p.next_samples(1)[0]
    receipt = p.pending[drawn.metadata["pool_selection_id"]]["metadata"]
    if priority:
        assert drawn.metadata["sampling_priority_group"] == receipt["sampling_priority_group"] == 1
    else:
        assert drawn.metadata["sampling_priority_group"] == 4
        assert "sampling_priority_group" not in receipt
