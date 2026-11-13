"""Static priorities affect group exposure while preserving traversal and resume."""

import copy
from collections import Counter
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import config, run_spec
from open_instruct.miles.rollout.priority_provider import PriorityPromptProvider

FIELD = "reward_priority_olmo35_sft_hero"


class Dataset:
    def __init__(self, sizes=(11, 120, 7, 20)):
        self.origin_samples = [
            SimpleNamespace(prompt=f"{g}:{i}", metadata={FIELD: g})
            for g, size in enumerate(sizes, 1)
            for i in range(size)
        ]

    def __len__(self):
        return len(self.origin_samples)


def provider(data=None, **kwargs):
    return PriorityPromptProvider(
        data if data is not None else Dataset(),
        field=FIELD,
        weights=kwargs.pop("weights", [0.5, 0.35, 0.1, 0.05]),
        seed=kwargs.pop("seed", 17),
        shuffle=kwargs.pop("shuffle", True),
        **kwargs,
    )


def prompts(p, n):
    return [s.prompt for s in p.next_samples(n)]


def test_probabilities_apply_to_groups_independent_of_size_and_cycle_without_replacement():
    p = provider()
    samples = p.next_samples(20000)
    counts = Counter(s.metadata[FIELD] for s in samples)
    for group, expected in enumerate(p.weights, 1):
        assert abs(counts[group] / len(samples) - expected) < 0.015
    for group, size in enumerate(p.counts, 1):
        seen = [s.prompt for s in samples if s.metadata[FIELD] == group]
        for offset in range(0, len(seen) - size, size):
            assert len(set(seen[offset : offset + size])) == size


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_rollback_and_batch_boundaries(shuffle):
    p = provider(shuffle=shuffle)
    prompts(p, 173)
    state = copy.deepcopy(p.state_dict())
    expected = prompts(p, 200)
    restored = provider(shuffle=shuffle)
    restored.load_state_dict(state)
    assert prompts(restored, 13) + prompts(restored, 187) == expected
    p.load_state_dict(state)
    assert prompts(p, 200) == expected
    assert len(str(state)) < 10000  # Cursors and RNG, not row permutations.


@pytest.mark.parametrize("value", [None, True, "1", 0, 5, 1.0])
def test_missing_or_invalid_annotations_fail(value):
    data = Dataset()
    data.origin_samples[0].metadata = {} if value is None else {FIELD: value}
    with pytest.raises(ValueError, match="metadata.*must be an integer"):
        provider(data)


def test_empty_groups_are_explicit_and_zero_weight_rows_are_never_drawn():
    with pytest.raises(ValueError, match="group 3.*no rows"):
        provider(Dataset((3, 4, 0, 5)))
    p = provider(Dataset((3, 4, 0, 5)), weights=[1, 0, 0, 0], shuffle=False)
    assert prompts(p, 7) == ["1:0", "1:1", "1:2", "1:0", "1:1", "1:2", "1:0"]
    with pytest.raises(ValueError, match="nonempty"):
        provider(Dataset((0, 0, 0, 0)))


@pytest.mark.parametrize(
    "key,value", [("field", "other"), ("weights", [1, 0, 0, 0]), ("seed", 99), ("shuffle", False)]
)
def test_changed_policy_is_rejected(key, value):
    state = provider().state_dict()
    state[key] = value
    with pytest.raises(ValueError, match="configuration mismatch"):
        provider().load_state_dict(state)


def test_changed_group_membership_and_bad_cursors_are_rejected_atomically():
    p = provider()
    state = copy.deepcopy(p.state_dict())
    bad = copy.deepcopy(state)
    bad["offsets"][2] = -1
    with pytest.raises(ValueError, match="cursor"):
        p.load_state_dict(bad)
    assert p.state_dict() == state
    data = Dataset()
    data.origin_samples[0].metadata[FIELD] = 2
    with pytest.raises(ValueError, match="mismatch"):
        provider(data).load_state_dict(state)
    assert prompts(p, 0) == []
    for n in (-1, True, 1.5):
        with pytest.raises(ValueError):
            p.next_samples(n)


@pytest.mark.parametrize("weights", [[1], [0, 0, 0, 0], [True, 0, 0, 0], [float("nan"), 0, 0, 0], [-1, 1, 1, 0]])
def test_bad_weights_fail_in_planning_and_provider(weights):
    with pytest.raises(ValueError):
        config.CoreConfig(prompt_provider="priority", prompt_priority_field=FIELD, prompt_priority_weights=weights)
    with pytest.raises(ValueError):
        provider(weights=weights)


def test_run_config_compiles_priority_and_preserves_default(tmp_path):
    document = {
        "schema_version": 1,
        "name": "priority-test",
        "model": {"source": "model"},
        "output": {"root": str(tmp_path)},
        "data": {"prompt_data": "prompts.jsonl", "reward_config": "rewards.json"},
        "core": {
            "prompt_provider": "priority",
            "prompt_priority_field": FIELD,
            "prompt_priority_weights": [0.5, 0.35, 0.1, 0.05],
        },
    }
    compiled = run_spec.RunSpec.from_dict(document, config_path=tmp_path / "run.toml").compile()
    assert compiled.core.prompt_provider == "priority"
    assert compiled.core.prompt_priority_field == FIELD
    assert config.CoreConfig().prompt_provider == "shuffle"
    for change in ({"prompt_priority_field": None}, {"prompt_priority_weights": None}):
        with pytest.raises(ValueError):
            config.RunConfig(config.CoreConfig(**(document["core"] | change)), {}).validate()
    with pytest.raises(ValueError, match="require.*priority"):
        config.RunConfig(config.CoreConfig(prompt_priority_field=FIELD), {}).validate()
    with pytest.raises(ValueError, match="rollout_global_dataset"):
        config.RunConfig(compiled.core, {"rollout_global_dataset": False}).validate()
    with pytest.raises(ValueError, match="cannot yet"):
        config.RunConfig(config.CoreConfig(**document["core"], prompt_replay_enabled=True), {}).validate()
