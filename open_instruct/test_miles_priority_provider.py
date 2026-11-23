"""Priority traversal exhausts groups in order, with deterministic row shuffles."""

import copy
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
        seed=kwargs.pop("seed", 17),
        shuffle=kwargs.pop("shuffle", True),
        **kwargs,
    )


def prompts(p, n):
    return [s.prompt for s in p.next_samples(n)]


def test_exhaust_groups_in_order_without_replacement_across_batch_boundaries():
    p = provider()
    samples = p.next_samples(9) + p.next_samples(124) + p.next_samples(25)
    assert [s.metadata[FIELD] for s in samples] == [1] * 11 + [2] * 120 + [3] * 7 + [4] * 20
    assert len({s.prompt for s in samples}) == 158
    assert p.epoch_id == 0 and p.offset == 158
    assert p.next_samples(1)[0].metadata[FIELD] == 1
    assert p.epoch_id == 1 and p.offset == 1


@pytest.mark.parametrize("shuffle", [False, True])
def test_resume_rollback_and_batch_boundaries(shuffle):
    p = provider(shuffle=shuffle)
    prompts(p, 9)
    state = copy.deepcopy(p.state_dict())
    expected = prompts(p, 145)
    restored = provider(shuffle=shuffle)
    restored.load_state_dict(state)
    assert prompts(restored, 13) + prompts(restored, 132) == expected
    p.load_state_dict(state)
    assert prompts(p, 145) == expected
    assert len(str(state)) < 1000


def test_seed_controls_only_within_group_order_and_no_shuffle_preserves_input_order():
    first, same, other = provider(), provider(), provider(seed=18)
    a, b, c = (p.next_samples(158) for p in (first, same, other))
    assert [s.prompt for s in a] == [s.prompt for s in b]
    assert [s.prompt for s in a] != [s.prompt for s in c]
    assert [s.metadata[FIELD] for s in a] == [s.metadata[FIELD] for s in c]
    d = Dataset()
    d.origin_samples.reverse()
    assert prompts(provider(d, shuffle=False), 158) == [
        s.prompt for g in range(1, 5) for s in d.origin_samples if s.metadata[FIELD] == g
    ]


@pytest.mark.parametrize("value", [None, True, "1", 0, 5, 1.0])
def test_missing_or_invalid_annotations_fail(value):
    data = Dataset()
    data.origin_samples[0].metadata = {} if value is None else {FIELD: value}
    with pytest.raises(ValueError, match="metadata.*must be an integer"):
        provider(data)


def test_empty_groups_are_skipped_and_empty_inventory_fails():
    p = provider(Dataset((0, 4, 0, 5)), shuffle=False)
    assert prompts(p, 9) == [f"2:{i}" for i in range(4)] + [f"4:{i}" for i in range(5)]
    with pytest.raises(ValueError, match="nonempty"):
        provider(Dataset((0, 0, 0, 0)))


@pytest.mark.parametrize("key,value", [("field", "other"), ("seed", 99), ("shuffle", False), ("kind", "priority/v1")])
def test_changed_policy_is_rejected(key, value):
    state = provider().state_dict()
    state[key] = value
    with pytest.raises(ValueError, match="configuration mismatch"):
        provider().load_state_dict(state)


def test_changed_group_membership_and_bad_cursors_are_rejected_atomically():
    p = provider()
    state = copy.deepcopy(p.state_dict())
    for value in (-1, 159, True):
        with pytest.raises(ValueError, match="cursor"):
            p.load_state_dict(state | {"offset": value})
        assert p.state_dict() == state
    data = Dataset()
    # Keep the counts identical while changing assignment order.
    data.origin_samples[0].metadata[FIELD] = 2
    data.origin_samples[11].metadata[FIELD] = 1
    with pytest.raises(ValueError, match="assignments_sha256"):
        provider(data).load_state_dict(state)
    assert prompts(p, 0) == []
    for n in (-1, True, 1.5):
        with pytest.raises(ValueError):
            p.next_samples(n)
    prompts(p, 150)
    assert len(p.next_samples(9)) == 9
    assert p.epoch_id == 1 and p.offset == 1
    for epoch in (-1, True):
        with pytest.raises(ValueError, match="epoch"):
            p.load_state_dict(p.state_dict() | {"epoch_id": epoch})


def test_run_config_compiles_priority_and_preserves_default(tmp_path):
    document = {
        "schema_version": 1,
        "name": "priority-test",
        "model": {"source": "model"},
        "output": {"root": str(tmp_path)},
        "data": {"prompt_data": "prompts.jsonl", "reward_config": "rewards.json"},
        "core": {"prompt_provider": "priority", "prompt_priority_field": FIELD},
    }
    compiled = run_spec.RunSpec.from_dict(document, config_path=tmp_path / "run.toml").compile()
    assert compiled.core.prompt_provider == "priority"
    assert compiled.core.prompt_priority_field == FIELD
    assert config.CoreConfig().prompt_provider == "shuffle"
    with pytest.raises(ValueError, match="requires.*field"):
        config.RunConfig(config.CoreConfig(prompt_provider="priority"), {}).validate()
    with pytest.raises(ValueError, match="require.*priority"):
        config.RunConfig(config.CoreConfig(prompt_priority_field=FIELD), {}).validate()
    with pytest.raises(ValueError, match="rollout_global_dataset"):
        config.RunConfig(compiled.core, {"rollout_global_dataset": False}).validate()
    with pytest.raises(ValueError, match="cannot yet"):
        config.RunConfig(config.CoreConfig(**document["core"], prompt_replay_enabled=True), {}).validate()


def weighted_dataset(sizes=(1000, 1000)):
    data = Dataset((0, 0, 0, 0))
    for group in range(1, 5):
        for source, size in zip(("a", "b"), sizes, strict=True):
            data.origin_samples.extend(
                SimpleNamespace(prompt=f"{group}:{source}:{i}", metadata={FIELD: group, "source_name": source})
                for i in range(size)
            )
    return data


def weighted(data=None, **kwargs):
    return provider(
        weighted_dataset() if data is None else data,
        source_field="source_name",
        source_weights=kwargs.pop("source_weights", {"a": 3, "b": 1}),
        **kwargs,
    )


def test_source_weights_apply_only_within_the_active_group_and_never_repeat():
    p = weighted()
    first = p.next_samples(400)
    assert all(s.metadata[FIELD] == 1 for s in first)
    assert abs(sum(s.metadata["source_name"] == "a" for s in first) / 400 - 0.75) < 0.06
    rest = p.next_samples(7600)
    all_rows = first + rest
    assert len({s.prompt for s in all_rows}) == 8000
    assert [s.metadata[FIELD] for s in all_rows] == sum(([g] * 2000 for g in range(1, 5)), [])
    assert p.epoch_id == 0 and p.offset == 8000


def test_source_depletion_renormalizes_and_missing_groups_are_skipped():
    data = weighted_dataset((1, 10))
    data.origin_samples = [
        s for s in data.origin_samples if not (s.metadata[FIELD] == 2 and s.metadata["source_name"] == "a")
    ]
    p = weighted(data, source_weights={"a": 1000000, "b": 1, "absent": 7})
    samples = p.next_samples(len(data))
    assert samples[0].metadata["source_name"] == "a"
    assert all(s.metadata["source_name"] == "b" for s in samples[1:11])
    assert [s.metadata[FIELD] for s in samples] == [1] * 11 + [2] * 10 + [3] * 11 + [4] * 11
    assert len({s.prompt for s in samples}) == 43


def test_weighted_resume_and_configuration_order_are_deterministic():
    p = weighted()
    prompts(p, 177)
    saved = copy.deepcopy(p.state_dict())
    expected = prompts(p, 2500)
    other = weighted(source_weights={"b": 1, "a": 3})
    other.load_state_dict(saved)
    assert prompts(other, 33) + prompts(other, 2467) == expected
    for changes in ({"source_weights": {"a": 1, "b": 1}}, {"source_field": "other"}):
        with pytest.raises(ValueError, match="configuration mismatch"):
            other.load_state_dict(saved | changes)


@pytest.mark.parametrize("weights", [{}, {"a": True, "b": 1}, {"a": 0, "b": 1}, {"a": float("inf"), "b": 1}, {"a": 1}])
def test_invalid_source_weights_and_unconfigured_rows_fail(weights):
    with pytest.raises(ValueError):
        weighted(source_weights=weights)


def test_weighted_config_validates_pairs_and_serializes(tmp_path):
    core = {
        "prompt_provider": "priority",
        "prompt_priority_field": FIELD,
        "prompt_priority_source_field": "source_name",
        "prompt_priority_source_weights": {"a": 3, "b": 1},
    }
    document = {
        "schema_version": 1,
        "name": "weighted-priority",
        "model": {"source": "model"},
        "output": {"root": str(tmp_path)},
        "data": {"prompt_data": "rows.jsonl", "reward_config": "rewards.json"},
        "core": core,
    }
    compiled = run_spec.RunSpec.from_dict(document, config_path=tmp_path / "run.toml").compile()
    assert compiled.core.prompt_priority_source_weights == {"a": 3, "b": 1}
    for field in ("prompt_priority_source_field", "prompt_priority_source_weights"):
        with pytest.raises(ValueError, match="requires both"):
            config.RunConfig(config.CoreConfig(**(core | {field: None})), {}).validate()
    with pytest.raises(ValueError, match="require.*priority"):
        config.RunConfig(config.CoreConfig(prompt_priority_source_weights={"a": 1}), {}).validate()


def test_workbench_canonical_source_path_selects_sources_inside_one_artifact():
    data = weighted_dataset((1000, 1000))
    for sample in data.origin_samples:
        source = sample.metadata.pop("source_name")
        sample.metadata["workbench"] = {"source": {"candidate_id": source}}
    p = PriorityPromptProvider(
        data,
        field=FIELD,
        seed=17,
        shuffle=True,
        source_field="workbench.source.candidate_id",
        source_weights={"a": 3, "b": 1},
    )
    first = p.next_samples(400)
    assert all(s.metadata[FIELD] == 1 for s in first)
    assert abs(sum(s.metadata["workbench"]["source"]["candidate_id"] == "a" for s in first) / 400 - 0.75) < 0.06


@pytest.mark.parametrize("use_weights", [False, True])
def test_multiple_epochs_change_shuffle_reproducibly_and_resume_across_boundaries(use_weights):
    make = weighted if use_weights else provider
    p = make()
    size = len(p.samples)
    first = prompts(p, size)
    second = prompts(p, size)
    assert set(first) == set(second) and len(set(first)) == size
    assert first != second
    other = make()
    assert prompts(other, size * 2) == first + second
    saved = copy.deepcopy(p.state_dict())
    expected = prompts(p, size + 5)
    resumed = make()
    resumed.load_state_dict(saved)
    assert prompts(resumed, 4) + prompts(resumed, size + 1) == expected
