"""Fail closed when causal controls lose cohort membership or policy age."""

import asyncio
import hashlib
import json
import random
from types import SimpleNamespace

import pytest
from miles.ray.placement_group import update_weights
from scripts.miles import opd_control, opd_control_runtime


def test_selection_replay_freezes_membership_not_completion_order():
    events = [{"event": "submitted", "prompt_sha256": x} for x in "abcdef"]
    events += [{"event": "batch_delivered", "prompt_sha256": x, "rollout_id": i // 2} for i, x in enumerate("dbfe")]
    assert opd_control.schedules(events, 2, 2) == {"admitted": list("abcd"), "selected": list("bdef")}
    with pytest.raises(ValueError, match="complete"):
        opd_control.schedules(events[:-1], 2, 2)


def test_policy_age_changes_without_changing_training_membership():
    for period in (1, 4):
        published = 0
        ages = []
        assert opd_control.publication_due(None, period, 8)
        for step in range(8):
            ages.append(step - published)
            assert ages[-1] == opd_control.expected_age(step, period)
            if opd_control.publication_due(step, period, 8):
                published = step + 1
        assert ages == ([0] * 8 if period == 1 else [0, 1, 2, 3] * 2)
        assert published == 8  # Final evaluation must see the trained checkpoint.


def test_exact_row_resolution_preserves_duplicate_prompt_metadata():
    rows = [dict(input="same", label="a", metadata={"row": 0}), dict(input="same", label="b", metadata={"row": 1})]
    order = [0, 1]
    random.Random(42).shuffle(order)
    events = [
        dict(event="submitted", group_index=i, prompt_sha256=opd_control.prompt_hash(rows[index]["input"]))
        for i, index in enumerate(order)
    ]
    events.append(dict(event="batch_delivered", group_index=1, prompt_sha256=events[1]["prompt_sha256"], rollout_id=0))
    cohorts = opd_control.resolve_cohorts(rows, events, updates=1, groups_per_update=1)
    assert cohorts["admitted"]["rows"] == [rows[order[0]]]
    assert cohorts["selected"]["rows"] == [rows[order[1]]]
    rows[order[0]]["input"] = "changed"
    with pytest.raises(ValueError, match="row order"):
        opd_control.resolve_cohorts(rows, events, updates=1, groups_per_update=1)


def test_runtime_only_publishes_assigned_steps_and_awaits_success(monkeypatch, tmp_path):
    monkeypatch.setenv("OI_OPD_OUTPUT", str(tmp_path))
    source = tmp_path / "cohort.jsonl"
    source.write_text('{"input":"Q"}\n' * 4)
    config = tmp_path / "control.json"
    config.write_text(
        json.dumps(
            dict(
                cohort=str(source), cohort_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), publication_period=4
            )
        )
    )
    calls = []

    async def native(*, rollout_id=None):
        calls.append(rollout_id)
        return len(calls)

    async def create(*values):
        return SimpleNamespace(update_weights=native, sentinel="unchanged"), None

    namespace = {"create_training_models": create}
    exec("def train(): pass", namespace)
    args = SimpleNamespace(
        fully_async=False,
        start_rollout_id=0,
        dynamic_sampling_filter_path=None,
        rollout_sample_filter_path=None,
        partial_rollout=False,
        global_batch_size=2,
        rollout_batch_size=1,
        n_samples_per_prompt=2,
        num_rollout=4,
    )
    opd_control_runtime.install(args, namespace["train"], config)
    actor, critic = asyncio.run(namespace["create_training_models"]())
    assert actor.sentinel == "unchanged" and critic is None
    published = []

    async def publish_version(version):
        published.append(version)

    executor = SimpleNamespace(set_weight_version=SimpleNamespace(remote=publish_version))
    for index in [None, 0, 1, 2, 3]:
        asyncio.run(update_weights(actor, executor, rollout_id=index))
    assert published == [1, 2]
    assert calls == [None, 3]
    assert not args.rollout_shuffle
    assert args.eval_interval is None
    assert args.prompt_data == str(source)
    rows = [json.loads(line) for line in (tmp_path / "controlled-exposure.jsonl").read_text().splitlines()]
    assert [r["served_optimizer_step"] for r in rows[1:]] == [0, 0, 0, 0, 4]


def test_fixed_cohort_refuses_replacements_and_membership_changes(monkeypatch, tmp_path):
    monkeypatch.setenv("OI_OPD_OUTPUT", str(tmp_path))
    opd_control_runtime.record("publication_completed", served_optimizer_step=0, publication_period=4)
    group = [SimpleNamespace(prompt="Q", response_length=2, status=SimpleNamespace(value="completed"))]
    source = SimpleNamespace(get_samples=lambda count: [group])
    args = SimpleNamespace(
        rollout_batch_size=1, n_samples_per_prompt=1, oi_opd_prompt_hashes=[opd_control.prompt_hash("Q")]
    )

    def native(args, step, data):
        groups = data.get_samples(1)
        data.add_samples([])
        return SimpleNamespace(samples=groups)

    monkeypatch.setattr(opd_control_runtime.sglang_rollout, "generate_rollout", native)
    assert opd_control_runtime.generate(args, 0, source).samples == [group]

    def replacement(args, step, data):
        data.get_samples(1)
        return data.get_samples(1)

    monkeypatch.setattr(opd_control_runtime.sglang_rollout, "generate_rollout", replacement)
    with pytest.raises(ValueError, match="replace or oversample"):
        opd_control_runtime.generate(args, 0, source)
    args.oi_opd_prompt_hashes = [opd_control.prompt_hash("different")]
    with pytest.raises(ValueError, match="substituted"):
        opd_control_runtime.generate(args, 0, source)
