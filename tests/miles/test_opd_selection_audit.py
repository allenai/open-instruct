"""CPU probes of the real pinned queue: completion order, abortion and policy age."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

from open_instruct.miles.distillation import async_buffer, opd_selection_audit


def test_preflight_does_not_inherit_production_capture_paths(tmp_path, monkeypatch):
    trace = tmp_path / "production.jsonl"
    captures = tmp_path / "production-evals"
    environment = {
        "OI_OPD_SELECTION_AUDIT": str(trace),
        "OI_OPD_EVAL_CAPTURE": str(captures),
        "PYTHONPATH": "/src/Megatron-LM",
        "OI_OPD_REWARD_CONFIG": "/run/rewards.json",
    }
    preflight = opd_selection_audit.preflight_environment(environment)
    assert preflight == {"PYTHONPATH": "/src/Megatron-LM", "OI_OPD_REWARD_CONFIG": "/run/rewards.json"}
    assert environment["OI_OPD_SELECTION_AUDIT"] == str(trace)
    assert environment["OI_OPD_EVAL_CAPTURE"] == str(captures)
    for key in ("OI_OPD_SELECTION_AUDIT", "OI_OPD_EVAL_CAPTURE"):
        monkeypatch.delenv(key, raising=False)
    # Startup tests use synthetic samples and reward-only mock evaluations.
    sample = Sample(index=0, group_index=0, prompt="synthetic")
    opd_selection_audit.begin([sample])
    opd_selection_audit.capture_eval({"math": {"rewards": [1]}}, 0)
    assert not trace.exists() and not captures.exists()
    # The learner still inherits capture paths after preflight finishes.
    for key in ("OI_OPD_SELECTION_AUDIT", "OI_OPD_EVAL_CAPTURE"):
        monkeypatch.setenv(key, environment[key])
    opd_selection_audit.begin([sample])
    assert json.loads(trace.read_text())["event"] == "submitted"


def entry(index, lengths, *, version=1, aborted=False):
    samples = [
        Sample(
            index=2 * index + i,
            group_index=index,
            prompt=f"private prompt {index}",
            response_length=length,
            weight_versions=[WeightVersionsPerCall([WeightVersionSpan(str(version), 0, length)])],
            status=Sample.Status.ABORTED if aborted and i == 1 else Sample.Status.COMPLETED,
            reward=0.0,
        )
        for i, length in enumerate(lengths)
    ]
    opd_selection_audit.begin(samples)
    return DataBufferInput(prompt_group=samples, group=samples)


def buffer(unused):
    args = SimpleNamespace(
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        global_batch_size=8,
        max_weight_staleness=3,
        async_data_buffer_capacity_factor=2,
        dynamic_sampling_filter_path=None,
        reward_key=None,
    )
    return async_buffer.MeasuredDataBuffer(DataBufferConstructorInput(args, unused))


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_aborted_sibling_removes_completed_sibling_without_stale_drops(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("OI_OPD_SELECTION_AUDIT", str(path))
    rejected = []

    def unused(group):
        rejected.append(group[0].group_index)
        for sample in group:
            sample.reset_for_retry()

    async def exercise():
        queue = buffer(unused)
        # The completed sibling's work is lost too; partial length 100 is not
        # the unknown length the aborted sibling would have reached.
        await queue.put(entry(0, [16000, 100], aborted=True))
        fresh = entry(1, [64, 80])
        await queue.put(fresh)
        assert await asyncio.wait_for(queue.get(current_version=1), 1) is fresh
        metrics = queue.get_metrics()
        assert metrics["rollout/fully_async/aborted_groups_filtered"] == 1
        assert metrics["rollout/fully_async/stale_groups_filtered"] == 0
        assert metrics["rollout/fully_async/completed_queue/dropped_groups"] == 0
        assert rejected == [0]

    asyncio.run(exercise())
    event = next(row for row in records(path) if row["event"] == "aborted_group_rejected")
    assert event["response_lengths"] == [16000, 100]
    assert event["statuses"] == ["completed", "aborted"]
    assert "private prompt" not in path.read_text()


@pytest.mark.parametrize("cancel_slow", [False, True])
def test_completion_order_changes_first_batch_even_without_staleness(cancel_slow):
    async def exercise():
        rejected = []
        queue = buffer(lambda group: rejected.append(group[0].group_index))
        # Submitted slow then fast; finish fast first. No wall-clock sleeps or
        # simulated throughput claims: this controls only completion order.
        slow, fast = entry(0, [12000, 16000], aborted=cancel_slow), entry(1, [64, 80])
        await queue.put(fast)
        await queue.put(slow)
        first = await asyncio.wait_for(queue.get(current_version=1), 1)
        assert first.prompt_group[0].group_index == 1
        if cancel_slow:
            assert rejected == [0]
        else:
            second = await asyncio.wait_for(queue.get(current_version=1), 1)
            assert second is slow
            assert not rejected
        assert queue.get_metrics()["rollout/fully_async/stale_groups_filtered"] == 0

    asyncio.run(exercise())


def test_staleness_boundary_does_not_depend_on_length(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("OI_OPD_SELECTION_AUDIT", str(path))

    async def exercise():
        rejected = []
        queue = buffer(lambda group: rejected.append(group[0].group_index))
        await queue.put(entry(0, [64, 80], version=1))  # Age 4: reject even if short.
        long_fresh = entry(1, [16384, 16384], version=2)  # Age 3: accept at boundary.
        await queue.put(long_fresh)
        assert await asyncio.wait_for(queue.get(current_version=5), 1) is long_fresh
        assert rejected == [0]

    asyncio.run(exercise())
    decisions = [row for row in records(path) if row["event"] != "submitted"]
    assert [(row["event"], row["policy_age"]) for row in decisions] == [
        ("stale_group_rejected", 4),
        ("queue_selected", 3),
    ]


def test_attempt_identity_survives_decisions_but_changes_on_retry(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("OI_OPD_SELECTION_AUDIT", str(path))
    item = entry(0, [5, 6])
    opd_selection_audit.record("generation_returned", item.group)
    opd_selection_audit.begin(item.prompt_group)
    rows = records(path)
    assert rows[0]["attempt_id"] == rows[1]["attempt_id"] != rows[2]["attempt_id"]
    assert len({row["prompt_sha256"] for row in rows}) == 1


def test_disabled_audit_does_not_mutate_samples(monkeypatch):
    monkeypatch.delenv("OI_OPD_SELECTION_AUDIT", raising=False)
    item = entry(0, [5, 6])
    opd_selection_audit.record("generation_returned", item.group)
    assert all(sample.metadata == {} for sample in item.group)


@pytest.mark.parametrize("outcome", ["returned", "cancelled", "failed"])
def test_generation_observer_preserves_outcome(tmp_path, monkeypatch, outcome):
    path = tmp_path / "events.jsonl"
    item = entry(0, [5, 6])
    monkeypatch.setenv("OI_OPD_SELECTION_AUDIT", str(path))

    async def generate(group):
        assert group is item.prompt_group
        if outcome == "cancelled":
            raise asyncio.CancelledError
        if outcome == "failed":
            raise RuntimeError("failure")
        return item

    async def exercise():
        if outcome == "returned":
            assert await opd_selection_audit.generate(item.prompt_group, generate) is item
        else:
            exception = asyncio.CancelledError if outcome == "cancelled" else RuntimeError
            with pytest.raises(exception):
                await opd_selection_audit.generate(item.prompt_group, generate)

    asyncio.run(exercise())
    rows = records(path)
    assert [row["event"] for row in rows] == ["submitted", f"generation_{outcome}"]
    assert rows[0]["attempt_id"] == rows[1]["attempt_id"]


def test_eval_capture_retains_alignment_and_does_not_replace_good_file_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OI_OPD_EVAL_CAPTURE", str(tmp_path))
    item = entry(0, [5, 6])
    data = {"math": {"samples": item.group, "rewards": [1, 0]}}
    opd_selection_audit.capture_eval(data, 7)
    path = tmp_path / "eval-7.jsonl"
    original = path.read_bytes()
    assert [row["reward"] for row in records(path)] == [1, 0]
    data["math"]["rewards"] = [1]
    with pytest.raises(ValueError):
        opd_selection_audit.capture_eval(data, 7)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
