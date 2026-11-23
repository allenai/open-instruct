"""Failed grades omit whole training groups without changing valid zero rewards."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import rewards


def sample(response="ok", group_index=0):
    return SimpleNamespace(
        group_index=group_index,
        tokens=[],
        response_length=0,
        response=response,
        prompt="Question",
        metadata={"verifiers": [{"name": "test", "target": "answer"}]},
    )


@pytest.fixture
def scoring(monkeypatch, tmp_path):
    async def verify(tokens, response, target, **kwargs):
        if response == "failure":
            raise RuntimeError("service unavailable")
        if response == "cancel":
            raise asyncio.CancelledError()
        return SimpleNamespace(
            score=float("nan") if response == "nan" else float(response == "ok"),
            cost=0.0,
            diagnostics={"status": response} if response in {"timeout", "service_error"} else None,
        )

    monkeypatch.setattr(rewards, "_registry", lambda path: {"test": SimpleNamespace(async_call=verify)})
    return SimpleNamespace(olmo_core=CoreConfig(reward_config="unused"), save=str(tmp_path))


def test_training_batch_preserves_missing_reward_and_diagnostics(scoring):
    samples = [sample(value) for value in ("ok", "failure", "wrong", "timeout", "nan")]
    assert asyncio.run(rewards.rollout_reward(scoring, samples)) == [1.0, None, 0.0, 0.0, None]
    assert samples[1].metadata["reward_omission"] == {"verifier": "test", "error": "RuntimeError: service unavailable"}
    assert "reward_components" not in samples[1].metadata
    assert "reward_omission" not in samples[2].metadata
    captures = list((Path(scoring.save) / "verifier_failures").glob("*.json"))
    assert any(json.loads(path.read_text())["response"] == "failure" for path in captures)


def test_single_training_failure_is_missing_not_zero(scoring):
    assert asyncio.run(rewards.rollout_reward(scoring, sample("failure"))) is None


@pytest.mark.parametrize("hook,group_index", [(rewards.registered_reward, 0), (rewards.rollout_reward, None)])
def test_offline_and_evaluation_scoring_stay_strict(scoring, hook, group_index):
    with pytest.raises(RuntimeError, match="service unavailable"):
        asyncio.run(hook(scoring, sample("failure", group_index)))


def test_cancellation_is_not_an_omission(scoring):
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(rewards.rollout_reward(scoring, sample("cancel")))


def test_registry_and_routing_errors_stay_fatal(scoring, monkeypatch):
    bad = sample()
    bad.metadata["verifiers"][0]["name"] = "unknown"
    with pytest.raises(ValueError, match="trusted reward registry"):
        asyncio.run(rewards.rollout_reward(scoring, bad))

    def broken_registry(path):
        raise ValueError("invalid factory configuration")

    monkeypatch.setattr(rewards, "_registry", broken_registry)
    with pytest.raises(ValueError, match="invalid factory configuration"):
        asyncio.run(rewards.rollout_reward(scoring, sample()))


@pytest.mark.parametrize("raises", [False, True])
def test_judge_failure_is_also_an_omission(scoring, monkeypatch, raises):
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: True)

    async def judge(args, sample, **kwargs):
        if raises:
            raise RuntimeError("judge unavailable")
        sample.metadata["verifier_diagnostics"] = {"test": {"status": "judge_error", "attempts": [1, 2, 3]}}
        return 0.0

    monkeypatch.setattr(rewards.general_judge, "general_judge_score", judge)
    scored = sample()
    assert asyncio.run(rewards.rollout_reward(scoring, scored)) is None
    assert scored.metadata["reward_omission"]["verifier"] == "test"
    if not raises:
        assert scored.metadata["verifier_diagnostics"]["test"]["attempts"] == [1, 2, 3]


@pytest.mark.parametrize("variance_filter", [False, True])
def test_native_collector_discards_failed_group_and_counts_record(scoring, variance_filter):
    filters = pytest.importorskip("miles.rollout.filter_hub.common_filters")
    types = pytest.importorskip("miles.utils.types")
    base = pytest.importorskip("miles.rollout.filter_hub.base_types")
    metrics_module = pytest.importorskip("open_instruct.miles.rollout.rollout_metrics")
    args = SimpleNamespace(reward_key=None)
    dropped = []
    kept = []
    metrics = base.MetricGatherer()
    dynamic_filter = filters.apply_reward_nonzero_std_filter if variance_filter else None
    # Two failures in a group still omit one source record. Healthy siblings
    # must be dropped too; then the collector can fill from the next group.
    for index, responses in enumerate((("failure", "failure", "ok"), ("wrong", "ok", "wrong"))):
        group = [types.Sample(**vars(sample(response, index))) for response in responses]
        for item, reward in zip(group, asyncio.run(rewards.rollout_reward(scoring, group)), strict=True):
            item.status = types.Sample.Status.COMPLETED
            item.reward = reward
        result = filters.apply_preput_filters(args, dynamic_filter, group)
        if result.keep:
            kept.extend(group)
        else:
            dropped.extend(group)
            metrics.on_dynamic_filter_drop(result.reason)
    assert len(dropped) == len(kept) == 3
    assert [item.reward for item in kept] == [0.0, 1.0, 0.0]
    extra = metrics.collect()
    assert metrics_module.log_rollout_data(0, SimpleNamespace(save=None), kept, extra, 1.0) is False
    assert extra["rollout/verification/omitted_records"] == 1
    # No process-local reward counter: the omitted count survives transport
    # from a separate producer, and absent omissions explicitly log zero.
    fresh = {}
    metrics_module.log_rollout_data(1, SimpleNamespace(save=None), kept, fresh, 1.0)
    assert fresh["rollout/verification/omitted_records"] == 0


def test_omission_metric_reaches_tracking_without_extra_metrics(monkeypatch):
    metrics_module = pytest.importorskip("open_instruct.miles.rollout.rollout_metrics")
    logged = []
    monkeypatch.setattr(metrics_module, "compute_rollout_step", lambda args, rollout_id: rollout_id)
    monkeypatch.setattr(metrics_module.tracking, "log", lambda args, values, **kwargs: logged.append(values))
    metrics_module.log_rollout_data(3, SimpleNamespace(save=None), [], None, 1.0)
    assert logged == [{"rollout/verification/omitted_records": 0, "rollout/step": 3}]


def test_returned_service_failure_is_omitted_and_recovery_clears_marker(scoring):
    scored = sample("service_error")
    assert asyncio.run(rewards.rollout_reward(scoring, scored)) is None
    assert scored.metadata["verifier_diagnostics"]["test"]["status"] == "service_error"
    scored.response = "ok"
    assert asyncio.run(rewards.rollout_reward(scoring, scored)) == 1.0
    assert "reward_omission" not in scored.metadata


@pytest.mark.parametrize("variance_filter", [False, True])
def test_async_buffer_waits_for_replacement_and_resets_omission_count(scoring, variance_filter):
    native = pytest.importorskip("miles.rollout.fully_async_data_buffer")
    types = pytest.importorskip("miles.utils.types")
    metrics_module = pytest.importorskip("open_instruct.miles.rollout.rollout_metrics")
    args = SimpleNamespace(
        reward_key=None,
        rollout_batch_size=1,
        async_data_buffer_capacity_factor=1,
        max_weight_staleness=None,
        dynamic_sampling_filter_path=(
            "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std" if variance_filter else None
        ),
    )

    async def scenario():
        retries = []
        buffer = native.DefaultDataBuffer(native.DataBufferConstructorInput(args, retries.append))
        waiting = asyncio.create_task(buffer.get())
        groups = []
        for index, responses in enumerate((("ok", "failure"), ("wrong", "ok"))):
            group = [types.Sample(**vars(sample(response, index))) for response in responses]
            for item, reward in zip(group, await rewards.rollout_reward(scoring, group), strict=True):
                item.status = types.Sample.Status.COMPLETED
                item.reward = reward
            groups.append(group)
            await buffer.put(native.DataBufferInput(prompt_group=group, group=group))
            await asyncio.sleep(0)
            if index == 0:
                assert not waiting.done()
                assert not buffer._buffer
        result = await asyncio.wait_for(waiting, timeout=1)
        assert result.group == groups[1]
        assert retries == []  # a bad prompt is discarded, not retried forever
        extra = buffer.get_metrics()
        metrics_module.log_rollout_data(0, SimpleNamespace(save=None), result.group, extra, 1.0)
        assert extra["rollout/verification/omitted_records"] == 1
        assert "rollout/dynamic_filter/drop_group_has_missing_reward" not in buffer.get_metrics()

    asyncio.run(scenario())
