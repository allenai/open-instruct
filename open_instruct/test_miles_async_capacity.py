"""CPU checks of fleet sizing, explicit overrides, and bounded discard accounting."""

import pytest

from open_instruct.miles.configuration import async_capacity, validation
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.rollout.queue_metrics import QueueMetrics


def configured(tmp_path, **sections):
    return RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "capacity-test",
            "model": {"source": "model", "format": "hf"},
            "output": {"root": str(tmp_path / "run")},
            "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 16}]},
            "async": {"fully_async": True},
            **sections,
        },
        config_path=tmp_path / "run.toml",
    )


def test_scaling_engines_scales_producer_but_not_completed_buffer(tmp_path):
    small = configured(tmp_path, inference={"gpus": 2}).compile()
    big = configured(tmp_path, inference={"gpus": 32}).compile()
    tp = configured(tmp_path, inference={"gpus": 32, "rollout_tensor_parallel_size": 4}).compile()
    assert small.miles["async_max_concurrent_samples"] == 256
    assert big.miles["async_max_concurrent_samples"] == 4096
    assert tp.miles["async_max_concurrent_samples"] == 1024
    for config in (small, big, tp):
        assert config.plan()["async_capacity"]["completed_buffer_samples"] == 128
        assert "--async-max-concurrent-samples" in config.arguments()


def test_explicit_budget_is_preserved_and_warned(tmp_path):
    config = configured(tmp_path, inference={"gpus": 8}, miles={"async_max_concurrent_samples": 32}).compile()
    report = config.plan()["async_capacity"]
    assert report["producer_sample_budget"] == 32
    assert any("below 512" in warning for warning in report["warnings"])


def test_defaults_respect_http_limit_and_round_up_groups(tmp_path):
    config = configured(
        tmp_path, inference={"gpus": 3, "sglang_server_concurrency": 11, "sglang_max_running_requests": 64}
    ).compile()
    assert config.miles["async_max_concurrent_samples"] == 72
    assert any("global HTTP semaphore" in w for w in config.plan()["async_capacity"]["warnings"])


def test_sync_and_unresolved_raw_settings_are_not_sized():
    assert async_capacity.report({}, 1) == {"enabled": False, "warnings": []}
    assert async_capacity.report({"fully_async": True}, 1)["producer_sample_budget"] is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_producer_limits_fail_early(value):
    with pytest.raises(ValueError, match="async_max_concurrent_samples"):
        validation.runtime_values({"async_max_concurrent_samples": value})


def test_subgroup_buffer_fails_early():
    with pytest.raises(ValueError, match="whole prompt group"):
        validation.runtime_values(
            {"fully_async": True, "rollout_batch_size": 4, "async_data_buffer_capacity_factor": 0.1}
        )


def test_capacity_warning_uses_actual_group_rounding_and_batch_budget():
    report = async_capacity.report(
        {
            "fully_async": True,
            "rollout_batch_size": 4,
            "n_samples_per_prompt": 4,
            "global_batch_size": 8,
            "async_max_concurrent_samples": 7,
            "async_data_buffer_capacity_factor": 4,
        },
        1,
    )
    assert report["producer_sample_budget"] == 4
    assert any("whole groups" in w for w in report["warnings"])
    assert any("expire" in w for w in report["warnings"])


def test_discard_fraction_reports_token_waste_and_length_bias_separately():
    metrics = QueueMetrics()
    metrics.record([4096, 512], age=3, accepted=False)
    metrics.record([100, 200], age=1, accepted=True)
    prefix = "rollout/fully_async/completed_queue/"
    first = metrics.collect()
    assert first[prefix + "dropped_samples"] == 2
    assert first[prefix + "dropped_samples_fraction"] == 0.5
    assert first[prefix + "dropped_response_tokens_fraction"] == pytest.approx(4608 / 4908)
    assert first[prefix + "dropped_samples_by_length/4096_8191"] == 1
    assert first[prefix + "dropped_samples_fraction_by_length/4096_8191"] == 1
    assert first[prefix + "dropped_samples_by_age/3"] == 2
    second = metrics.collect()
    assert second.keys() == first.keys()
    assert all(value == 0 for value in second.values())


def test_producer_headroom_and_partial_groups_are_explained(tmp_path):
    config = configured(tmp_path, inference={"gpus": 32}, miles={"rollout_submission_granularity": "sample"}).compile()
    warnings = config.plan()["async_capacity"]["warnings"]
    assert any("not a predicted age" in w for w in warnings)
    assert any("retained siblings" in w for w in warnings)


def test_combined_group_backlog_warns_even_when_each_limit_looks_small():
    options = dict(
        fully_async=True,
        n_samples_per_prompt=4,
        rollout_batch_size=8,
        global_batch_size=32,
        async_max_concurrent_samples=64,
        async_data_buffer_capacity_factor=1,
        rollout_submission_granularity="group",
    )
    report = async_capacity.report(options, 2)
    assert report["owned_plus_buffer_samples"] == 96
    assert any("retain 3 future" in warning for warning in report["warnings"])
    report = async_capacity.report(options | {"async_max_concurrent_samples": 32}, 2)
    assert report["owned_plus_buffer_samples"] == 64
    assert not any("future optimizer batches" in warning for warning in report["warnings"])
