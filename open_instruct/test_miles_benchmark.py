"""Inference benchmarks must allocate no trainer and account only measured work."""

import copy
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.execution import launch
from open_instruct.miles.rollout import benchmark_metrics


def document(tmp_path):
    return {
        "schema_version": 1,
        "name": "benchmark",
        "model": {"source": "/weka/models/policy"},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 32}]},
        "output": {"root": str(tmp_path), "export_hf": False},
        "launch": {"auto_resume": False, "weka_mounts": [{"weka": "models", "mount_path": "/weka/models"}]},
        "trainer": {"gpus": 8, "expert_parallel_size": 8},
        "training": {"save_checkpoints": False, "filter_zero_std_groups": False},
        "inference": {"gpus": 1, "placement_mode": "disaggregated"},
        "async": {"fully_async": True},
        "benchmark": {"measure_seconds": 60},
    }


@pytest.mark.parametrize("gpus", [1, 2, 8])
def test_benchmark_allocates_only_inference_and_preserves_trainer_settings(tmp_path, gpus):
    payload = document(tmp_path)
    payload["inference"]["gpus"] = gpus
    spec = RunSpec.from_dict(payload)
    plan = spec.plan()
    assert plan["allocation"]["allocated_gpus"] == gpus
    assert plan["allocation"]["nodes"][0]["trainer_gpus"] == 0
    assert plan["runtime"]["shape"]["trainer_gpus"] == 0
    assert spec.sections["trainer"] == payload["trainer"]
    restored = RunSpec.from_dict(spec.to_dict())
    assert restored.benchmark == spec.benchmark
    task = launch.specification("image", spec)["tasks"][0]
    assert task["resources"]["gpuCount"] == gpus
    assert "preflight_attention" not in task["arguments"][0]


@pytest.mark.parametrize(
    "section,key,value,match",
    [
        ("training", "filter_zero_std_groups", True, "reward_mode=none"),
        ("training", "save_checkpoints", True, "save_checkpoints=false"),
        ("launch", "auto_resume", True, "auto_resume=false"),
        ("inference", "gpus", 16, "at most eight"),
        ("benchmark", "measure_seconds", 0, "measure_seconds"),
        ("benchmark", "measure_seconds", float("nan"), "measure_seconds"),
        ("benchmark", "reward_mode", "fake", "reward_mode"),
    ],
)
def test_benchmark_rejects_misleading_or_unbounded_runs(tmp_path, section, key, value, match):
    payload = document(tmp_path)
    payload[section][key] = value
    with pytest.raises(ValueError, match=match):
        RunSpec.from_dict(payload)


def test_measured_throughput_excludes_warmup_and_includes_consumer_overhead():
    batches = [
        dict(phase="warmup", started=0, finished=100, groups=100, samples=400, response_tokens=9999, route_bytes=1),
        dict(phase="measure", started=100, finished=105, groups=2, samples=8, response_tokens=200, route_bytes=100),
        dict(phase="measure", started=110, finished=115, groups=2, samples=8, response_tokens=100, route_bytes=100),
    ]
    summary = benchmark_metrics.summarize(batches, 2)
    assert summary["seconds"] == 15
    assert summary["response_tokens_per_second"] == 20
    assert summary["response_tokens_per_gpu_second"] == 10
    assert summary["samples"] == 16


def test_response_checks_reject_missing_payload_and_wrong_policy():
    sample = SimpleNamespace(
        tokens=[1, 2, 3],
        response_length=2,
        rollout_log_probs=[-0.1, -1.0],
        status=SimpleNamespace(value="completed"),
        all_weight_version_spans=[SimpleNamespace(version="0")],
        rollout_routed_experts=SimpleNamespace(shape=(2, 16, 16), nbytes=2048),
        index=0,
        group_index=0,
    )
    assert benchmark_metrics.sample_record(sample, require_routes=True)["route_bytes"] == 2048
    for field, value in [
        ("rollout_log_probs", [-0.1]),
        ("rollout_log_probs", [float("nan"), -1]),
        ("all_weight_version_spans", [SimpleNamespace(version="1")]),
        ("rollout_routed_experts", None),
    ]:
        broken = copy.copy(sample)
        setattr(broken, field, value)
        with pytest.raises(ValueError):
            benchmark_metrics.sample_record(broken, require_routes=True)
