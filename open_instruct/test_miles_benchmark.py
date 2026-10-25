"""Inference benchmarks must allocate no trainer and account only measured work."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.miles import analyze_inference_benchmark, compare_inference_replay, prepare_inference_workload

from open_instruct.miles.configuration import throughput
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
        all_weight_version_spans=[SimpleNamespace(version="0", abs_start=1, abs_end=3)],
        rollout_routed_experts=SimpleNamespace(shape=(2, 16, 16), nbytes=2048),
        index=0,
        group_index=0,
    )
    assert benchmark_metrics.sample_record(sample, require_routes=True)["route_bytes"] == 2048
    for field, value in [
        ("rollout_log_probs", [-0.1]),
        ("rollout_log_probs", [float("nan"), -1]),
        ("all_weight_version_spans", [SimpleNamespace(version="1")]),
        ("all_weight_version_spans", [SimpleNamespace(version="0", abs_start=2, abs_end=3)]),
        ("all_weight_version_spans", [SimpleNamespace(version="0", abs_start=1, abs_end=2)]),
        ("rollout_routed_experts", None),
    ]:
        broken = copy.copy(sample)
        setattr(broken, field, value)
        with pytest.raises(ValueError):
            benchmark_metrics.sample_record(broken, require_routes=True)


def test_panel_resolves_overrides_and_isolates_case_outputs(tmp_path):
    payload = document(tmp_path)
    payload["benchmark"]["variants"] = [
        {"name": "control", "overrides": []},
        {"name": "larger", "overrides": ["inference.sglang_max_running_requests=96"]},
    ]
    spec = RunSpec.from_dict(payload)
    control, larger = spec.benchmark_variants()
    assert control.output["root"] == str(tmp_path / "variants/control")
    assert larger.conversion["hf_output"].startswith(str(tmp_path / "variants/larger"))
    assert control.compile().miles["sglang_max_running_requests"] == 64
    assert larger.compile().miles["sglang_max_running_requests"] == 96
    assert control.benchmark["variants"] == larger.benchmark["variants"] == []
    assert control.plan()["allocation"] == larger.plan()["allocation"] == spec.plan()["allocation"]


@pytest.mark.parametrize(
    "override",
    [
        'model.source="other"',
        "inference.gpus=8",
        "trainer.gpus=1",
        'miles.sglang_dtype="float16"',
        "miles.sglang_tp_size=2",
        "miles.sglang_enable_return_routed_experts=false",
    ],
)
def test_panel_cannot_change_model_workload_or_gpu_allocation(tmp_path, override):
    payload = document(tmp_path)
    payload["benchmark"]["variants"] = [{"name": "bad", "overrides": [override]}]
    with pytest.raises(ValueError, match="serving/queue controls only"):
        RunSpec.from_dict(payload)


def test_miles_router_warns_about_ineffective_policy_and_connection_gate():
    report = throughput.report(
        {
            "use_miles_router": True,
            "sglang_router_policy": "round_robin",
            "rollout_num_gpus": 2,
            "sglang_server_concurrency": 128,
            "miles_router_max_connections": 64,
        },
        SimpleNamespace(),
    )
    assert {warning["code"] for warning in report["warnings"]} == {
        "miles_router_ignores_native_policy",
        "miles_router_connection_limit",
    }


def test_inference_analysis_validates_accounting_and_keeps_missing_occupancy_unknown(tmp_path):
    batches = [
        {
            "phase": "measure",
            "started": 1,
            "finished": 11,
            "finished_unix": 100,
            "groups": 1,
            "samples": 1,
            "response_tokens": 50,
            "route_bytes": 200,
        }
    ]
    files = {
        "workflow.json": {"status": "complete"},
        "benchmark-summary.json": benchmark_metrics.summarize(batches, 2),
        "resolved-plan.json": {"miles": {"rollout_num_gpus": 2}},
        "benchmark-boundaries.json": {},
        "benchmark-shutdown.json": {},
        "benchmark-server-start.json": [],
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    streams = {
        "benchmark-batches.jsonl": batches,
        "benchmark-responses.jsonl": [
            {"phase": "measure", "prompt_tokens": 5, "response_tokens": 50, "route_bytes": 200, "status": "truncated"}
        ],
        "benchmark-resources.jsonl": [],
    }
    for name, values in streams.items():
        (tmp_path / name).write_text("".join(json.dumps(value) + "\n" for value in values))
    report = analyze_inference_benchmark.analyze(tmp_path)
    assert report["summary"]["response_tokens_per_gpu_second"] == 2.5
    assert report["truncated_fraction"] == 1
    assert report["occupancy"]["pipeline"]["http_active_requests"]["mean"] is None
    files["benchmark-summary.json"]["response_tokens"] = 100
    (tmp_path / "benchmark-summary.json").write_text(json.dumps(files["benchmark-summary.json"]))
    with pytest.raises(ValueError, match="summary disagrees"):
        analyze_inference_benchmark.analyze(tmp_path)


def test_replay_requires_explicit_diagnostic_server_setting(tmp_path):
    payload = document(tmp_path)
    payload["benchmark"]["replay_reference"] = str(tmp_path / "reference")
    with pytest.raises(ValueError, match="custom_logit_processor"):
        RunSpec.from_dict(payload)
    payload.setdefault("miles", {})["sglang_enable_custom_logit_processor"] = True
    assert RunSpec.from_dict(payload).benchmark["replay_repeats"] == 2


def test_workload_preparation_filters_without_truncating_and_keeps_sources(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer_config.json").write_text("{}")
    tokenizer = SimpleNamespace(chat_template="template", encode=lambda text, **kwargs: list(text))
    monkeypatch.setattr(prepare_inference_workload.run_data, "_tokenizer", lambda path: tokenizer)
    candidates = [
        {
            "input": text,
            "metadata": {"prepared_sample_id": str(index), "verifiers": [{"name": "gsm8k", "target": "1"}]},
        }
        for index, text in enumerate(["too long", "one", "two", "last"])
    ]
    monkeypatch.setattr(prepare_inference_workload.run_data, "_tasks", lambda *args: ({"train": candidates}, {}, None))
    result = prepare_inference_workload.prepare(model, tmp_path / "corpus", counts={"gsm8k": 2}, max_prompt_length=3)
    rows = [json.loads(line) for line in Path(result["prompt_data"]).read_text().splitlines()]
    assert {r["input"] for r in rows} == {"one", "two"}
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert all(Path(path).is_file() for path in manifest["inputs"])


def test_replay_comparison_pairs_prompt_clusters_and_rejects_different_trajectories():
    left = [
        {
            "index": i,
            "group_index": i // 2,
            "prompt_tokens": 1,
            "response_tokens": 2,
            "logprobs": [-1.0, -2.0],
            "route_sha256": "same",
        }
        for i in range(8)
    ]
    exact = compare_inference_replay.paired_metrics(left, left)
    assert exact["prompt_clusters"] == 4
    assert exact["metrics"]["mean_absolute_logprob_delta"]["value"] == 0
    right = copy.deepcopy(left)
    for row in right:
        row["logprobs"] = [-0.5, -1.5]
    shifted = compare_inference_replay.paired_metrics(left, right)
    assert shifted["metrics"]["mean_signed_logprob_delta"]["prompt_cluster_95pct_interval"] == [0.5, 0.5]
    assert shifted["metrics"]["ratio_outside_0_8_to_1_2_fraction"]["value"] == 1
    right[0]["index"] = 9
    with pytest.raises(ValueError, match="identities"):
        compare_inference_replay.paired_metrics(left, right)
