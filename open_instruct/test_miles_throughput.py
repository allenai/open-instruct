"""Hard constraints, advisory limits, and reproducible topology comparisons."""

import json
from pathlib import Path

import pytest
from scripts.miles import analyze_throughput, sample_gpu_usage, throughput_occupancy

from open_instruct.miles.configuration import throughput, validation
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.configuration.run_spec import RunSpec


@pytest.mark.parametrize(
    "values",
    [
        {"rollout_num_gpus": 3, "rollout_num_gpus_per_engine": 2},
        {"update_weights_interval": 0},
        {"sglang_chunked_prefill_size": 0},
        {"sglang_chunked_prefill_size": -2},
        {"sglang_chunked_prefill_size": True},
    ],
)
def test_invalid_limits_fail_before_runtime(values):
    with pytest.raises(ValueError):
        validation.runtime_values(values)


def test_prefill_disable_sentinel_is_preserved():
    validation.runtime_values({"sglang_chunked_prefill_size": -1})


def test_advice_does_not_reject_intentional_small_pools_or_diagnostics():
    options = dict(
        rollout_num_gpus=4,
        rollout_num_gpus_per_engine=1,
        sglang_max_running_requests=16,
        sglang_server_concurrency=16,
        sglang_max_total_tokens=4096,
        rollout_max_context_len=6144,
        sglang_chunked_prefill_size=8192,
        rollout_batch_size=2,
        n_samples_per_prompt=4,
        sglang_cuda_graph_backend_decode="full",
        sglang_cuda_graph_max_bs_decode=4,
        colocate=True,
        save_interval=2,
        kl_loss_coef=0.01,
    )
    codes = {r["code"] for r in throughput.report(options, CoreConfig(diagnostic_interval=1))["warnings"]}
    assert {
        "token_pool_below_context",
        "prefill_chunk_above_pool",
        "decode_graph_coverage",
        "sync_collection_underfeeds_fleet",
        "resident_colocation_fit",
        "frequent_save",
        "reference_policy_cost",
        "diagnostic_overhead",
    } <= codes


@pytest.mark.parametrize("profile,gpus", [("dev", 1), ("small", 2), ("medium", 16), ("large", 56)])
def test_self_contained_example_plans(profile, gpus):
    spec = RunSpec.load(Path(__file__).resolve().parents[1] / f"configs/miles/examples/{profile}.toml")
    assert spec.data.get("prompt_data") or spec.data["tasks"][0]["task"] == "gsm8k-less-noise"
    assert spec.plan()["allocation"]["allocated_gpus"] == gpus
    assert spec.plan()["runtime"]["throughput"]["publication_mode"] in ("barrier", "refresh")
    assert spec.compile().core.max_policy_lag == (6 if profile in ("medium", "large") else 0)


def test_analyzer_requires_complete_updates_and_excludes_lifecycle_time(tmp_path):
    metrics = tmp_path / "checkpoints"
    metrics.mkdir()

    def write(name, rows):
        (metrics / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    (tmp_path / "plan.json").write_text(json.dumps({"runtime": {"miles": {"num_rollout": 4}}}))
    (tmp_path / "workflow.json").write_text(json.dumps({"status": "complete"}))
    write(
        "driver_timing.jsonl",
        [
            dict(stage=stage, rollout_id=i, seconds=1, passed=True)
            for i in range(4)
            for stage in ("generation_wait", "training", "publication")
        ]
        + [dict(stage="checkpoint", rollout_id=3, seconds=100, passed=True)],
    )
    write(
        "rollout_flow.jsonl",
        [
            dict(
                rollout_id=i,
                response_tokens=100,
                mixed_responses=1,
                queue_metrics={"rollout/fully_async/completed_queue/consumer_wait_seconds": 0.25},
            )
            for i in range(4)
        ],
    )
    write(
        "training_contract_rank0.jsonl",
        [dict(event="optimizer", step=i + 1, optimizer_skipped=False) for i in range(4)],
    )
    result = analyze_throughput.analyze(tmp_path, warmup=1)
    assert result["batch_collection_breakdown"]["completed_queue_get_seconds"] == 0.75
    assert result["batch_collection_breakdown"]["other_collection_seconds"] == 2.25
    assert result["warm_cycle_seconds"] == 9
    assert result["useful_response_tokens_per_second"] == pytest.approx(300 / 9)
    assert result["all_driver_stage_seconds"]["checkpoint"] == 100
    assert len(result["per_update"]) == 3
    assert result["validated_trainer_ranks"] == 1
    (tmp_path / "workflow.json").write_text(json.dumps({"status": "failed", "error": "cleanup timed out"}))
    with pytest.raises(ValueError, match="Workflow"):
        analyze_throughput.analyze(tmp_path, warmup=1)
    measured = analyze_throughput.analyze(tmp_path, warmup=1, allow_incomplete_workflow=True)
    assert not measured["end_to_end_passed"]
    assert measured["workflow"]["error"] == "cleanup timed out"
    (tmp_path / "workflow.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "plan.json").write_text(
        json.dumps({"runtime": {"miles": {"num_rollout": 4, "actor_num_gpus_per_node": 2}}})
    )
    with pytest.raises(ValueError, match="rank 1"):
        analyze_throughput.analyze(tmp_path, warmup=1)
    (tmp_path / "plan.json").write_text(json.dumps({"runtime": {"miles": {"num_rollout": 4, "fully_async": True}}}))
    with pytest.raises(ValueError, match="queue counters"):
        analyze_throughput.analyze(tmp_path, warmup=1)
    prefix = "rollout/fully_async/completed_queue/"
    write(
        "rollout_flow.jsonl",
        [
            dict(
                rollout_id=i,
                response_tokens=100,
                mixed_responses=1,
                queue_metrics={prefix + "dropped_response_tokens": 0, prefix + "delivered_response_tokens": 101},
            )
            for i in range(4)
        ],
    )
    with pytest.raises(ValueError, match="delivery accounting"):
        analyze_throughput.analyze(tmp_path, warmup=1)
    write("training_contract_rank0.jsonl", [])
    with pytest.raises(ValueError, match="optimizer"):
        analyze_throughput.analyze(tmp_path, warmup=1)


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_refresh_request_deadline_is_finite_and_positive(value):
    with pytest.raises(ValueError):
        CoreConfig(refresh_request_timeout=value)


@pytest.mark.parametrize("value", [-1, True, float("inf"), float("nan")])
def test_pipeline_observation_interval_is_nonnegative_and_finite(value):
    with pytest.raises(ValueError):
        CoreConfig(pipeline_observation_interval=value)


def test_occupancy_summary_weights_time_and_preserves_missing_intervals():
    result = throughput_occupancy.summarize([(0, 2), (2, 0), (4, None), (20, 10)], 1, 25, max_hold=5)
    assert result["coverage_fraction"] == pytest.approx(8 / 24)
    assert result["mean"] == pytest.approx(52 / 8)
    assert result["empty_fraction"] == pytest.approx(2 / 8)
    assert result["p95"] == result["maximum"] == 10
    assert throughput_occupancy.summarize([], 0, 10)["mean"] is None


def test_gpu_usage_preserves_unavailable_values(monkeypatch):
    monkeypatch.setattr(
        sample_gpu_usage.subprocess, "check_output", lambda *a, **kw: "0, GPU-one, 75, N/A, 1000, 2000\n"
    )
    row = sample_gpu_usage.sample()[0]
    assert row["uuid"] == "GPU-one" and row["utilization.gpu"] == 75
    assert row["utilization.memory"] is None
    assert row["memory.used"] == 1000


def test_graph_warning_uses_json_over_convenience_and_legacy_flags():
    report = throughput.report(
        {
            "sglang_max_running_requests": 8,
            "sglang_disable_cuda_graph": True,
            "sglang_cuda_graph_backend_decode": "disabled",
            "sglang_cuda_graph_max_bs_decode": 32,
            "sglang_cuda_graph_config": {"decode": {"backend": "full", "max_bs": 4}},
        },
        CoreConfig(),
    )
    assert any(issue["code"] == "decode_graph_coverage" for issue in report["warnings"])


def test_hardware_roles_follow_observed_ips_not_replica_order(tmp_path):
    cluster = tmp_path / "cluster" / "attempt"
    cluster.mkdir(parents=True)
    nodes = [dict(trainer_gpus=8, rollout_gpus=0), dict(trainer_gpus=0, rollout_gpus=8)]
    layout = dict(replicas=2, nodes=nodes)
    for rank, address in enumerate(["10.0.0.20", "10.0.0.3"]):
        (cluster / f"node-{rank}.json").write_text(json.dumps(dict(rank=rank, address=address)))
        (cluster / f"placement-{rank}.json").write_text(json.dumps(dict(address=address, layout=layout)))
    assert throughput_occupancy.node_roles(tmp_path) == {"0": nodes[1], "1": nodes[0]}


def test_terminal_unused_work_requires_final_boundary_and_uses_entire_run():
    assert analyze_throughput.terminal_unused_work(None, delivered_tokens=80, stale_dropped_tokens=10) is None
    record = {
        "event": "shutdown_complete",
        "completed_queue": dict(groups=2, samples=8, response_tokens=20),
        "producer_ready": dict(groups=0, samples=0, response_tokens=0),
        "shutdown_unqueued": dict(groups=1, samples=4, response_tokens=10),
    }
    result = analyze_throughput.terminal_unused_work([record], delivered_tokens=80, stale_dropped_tokens=10)
    assert result["samples"] == 12 and result["response_tokens"] == 30
    assert result["fraction_of_accounted_response_tokens"] == 0.25
    record["event"] = "shutdown_incomplete"
    assert analyze_throughput.terminal_unused_work([record], delivered_tokens=80, stale_dropped_tokens=10) is None


def test_triton_artifact_sampler_tracks_growth_without_unrelated_caches(tmp_path):
    worker = tmp_path / "core-triton-test" / "triton" / "kernel"
    worker.mkdir(parents=True)
    (worker / "a.cubin").write_bytes(b"compiled")
    (worker / "a.json").write_text("{}")
    (tmp_path / "unrelated.cubin").write_bytes(b"ignore")
    first = sample_gpu_usage.triton_artifacts(tmp_path)
    assert len(first) == 1 and first[0]["cubins"] == 1
    assert first[0]["newest_write_unix"] > 0
    (worker / "b.cubin").write_bytes(b"another")
    assert sample_gpu_usage.triton_artifacts(tmp_path)[0]["cubins"] == 2
