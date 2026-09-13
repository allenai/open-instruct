"""Hard constraints, advisory limits, and reproducible topology comparisons."""

import json

import pytest
from scripts.miles import throughput_basket

from open_instruct.miles import throughput, validation
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.run_spec import RunSpec


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


@pytest.mark.parametrize(
    "case,gpus,replicas",
    [
        ("dev", 1, 1),
        ("tiny", 2, 1),
        ("small-2t4i-group", 6, 1),
        ("small-4t2i-group", 6, 1),
        ("small-2t4i-sample", 6, 1),
        ("small-2t4i-c8", 6, 1),
        ("bridge-2t6i", 8, 1),
        ("bridge-8t8i", 16, 2),
        ("large-8t56i", 64, 8),
    ],
)
def test_basket_allocates_requested_policy_gpus(case, gpus, replicas):
    spec = throughput_basket.specification(case, "/weka/oe-training-default/test/" + case + "/run")
    plan = spec.plan()
    assert plan["allocation"]["policy_gpus"] == gpus
    assert plan["allocation"]["replicas"] == replicas
    assert not spec.compile().miles.get("eval_interval")
    assert plan["launch"]["auto_resume"] is False


def test_small_comparisons_hold_model_data_and_objective_fixed():
    cases = [
        throughput_basket.specification(name, "/weka/oe-training-default/test/" + name + "/run")
        for name in throughput_basket.CASES
        if name.startswith("small-")
    ]
    for spec in cases:
        assert spec.model == cases[0].model
        assert spec.data == cases[0].data
        config = spec.compile()
        assert config.miles["global_batch_size"] == 32
        assert config.miles["rollout_max_response_len"] == 4096
        assert config.miles["lr"] == 1e-6
        assert config.core.max_policy_lag == 2
        assert config.core.publication_mode == "refresh"


@pytest.mark.parametrize("profile", ["dev", "tiny", "small", "large"])
def test_self_contained_example_plans(profile):
    spec = RunSpec.load(throughput_basket.ROOT / f"configs/miles/examples/{profile}.toml")
    assert spec.data["tasks"][0]["task"] == "gsm8k"
    assert spec.plan()["runtime"]["throughput"]["publication_mode"] in ("barrier", "refresh")


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
        [dict(rollout_id=i, response_tokens=100, mixed_responses=1, queue_metrics={}) for i in range(4)],
    )
    write(
        "training_contract_rank0.jsonl",
        [dict(event="optimizer", step=i + 1, optimizer_skipped=False) for i in range(4)],
    )
    result = throughput_basket.analyze(tmp_path, warmup=1)
    assert result["warm_cycle_seconds"] == 9
    assert result["useful_response_tokens_per_second"] == pytest.approx(300 / 9)
    assert result["all_driver_stage_seconds"]["checkpoint"] == 100
    assert len(result["per_update"]) == 3
    assert result["validated_trainer_ranks"] == 1
    (tmp_path / "plan.json").write_text(
        json.dumps({"runtime": {"miles": {"num_rollout": 4, "actor_num_gpus_per_node": 2}}})
    )
    with pytest.raises(ValueError, match="rank 1"):
        throughput_basket.analyze(tmp_path, warmup=1)
    (tmp_path / "plan.json").write_text(json.dumps({"runtime": {"miles": {"num_rollout": 4, "fully_async": True}}}))
    with pytest.raises(ValueError, match="queue counters"):
        throughput_basket.analyze(tmp_path, warmup=1)
    write("training_contract_rank0.jsonl", [])
    with pytest.raises(ValueError, match="optimizer"):
        throughput_basket.analyze(tmp_path, warmup=1)


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_refresh_request_deadline_is_finite_and_positive(value):
    with pytest.raises(ValueError):
        CoreConfig(refresh_request_timeout=value)
