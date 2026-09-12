"""Check matching arms, actual CLI parsing and measurement failure semantics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from miles.utils import arguments
from scripts.miles import analyze_control_exercise, exercise_controls, launch_control_exercise
from transformers import Qwen3Config

from open_instruct.miles.timing import evaluation_stage, stage


def test_pair_differs_only_in_scheduling_and_output(tmp_path):
    sync = exercise_controls.configuration(Path("/prepared"), tmp_path, "sync", 24)
    asynchronous = exercise_controls.configuration(Path("/prepared"), tmp_path, "async", 24)
    differing = {
        key
        for key in sync.miles.keys() | asynchronous.miles.keys()
        if sync.miles.get(key) != asynchronous.miles.get(key)
    }
    assert differing == {
        "fully_async",
        "async_data_buffer_capacity_factor",
        "async_unused_samples_handler",
        "rollout_submission_granularity",
    }
    assert sync.core.row_specialization == asynchronous.core.row_specialization == "dynamic"
    assert sync.core.max_policy_lag == 0 and asynchronous.core.max_policy_lag == 1


@pytest.mark.parametrize("arm", exercise_controls.ARMS)
def test_toml_roundtrip_and_native_parser(tmp_path, arm, monkeypatch):
    Qwen3Config().save_pretrained(tmp_path / "hf")
    config = exercise_controls.configuration(tmp_path, tmp_path, arm, 4)
    exercise_controls.write_config(tmp_path / "run.toml", config)
    monkeypatch.setattr("sys.argv", ["control-exercise", *config.arguments()])
    args = arguments.parse_args()
    assert args.olmo_core.row_specialization == "dynamic"
    assert args.olmo_core.scoring_pass_required, "historical exercise audits every standalone scoring pass"
    assert args.num_rollout == 4
    if arm == "controls":
        assert args.use_tis and not args.use_rollout_logprobs
        assert args.save_interval == args.eval_interval == 4
        assert args.entropy_coef > 0


def test_stage_records_failures_without_swallowing(tmp_path):
    with pytest.raises(ValueError), stage(SimpleNamespace(save=str(tmp_path)), "training", 3):
        raise ValueError("injected")
    row = json.loads((tmp_path / "driver_timing.jsonl").read_text())
    assert not row["passed"] and row["rollout_id"] == 3 and row["seconds"] >= 0


def test_grouped_launch_is_bounded_and_on_holmes():
    tasks = launch_control_exercise.specification("image")["tasks"]
    assert sum(t["resources"]["gpuCount"] for t in tasks) == 8
    assert all(t["context"]["priority"] == "urgent" and t["context"]["minRuntime"] == "1h" for t in tasks)
    assert all(t["constraints"]["cluster"] == ["ai2/holmes"] for t in tasks)
    assert "for arm in sync async" in tasks[1]["arguments"][0]


@pytest.mark.parametrize("arm,groups_per_collection", (("sync", 4), ("sync-admission64", 16)))
def test_audit_reads_metric_schemas_without_prompt_metadata(tmp_path, monkeypatch, arm, groups_per_collection):
    campaign, output = tmp_path / "prepared", tmp_path / "output"
    campaign.mkdir()
    (output / "metrics").mkdir(parents=True)
    (output / "rollouts").mkdir()
    prepared = [
        {"input": f"prompt{i}", "metadata": {"prepared_sample_id": str(i)}} for i in range(4 * groups_per_collection)
    ]
    (campaign / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prepared))
    monkeypatch.setattr(
        exercise_controls.prepare_gsm8k_parity,
        "verify_preparation",
        lambda _: {
            "partitions": {
                "train": {"rows": [{"prepared_sample_id": str(i)} for i in range(4 * groups_per_collection)]}
            }
        },
    )
    monkeypatch.setattr(
        exercise_controls.evidence,
        "audit_dump",
        lambda *a, **k: {"valid": True, "summary": {"mean_response_tokens": 10}},
    )
    monkeypatch.setattr(exercise_controls.evidence, "parse_timing_log", lambda *a, **k: {})
    config = exercise_controls.configuration(campaign, output, arm, 4)
    exercise_controls.write_config(output / "run.toml", config)
    rows, stages = [], []
    for update in range(4):
        samples = [
            {"metadata": {"prepared_sample_id": str(group)}, "group_index": group, "weight_versions": [str(update)]}
            for group in range(groups_per_collection * update, groups_per_collection * (update + 1))
            for _ in range(4)
        ]
        torch.save({"samples": samples}, output / f"rollouts/{update}.pt")
        rows += [
            {
                "event": "optimizer",
                "step": update + 1,
                "optimizer_skipped": False,
                "local_behavior_versions": [update],
                "normalization": {"samples": 4 * groups_per_collection},
            },
            {"event": "score_timing", "rollout_id": update, "row_specialization": "dynamic", "seconds": 1.0},
        ]
        stages += [
            {"stage": name, "rollout_id": update, "passed": True, "seconds": 1.0}
            for name in ("generation_wait", "training", "publication")
        ]
    for rank in (0, 1):
        (output / f"metrics/training_contract_rank{rank}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    (output / "metrics/publication.jsonl").write_text(
        "".join(json.dumps({"version": v, "repeated_version": False}) + "\n" for v in range(5))
    )
    stages += [{"stage": "evaluation", "rollout_id": i, "passed": True, "seconds": 100} for i in (0, 3)]
    (output / "metrics/driver_timing.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stages))
    (output / "elapsed.json").write_text('{"seconds": 12}')
    exercise_controls.audit(campaign, output, arm, 4)
    report = json.loads((output / "audit.json").read_text())
    assert report["passed"]
    assert report["cycle_seconds"]["mean"] == 3
    assert len(report["evaluation_timings"]) == 2


@pytest.mark.parametrize("base", ("sync", "async"))
def test_admission_followup_changes_only_capacity_and_batch_size(tmp_path, base):
    before = exercise_controls.configuration(Path("/prepared"), tmp_path, base, 12)
    after = exercise_controls.configuration(Path("/prepared"), tmp_path, base + "-admission64", 12)
    assert before.core == after.core
    assert {k for k in before.miles if before.miles[k] != after.miles[k]} == set(exercise_controls.ADMISSION_KEYS)
    assert after.miles["sglang_server_concurrency"] == after.miles["sglang_max_running_requests"] == 64
    assert after.miles["rollout_batch_size"] * after.miles["n_samples_per_prompt"] == 64
    assert after.miles["sglang_max_total_tokens"] >= 64 * after.miles["sglang_context_length"]
    assert exercise_controls.is_async(base + "-admission64") == (base == "async")


def test_admission_launch_uses_one_bounded_allocation():
    (task,) = launch_control_exercise.specification("image", admission_only=True)["tasks"]
    assert task["resources"]["gpuCount"] == 3
    assert task["context"]["priority"] == "urgent"
    assert task["context"]["minRuntime"] == "1h"
    assert "for arm in sync-admission64 async-admission64" in task["arguments"][0]
    assert "--updates 12" in task["arguments"][0]


@pytest.mark.parametrize("samples", (16, 64))
def test_warm_throughput_excludes_cold_tokens_and_handles_larger_batches(tmp_path, samples):
    # Four cold collections are deliberately much larger than the two warm ones.
    training = [{"summary": {"mean_response_tokens": n}, "policy_lags": [0]} for n in (1000, 1000, 1000, 1000, 10, 30)]
    scores = [{"rollout_id": i, "seconds": 1, "model_tokens": 100} for i in range(6)]
    report = dict(
        passed=True,
        updates=6,
        consumed_samples=samples * 6,
        training=training,
        scoring_by_rank=[scores, scores],
        consumed_response_tokens=4040 * samples,
        measured_cycle_seconds=104,
        post_first_four_cycle_seconds={"mean": 2},
        native_log_timing={"phases": {"generation": {}}},
    )
    (tmp_path / "audit.json").write_text(json.dumps(report))
    measured = analyze_control_exercise.live(tmp_path)
    assert measured["post_first_four_consumed_response_tokens"] == 40 * samples
    assert measured["post_first_four_consumed_tokens_per_cycle_second"] == 10 * samples
    assert measured["samples_per_collection"] == samples


@pytest.mark.parametrize("snapshots", (False, True))
def test_evaluation_failure_and_snapshot_dispatch_are_labeled(tmp_path, snapshots):
    args = SimpleNamespace(save=str(tmp_path), eval_uses_snapshots=snapshots, sglang_server_concurrency=64)
    with pytest.raises(RuntimeError, match="eval failed"), evaluation_stage(args, 0, initial=True):
        raise RuntimeError("eval failed")
    row = json.loads((tmp_path / "driver_timing.jsonl").read_text())
    assert not row["passed"]
    assert row["stage"] == ("evaluation_dispatch" if snapshots else "evaluation")
    assert row["details"]["scope"] == ("snapshot_submission" if snapshots else "blocking_shared_engine_evaluation")
