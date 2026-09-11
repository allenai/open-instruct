"""Check matching arms, actual CLI parsing and measurement failure semantics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from miles.utils import arguments
from scripts.miles import exercise_controls, launch_control_exercise
from transformers import Qwen3Config

from open_instruct.miles.timing import stage


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


def test_audit_reads_metric_schemas_without_prompt_metadata(tmp_path, monkeypatch):
    campaign, output = tmp_path / "prepared", tmp_path / "output"
    campaign.mkdir()
    (output / "metrics").mkdir(parents=True)
    (output / "rollouts").mkdir()
    prepared = [{"input": f"prompt{i}", "metadata": {"prepared_sample_id": str(i)}} for i in range(16)]
    (campaign / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in prepared))
    monkeypatch.setattr(
        exercise_controls.prepare_gsm8k_parity,
        "verify_preparation",
        lambda _: {"partitions": {"train": {"rows": [{"prepared_sample_id": str(i)} for i in range(16)]}}},
    )
    monkeypatch.setattr(
        exercise_controls.evidence,
        "audit_dump",
        lambda *a, **k: {"valid": True, "summary": {"mean_response_tokens": 10}},
    )
    monkeypatch.setattr(exercise_controls.evidence, "parse_timing_log", lambda *a, **k: {})
    config = exercise_controls.configuration(campaign, output, "sync", 4)
    exercise_controls.write_config(output / "run.toml", config)
    rows, stages = [], []
    for update in range(4):
        samples = [
            {"metadata": {"prepared_sample_id": str(group)}, "group_index": group, "weight_versions": [str(update)]}
            for group in range(4 * update, 4 * update + 4)
            for _ in range(4)
        ]
        torch.save({"samples": samples}, output / f"rollouts/{update}.pt")
        rows += [
            {
                "event": "optimizer",
                "step": update + 1,
                "optimizer_skipped": False,
                "local_behavior_versions": [update],
                "normalization": {"samples": 16},
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
    (output / "metrics/driver_timing.jsonl").write_text("".join(json.dumps(row) + "\n" for row in stages))
    (output / "elapsed.json").write_text('{"seconds": 12}')
    exercise_controls.audit(campaign, output, "sync", 4)
    assert json.loads((output / "audit.json").read_text())["passed"]
