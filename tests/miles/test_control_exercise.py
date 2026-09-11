"""Check matching arms, actual CLI parsing and measurement failure semantics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
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
