"""Throughput denominators and missing observations must not inflate efficiency."""

import json

import pytest
from scripts.miles import capacity_metrics, throughput_occupancy

from open_instruct.miles.training import performance


def test_training_rate_counts_global_tokens_once():
    rates = performance.training_rates(1200, 800, 2, 4)
    assert rates == {
        "model_tokens_per_second": 600,
        "model_tokens_per_gpu_second": 150,
        "active_response_tokens_per_gpu_second": 100,
    }


@pytest.mark.parametrize("args", [(10, 11, 2, 1), (10, 2, 0, 1), (10, 2, 2, 0), (10, 2, float("nan"), 1)])
def test_invalid_rate_denominators_rejected(args):
    with pytest.raises(ValueError):
        performance.training_rates(*args)


def test_window_missing_coverage_is_not_idle():
    summary = throughput_occupancy.summarize([(0, 100), (20, None)], 5, 25)
    assert summary["coverage_fraction"] == 0.25
    assert summary["mean"] == 100
    assert summary["empty_fraction"] == 0


def test_failed_or_missing_run_cannot_be_published_as_capacity_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        capacity_metrics.measurements(tmp_path)


def test_reconstructed_rates_use_global_counts_and_slowest_rank(tmp_path, monkeypatch):
    plan = {
        "allocation": {"nodes": [{"trainer_gpus": 2, "rollout_gpus": 2}], "allocated_gpus": 4},
        "runtime": {"miles": {"rollout_num_gpus_per_engine": 1}},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    cycle = dict(
        rollout_id=0,
        response_tokens=600,
        generation_wait_seconds=1,
        training_seconds=10,
        publication_seconds=1,
        completed_queue_get_seconds=0.2,
        other_collection_seconds=0.8,
    )
    monkeypatch.setattr(capacity_metrics.throughput_basket, "analyze", lambda *a, **k: {"per_update": [cycle]})

    def rows(path):
        if path.name.startswith("training_contract"):
            rank = int(path.stem[-1])
            return [
                dict(
                    event="optimizer",
                    rollout_id=0,
                    elapsed_seconds=4 + 2 * rank,
                    normalization=dict(model_tokens=1000, active_tokens=600),
                ),
                dict(event="score_timing", rollout_id=0, seconds=2 + rank, model_tokens=400 + 200 * rank),
            ]
        if path.name == "driver_timing.jsonl":
            return [
                dict(rollout_id=0, stage=name, seconds=seconds, started_unix=start)
                for name, seconds, start in [("generation_wait", 1, 0), ("training", 10, 1), ("publication", 1, 11)]
            ]
        return [dict(rollout_id=0, queue_metrics={}, mixed_responses=0)]

    monkeypatch.setattr(capacity_metrics.throughput_basket, "rows", rows)
    monkeypatch.setattr(throughput_occupancy, "node_roles", lambda _: {})
    monkeypatch.setattr(
        throughput_occupancy, "analyze", lambda *a, **k: {"pipeline": {}, "hardware": {}, "engines": {}}
    )
    row = capacity_metrics.measurements(tmp_path)["rows"][0]
    assert row["trainer/model_tokens_per_gpu_second"] == pytest.approx(1000 / 6 / 2)
    assert row["trainer/scoring_model_tokens_per_gpu_second"] == pytest.approx(1000 / 3 / 2)
    assert row["inference/useful_response_tokens_per_gpu_cycle_second"] == 25
    assert row["pipeline/useful_response_tokens_per_allocated_gpu_second"] == 12.5
    assert "inference/observed_decode_tokens_per_gpu_second" not in row
