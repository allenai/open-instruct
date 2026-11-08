"""Regression checks for the experiment's numerical and training coverage gates."""

import json

from scripts.miles.cache_panel import run


def test_different_trajectories_are_not_reported_as_logprob_parity():
    a = {"tokens": [1, 2], "logprobs": [-1.0, -2.0]}
    b = {"tokens": [1, 3], "logprobs": [-1.0, -2.0]}
    assert not run.parity(a, b)["passed"]
    assert run.parity(a, b)["max_logprob_error"] is None
    assert run.parity(a, a)["passed"]
    assert not run.parity(a, {**a, "logprobs": [-1.0, -2.01]})["passed"]


def test_training_coverage_uses_consumed_contract_not_generated_spans(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    contract = [
        {"event": "optimizer", "local_behavior_versions": [0]},
        {"event": "refresh_scores", "mixed_responses": 0},
    ]
    (root / "training_contract_rank0.jsonl").write_text("\n".join(map(json.dumps, contract)))
    timing = [
        {"stage": "training", "rollout_id": 0, "seconds": 100},
        {"stage": "training", "rollout_id": 4, "seconds": 20},
        {"stage": "publication", "rollout_id": 4, "seconds": 2},
        {"stage": "generation_wait", "rollout_id": 4, "seconds": 3},
    ]
    (root / "driver_timing.jsonl").write_text("\n".join(map(json.dumps, timing)))
    (root / "sibling_timing_a.jsonl").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "engine_reported_cached_tokens": 100,
                        "engine_reported_prompt_tokens": 120,
                        "engine_weight_versions": [{"version": 0}, {"version": 1}],
                        "response_tokens": 16,
                    },
                    {"response_tokens": 8},
                ]
            }
        )
    )
    result = run.training_summary(tmp_path)
    assert result["consumed_cross_version_responses"] == 0
    assert result["steady_mean_seconds"] == 25
    assert result["cached_tokens"] == 100
    assert result["cache_observation_coverage"] == 1
    assert result["generated_response_records"] == 2


def test_logprob_comparison_stops_when_token_histories_diverge():
    a = {"tokens": [1, 2, 3], "logprobs": [-1.0, -2.0, -99.0]}
    b = {"tokens": [1, 4, 3], "logprobs": [-1.001, -2.0, 99.0]}
    result = run.trajectory_difference(a, b)
    assert result["common_prefix_tokens"] == 1
    assert result["max_common_prefix_logprob_error"] < 0.002
    assert not result["same_tokens"]
