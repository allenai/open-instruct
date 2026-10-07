"""Concurrency contract tests, independent of the model/runtime dependencies."""

import pytest
from scripts.miles import analyze_engine_drain

from open_instruct.miles.configuration.config import CoreConfig, RunConfig


def test_mode_configuration_rejects_unsupported_combinations():
    options = dict(
        hf_checkpoint="/fixture/hf",
        global_batch_size=4,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        fully_async=True,
    )
    core = CoreConfig(publication_mode="engine_drain", max_policy_lag=2)
    RunConfig(core, options).validate()
    for change, match in [
        ({"fully_async": False}, "fully_async"),
        ({"rollout_num_gpus_per_engine": 2}, "TP1"),
        ({"eval_num_gpus": 1}, "shared-engine"),
        ({"prefill_num_servers": 1}, "prefill/decode"),
        ({"sglang_config": "/some/servers.yaml"}, "single-turn"),
        ({"load_debug_rollout_data": "/some/batch.pt"}, "single-turn"),
        ({"use_fault_tolerance": True}, "use_fault_tolerance"),
        ({"custom_generate_function_path": "custom.generate"}, "single-turn"),
    ]:
        with pytest.raises(ValueError, match=match):
            RunConfig(core, options | change).validate()
    with pytest.raises(ValueError, match="engine_drain_timeout"):
        CoreConfig(engine_drain_timeout=0)


def test_timeline_audit_does_not_confuse_phase_sums_with_overlap():
    events = [
        {"event": "group_reserved", "engine": "fast", "version": 0, "group": 1, "requests": ["r1"]},
        {
            "event": "decode_finished",
            "engine": "fast",
            "group": 1,
            "request": "r1",
            "tokens": 10,
            "assigned_version": 0,
            "executed_version": 0,
        },
        {"event": "drain_started", "engine": "fast", "version": 0, "time": 1},
        {"event": "drain_started", "engine": "slow", "version": 0, "time": 1},
        {"event": "drain_finished", "engine": "fast", "time": 2},
        {"event": "update_started", "engine": "fast", "time": 2},
        {"event": "engine_reopened", "engine": "fast", "version": 1, "time": 3},
        {"event": "drain_finished", "engine": "slow", "time": 5},
    ]
    result = analyze_engine_drain.analyze(
        events,
        [
            {"stage": "training", "passed": True, "started_unix": 2, "seconds": 2, "rollout_id": 1},
            {"stage": "training", "passed": True, "started_unix": 6, "seconds": 2, "rollout_id": 2},
        ],
    )
    assert result["ownership_errors"] == []
    assert len(result["fast_reopened_while_older_peer_drained"]) == 1
    assert [r["rollout_id"] for r in result["optimizer_completions_during_publication"]] == [1]
    assert result["generated_tokens_observed"] == 10
