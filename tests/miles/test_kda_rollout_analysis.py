import json
import pickle

import pytest
from scripts.miles import analyze_kda_rollout, benchmark_kda_rollout


def populate(root, native_flags=None, count=4096):
    for precision in ("bf16", "combined"):
        for mode in ("pinned", "native"):
            folder = root / "speed/kda-throughput-20261116" / precision / mode
            folder.mkdir(parents=True)
            flags = native_flags if mode == "native" and native_flags is not None else {"capacity": 32}
            (folder / "requested-server-options.json").write_text(json.dumps(flags))
            for repeat in range(2):
                seconds = 2 if mode == "native" else 1
                record = {
                    "warmup": False,
                    "responses": 64,
                    "output_tokens": 64 * 4096,
                    "samples": [{"output_tokens": count}] * 64,
                    "seconds": seconds,
                    "output_tokens_per_second": 64 * 4096 / seconds,
                }
                (folder / f"fixed-{repeat}.json").write_text(json.dumps(record))


def test_no_measurements_remain_in_progress(tmp_path):
    assert analyze_kda_rollout.summarize(tmp_path)["status"] == "IN PROGRESS"


def test_fixed_workload_ratio_and_missing_counts(tmp_path):
    populate(tmp_path)
    report = analyze_kda_rollout.summarize(tmp_path)
    assert report["status"] == "MEASURED"
    assert report["precisions"]["bf16"]["pinned_over_native_throughput_ratio"] == 2
    record = tmp_path / "speed/kda-throughput-20261116/bf16/native/fixed-0.json"
    value = json.loads(record.read_text())
    value["samples"][0]["output_tokens"] = 4095
    record.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Per-response"):
        analyze_kda_rollout.summarize(tmp_path)


def test_changed_server_configuration_rejected(tmp_path):
    populate(tmp_path, native_flags={"capacity": 64})
    with pytest.raises(ValueError, match="Server settings differ"):
        analyze_kda_rollout.summarize(tmp_path)


def test_scheduler_receipt_removes_unpickleable_runtime_configs():
    class DynamicModelConfig:
        pass

    raw = {"model_config": DynamicModelConfig(), "max_running_requests": 32}
    with pytest.raises(AttributeError):
        pickle.dumps(raw)
    receipt = benchmark_kda_rollout.plain_receipt(raw)
    assert pickle.loads(pickle.dumps(receipt))["max_running_requests"] == 32
    assert isinstance(receipt["model_config"], str)


def test_window_metrics_exclude_warmup_and_separate_processing(tmp_path):
    timeline = [
        {"monotonic": 9, "unix": 109, "metrics": 'sglang:num_running_reqs{rank="0"} 99\n'},
        {
            "monotonic": 11,
            "unix": 111,
            "metrics": 'sglang:num_running_reqs{rank="0"} 32\nsglang:num_queue_reqs{rank="0"} 16\n',
        },
    ]
    (tmp_path / "engine-timeline.jsonl").write_text("\n".join(json.dumps(r) for r in timeline))
    sample = {
        "submitted": 10,
        "admitted": 10.5,
        "response_processed": 11,
        "assembled": 12,
        "reward_seconds": 0.25,
        "prompt_tokens": 128,
        "output_tokens": 4096,
    }
    row = {
        "samples": [sample],
        "seconds": 2,
        "responses": 1,
        "groups": [{"tail_seconds": 1}],
        "time_origin_monotonic": 10,
    }
    result = analyze_kda_rollout.window_details(row, tmp_path)
    assert result["engine_gauges"]["num_running_reqs"]["mean"] == 32
    assert result["admission_seconds"]["mean"] == 0.5
    assert result["processed_to_group_assembled_seconds"]["mean"] == 1
    assert result["reward_seconds"]["mean"] == 0.25


def test_audit_rejects_native_with_installed_policy(tmp_path):
    state = {
        "internal_states": [{"kda_benchmark": {"loader_context_fix": True, "targets": [{"installed_policy": "hash"}]}}]
    }
    (tmp_path / "warmed-state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="Actual policy"):
        analyze_kda_rollout.runtime_audit(tmp_path, "native")
