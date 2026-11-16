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
