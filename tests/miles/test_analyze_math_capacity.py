"""Capacity selection must fail closed on mixed allocations and incomplete probes."""

import json

import pytest
from scripts.miles import analyze_math_capacity as analysis


@pytest.fixture
def pilot(tmp_path):
    attempt = tmp_path / "attempt-job"
    best = None
    for concurrency in (16, 32):
        folder = attempt / str(concurrency)
        folder.mkdir(parents=True)
        options = {
            "max_running_requests": concurrency,
            "context_length": 18434,
            "max_total_tokens": concurrency * 18434,
        }
        state = {
            "capacity_pilot": {
                "peak_allocated_bytes": 1024,
                "free_bytes": 8192,
                "full_kv_bytes": 256,
                "recurrent_temporal": {"bytes": 512},
                "recurrent_conv": [{"bytes": 128}],
            },
            "decode_cuda_graph_bs": [],
        }
        record = {
            "completion_counts": {str(i): 16384 for i in range(concurrency)},
            "num_retractions": 0,
            "seconds": 100,
            "output_tokens_per_second": concurrency * 16384 / 100,
            "request_latency_seconds": {},
            "decode_seconds_per_token": {},
            "after": {"internal_states": [state]},
        }
        for name in ("warmup", "repeat-0", "repeat-1", "repeat-2"):
            (folder / f"{name}.json").write_text(json.dumps(record))
        (folder / "complete.json").write_text(
            json.dumps(
                {
                    "job_id": "job",
                    "device": "B300",
                    "source_sha256": "source",
                    "options": options,
                    "stable": True,
                    "mean_output_tokens_per_second": record["output_tokens_per_second"],
                }
            )
        )
        best = options
    (tmp_path / "pilot-complete.json").write_text(
        json.dumps({"attempt": str(attempt), "measured_capacities": [16, 32], "best_measured_stable_options": best})
    )
    return tmp_path


def test_validated_best_capacity_keeps_graph_uncertainty(pilot):
    result = analysis.analyze(pilot)
    assert result["best_concurrency"] == 32
    assert result["capacities"][0]["repeats"][0]["recurrent_bytes"] == 640
    assert result["capacities"][0]["repeats"][0]["reported_captured_graph_sizes"] == []
    assert len(result["artifact_sha256"]) == 10


@pytest.mark.parametrize(
    "filename,key,value,error",
    [
        ("complete", "job_id", "different-job", "different allocations"),
        ("repeat-1", "completion_counts", {"0": 16383}, "completion lengths"),
        ("warmup", "num_retractions", 1, "Retracted"),
        ("repeat-2", "seconds", 0, "Invalid timing"),
        ("complete", "mean_output_tokens_per_second", 1, "disagrees"),
    ],
)
def test_reject_invalid_evidence(pilot, filename, key, value, error):
    path = pilot / "attempt-job/32" / f"{filename}.json"
    record = json.loads(path.read_text())
    record[key] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=error):
        analysis.analyze(pilot)
