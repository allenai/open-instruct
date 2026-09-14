"""Sibling timing distinguishes admission skew, execution skew, and missing data."""

import json
from types import SimpleNamespace

from open_instruct.miles import sibling_timing


def samples(count=4):
    return [SimpleNamespace(group_index=7, index=i, metadata={"existing": 1}, response_length=0) for i in range(count)]


def test_admission_skew_is_distinct_from_engine_start_and_finish_skew():
    records = [
        dict(
            group_created_monotonic=10,
            admitted_monotonic=11,
            engine_received_unix=101,
            engine_first_forward_unix=102,
            engine_finished_unix=112,
        ),
        dict(
            group_created_monotonic=10,
            admitted_monotonic=15,
            engine_received_unix=105,
            engine_first_forward_unix=109,
            engine_finished_unix=140,
        ),
    ]
    result = sibling_timing.summarize(records)
    assert result["admission_skew_seconds"] == 4
    assert result["engine_first_forward_skew_seconds"] == 7
    assert result["engine_finished_skew_seconds"] == 28
    assert result["engine_first_forward_first"] == 102
    assert result["engine_first_forward_last"] == 109
    assert result["admission_wait_mean_seconds"] == 3
    assert result["engine_initial_wait_mean_seconds"] == 2.5
    assert result["engine_execution_mean_seconds"] == 20.5


def test_missing_sibling_timestamps_do_not_report_false_zero_skew():
    result = sibling_timing.summarize([{"engine_first_forward_unix": 10}, {}])
    assert result["engine_first_forward_samples"] == 1
    assert "engine_first_forward_skew_seconds" not in result
    assert result["admission_samples"] == 0
    assert "admission_skew_seconds" not in result


def test_invalid_clock_order_is_counted_not_clamped_to_zero():
    result = sibling_timing.summarize([{"engine_received_unix": 20, "engine_first_forward_unix": 10}])
    assert result["engine_initial_wait_invalid_order_samples"] == 1
    assert "engine_initial_wait_mean_seconds" not in result


def test_attempts_are_fresh_and_payload_provenance_is_preserved():
    values = samples()
    old = sibling_timing.start_group(values)
    sibling_timing.admitted(values[0])
    assert "admitted_monotonic" in old[0]
    new = sibling_timing.start_group(values)
    assert old[0]["group_attempt"] != new[0]["group_attempt"]
    assert len({r["group_attempt"] for r in new}) == 1
    assert "admitted_monotonic" not in new[0]
    assert values[0].metadata["existing"] == 1


def test_only_numeric_engine_observations_are_retained_and_partial_groups_are_written(tmp_path):
    values = samples(2)
    records = sibling_timing.start_group(values)
    sibling_timing.admitted(values[0])
    sibling_timing.response_received(
        values[0],
        "rid",
        {
            "request_received_ts": 100,
            "forward_entry_time": 102,
            "prefill_finished_time": 103,
            "request_finished_ts": 110,
            "queue_time": 2,
            "e2e_latency": float("nan"),
            "num_retractions": 3,
            "prompt": "private",
            "output_token_logprobs": [[-1, 3]],
        },
    )
    values[0].response_length = 5
    sibling_timing.finished(values[0], "completed")
    sibling_timing.write_group(SimpleNamespace(save=str(tmp_path)), records, "ReadError")
    row = json.loads(next(tmp_path.glob("sibling_timing_*.jsonl")).read_text())
    assert row["outcome"] == "ReadError" and len(row["samples"]) == 2
    assert row["samples"][0]["request_id"] == "rid"
    assert row["samples"][0]["response_tokens"] == 5
    assert "engine_reported_e2e_latency" not in row["samples"][0]
    assert row["summary"]["engine_first_forward_samples"] == 1
    assert "private" not in json.dumps(row) and "output_token_logprobs" not in json.dumps(row)


def test_artifact_failure_does_not_break_generation(tmp_path, caplog):
    bad = tmp_path / "file"
    bad.write_text("not a directory")
    sibling_timing.write_group(SimpleNamespace(save=str(bad)), sibling_timing.start_group(samples()), "completed")
    assert "Sibling timing observation unavailable" in caplog.text


def test_consumed_metrics_include_later_complete_observations_and_exclude_partial_groups():
    a, b = samples(2), samples(2)
    sibling_timing.start_group(a)
    rows = sibling_timing.start_group(b)
    rows[0]["engine_first_forward_unix"] = 100
    rows[1]["engine_first_forward_unix"] = 104
    metrics = sibling_timing.consumed_metrics(a + b)
    prefix = "rollout/siblings/consumed/"
    assert metrics[prefix + "groups"] == 2
    assert metrics[prefix + "engine_first_forward_skew_seconds/groups"] == 1
    assert metrics[prefix + "engine_first_forward_skew_seconds/mean"] == 4
    metrics = sibling_timing.consumed_metrics(b[:1])
    assert metrics[prefix + "incomplete_groups"] == 1
    assert metrics[prefix + "groups"] == 0
    assert sibling_timing.consumed_metrics([SimpleNamespace()]) == {}
