"""Reject partial/incorrect evidence and preserve question-level uncertainty."""

import copy
import json

import pytest
from scripts.miles import analyze_quantization_study as analysis


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def aime_fixture(tmp_path):
    data = [{"id": i, "year": 2024, "answer": "004"} for i in range(30)]
    rows = []
    for i in range(30):
        outputs = [
            {
                "text": "word " * 80,
                "extracted_answer": "4" if j < i else "",
                "sample_metrics": {analysis.SCORER: {analysis.SCORER: int(j < i)}},
                "completion_tokens": 16384 if j == 31 else 100,
                "finish_reason": "length" if j == 31 else "stop",
            }
            for j in range(32)
        ]
        rows.append({"native_id": i, "label": "4", "model_output": outputs})
    snapshot, predictions, metrics = [tmp_path / name for name in ("source.json", "predictions.jsonl", "metrics.json")]
    snapshot.write_text(json.dumps(data))
    write_rows(predictions, rows)
    metrics.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": "aime_2024:pass_at_32",
                        "num_instances": 30,
                        "completion_tokens_total": 30 * (3100 + 16384),
                        "duration_seconds": 100,
                        "metrics": {
                            "pass_at_1": {analysis.SCORER: 14.5 / 32},
                            "pass_at_32": {analysis.SCORER: 29 / 30},
                        },
                    }
                ]
            }
        )
    )
    return snapshot, predictions, metrics, rows


def test_aime_question_bootstrap_and_diagnostics(tmp_path):
    snapshot, predictions, metrics, _ = aime_fixture(tmp_path)
    result = analysis.aime_year(predictions, snapshot, metrics, 2024, analysis.digest(snapshot))
    assert result["pass_at_1_estimate"]["mean"] == pytest.approx(14.5 / 32)
    assert result["observed_pass_at_32"]["mean"] == pytest.approx(29 / 30)
    assert result["truncation_rate"] == 1 / 32
    assert result["repetition_rate"] == 1
    assert result["tokens_per_task_second"] == 30 * (3100 + 16384) / 100
    assert result["tokens_per_attributed_inference_second"] is None
    assert analysis.question_interval([0, 1], repeats=1000) == [0, 1]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_draw",
        "duplicate_question",
        "provider_abort",
        "scoring_failure",
        "wrong_hash",
        "wrong_year",
        "wrong_tokens",
    ],
)
def test_aime_rejects_incomplete_or_wrong_evidence(tmp_path, failure):
    snapshot, predictions, metrics, rows = aime_fixture(tmp_path)
    if failure == "missing_draw":
        rows[0]["model_output"].pop()
    elif failure == "duplicate_question":
        rows[-1] = copy.deepcopy(rows[0])
    elif failure == "provider_abort":
        rows[0]["model_output"][0]["finish_reason"] = "abort"
    elif failure == "scoring_failure":
        rows[0]["scoring_error"] = "timeout"
    elif failure == "wrong_tokens":
        rows[0]["model_output"][0]["completion_tokens"] += 1
    write_rows(predictions, rows)
    with pytest.raises(ValueError):
        analysis.aime_year(
            predictions,
            snapshot,
            metrics,
            2025 if failure == "wrong_year" else 2024,
            "bad" if failure == "wrong_hash" else analysis.digest(snapshot),
        )


def speed_rows(precision):
    return [
        {
            **dict(zip(analysis.KEYS, key)),
            "precision": precision,
            "seconds": 2,
            "context_length": 18434,
            "completion_counts": {str(i): key[1] for i in range(key[2])},
            "provenance": {"timing": "warmup excluded"},
        }
        for key in sorted(analysis.GRID)
    ]


def test_speed_pairing_preserves_legacy_provenance(tmp_path):
    rows = speed_rows("bf16")
    for row in rows[:45]:
        del row["context_length"], row["completion_counts"], row["provenance"]
    original, validation, measurements = [tmp_path / name for name in ("original.jsonl", "repair.json", "all.jsonl")]
    write_rows(original, rows[:45])
    validation.write_text(json.dumps({"original_measurements_sha256": analysis.digest(original)}))
    write_rows(measurements, rows)
    bf16 = analysis.speed_cases(measurements, "bf16", legacy_original=original, legacy_validation=validation)
    write_rows(measurements, speed_rows("nvfp4"))
    nvfp4 = analysis.speed_cases(measurements, "nvfp4")
    report = analysis.paired_speed({"bf16": bf16, "nvfp4": nvfp4})
    pairs = report["comparisons"]["nvfp4"]["pairs"]
    assert sum(not p["same_context"] for p in pairs) == 45
    assert all(p["speed_ratio_vs_bf16"] == 1 for p in pairs)
    assert len(report["comparisons"]["nvfp4"]["workloads"]) == 18


@pytest.mark.parametrize("failure", ["short_grid", "duplicate", "truncated", "context", "duration"])
def test_speed_rejects_invalid_grid(tmp_path, failure):
    rows = speed_rows("nvfp4")
    if failure == "short_grid":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = rows[0]
    elif failure == "truncated":
        rows[0]["completion_counts"]["0"] -= 1
    elif failure == "context":
        rows[0]["context_length"] = 18432
    else:
        rows[0]["seconds"] = float("nan")
    path = tmp_path / "measurements.jsonl"
    write_rows(path, rows)
    with pytest.raises(ValueError):
        analysis.speed_cases(path, "nvfp4")


def warmed_fixture(tmp_path):
    rows = speed_rows("bf16")
    warmups = {}
    provenance = {"timing_protocol": "exact-workload-warmup-v1", "context_length": 18434, "process_id": "job-pid-time"}
    for i, row in enumerate(rows):
        workload = tuple(row[k] for k in analysis.KEYS[:3])
        warmup_id = str(workload)
        start = 10 * (i // 3) + 1
        row.update(
            provenance=provenance.copy(),
            warmup_id=warmup_id,
            wall_start=start + 3 + 2 * (i % 3),
            wall_end=start + 5 + 2 * (i % 3),
            monotonic_start=start + 3 + 2 * (i % 3),
            monotonic_end=start + 5 + 2 * (i % 3),
        )
        warmups.setdefault(
            warmup_id,
            {
                **copy.deepcopy(row),
                "id": warmup_id,
                "wall_start": start,
                "wall_end": start + 2,
                "monotonic_start": start,
                "monotonic_end": start + 2,
            },
        )
    path = tmp_path / "measurements.jsonl"
    write_rows(path, rows)
    write_rows(tmp_path / "warmups.jsonl", list(warmups.values()))
    (tmp_path / "complete.json").write_text(json.dumps({"cases": 54, "options": {"context_length": 18434}}))
    return path, rows, list(warmups.values())


def test_exact_workload_warmup_receipts(tmp_path):
    path, _, _ = warmed_fixture(tmp_path)
    cases, receipt = analysis.warmed_speed_cases(path, "bf16")
    assert len(cases) == 54
    assert receipt["referenced_warmups"] == 18
    assert len(receipt["sources"]) == 3


@pytest.mark.parametrize(
    "failure",
    [
        "overlap",
        "wrong_shape",
        "short_warmup",
        "wrong_process",
        "bad_clock",
        "nan",
        "missing",
        "duplicate",
        "wrong_request_ids",
    ],
)
def test_warmed_gate_rejects_invalid_receipts(tmp_path, failure):
    path, rows, warmups = warmed_fixture(tmp_path)
    if failure == "overlap":
        rows[0]["monotonic_start"] = 2
        rows[0]["monotonic_end"] = 4
    elif failure == "wrong_shape":
        warmups[0]["prompt_tokens"] = 2048
    elif failure == "short_warmup":
        warmups[0]["completion_counts"] = {"0": 1}
    elif failure == "wrong_process":
        warmups[0]["provenance"]["process_id"] = "different"
    elif failure == "bad_clock":
        rows[0]["monotonic_end"] += 1
    elif failure == "nan":
        warmups[0]["wall_start"] = float("nan")
    elif failure == "missing":
        warmups.pop(0)
    elif failure == "duplicate":
        warmups.append(warmups[0])
    else:
        rows[0]["completion_counts"] = {"999": 256}
    write_rows(path, rows)
    write_rows(tmp_path / "warmups.jsonl", warmups)
    with pytest.raises((ValueError, KeyError)):
        analysis.warmed_speed_cases(path, "bf16")
