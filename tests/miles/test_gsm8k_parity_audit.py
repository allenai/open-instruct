"""Offline audit fixtures exercise score, identity, version, and pairing contracts."""

import copy
import hashlib
import json
import sys

import pytest
import torch
from scripts.miles import analyze_gsm8k_parity as audit
from scripts.miles import launch_gsm8k_parity


def _row(key):
    return {
        "id": key,
        "input": f"Question {key}",
        "label": "-2",
        "metadata": {"prepared_sample_id": key, "verifiers": [{"name": "gsm8k", "target": "-2", "weight": 1.0}]},
    }


def _sample(row, *, correct=True, version=0):
    return {
        "prompt": row["input"],
        "label": row["label"],
        "metadata": copy.deepcopy(row["metadata"]),
        "response": "PRIVATE RESPONSE: final answer -2" if correct else "PRIVATE RESPONSE: final answer 3",
        "response_length": 2,
        "tokens": [10, 11, 12],
        "rollout_log_probs": [-0.1, -0.2],
        "reward": float(correct),
        "status": "completed",
        "weight_versions": [str(version)],
    }


def _write_dump(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rollout_id": int(path.stem.removeprefix("eval_")), "samples": samples, "metadata": {}}, path)


@pytest.mark.parametrize(
    "fault", ["reward", "version", "membership", "multiplicity", "target", "status", "logprob", "tokens"]
)
def test_rejects_corrupt_stored_evidence(fault, tmp_path):
    row = _row("heldout-id")
    sample = _sample(row)
    samples = [sample]
    if fault == "reward":
        sample["reward"] = 0.0
    elif fault == "version":
        sample["weight_versions"] = ["1"]
    elif fault == "membership":
        sample["metadata"]["prepared_sample_id"] = "unknown-id"
    elif fault == "multiplicity":
        samples.append(copy.deepcopy(sample))
    elif fault == "target":
        sample["label"] = "3"
        sample["metadata"]["verifiers"][0]["target"] = "3"
    elif fault == "status":
        sample["status"] = "aborted"
    elif fault == "logprob":
        sample["rollout_log_probs"][0] = float("nan")
    elif fault == "tokens":
        sample["tokens"][0] = 99
    path = tmp_path / "eval_0.pt"
    _write_dump(path, samples)
    token_proof = hashlib.sha256((json.dumps([10], indent=2) + "\n").encode()).hexdigest()
    result = audit.audit_dump(
        path, [row], version=0, multiplicity=1, token_proofs={"heldout-id": {"token_ids_sha256": token_proof}}
    )
    assert not result["valid"]
    assert result["errors"]
    assert "PRIVATE RESPONSE" not in json.dumps(result)
    if fault == "target":
        # The direct verifier uses immutable preparation rather than the corrupted target.
        assert result["samples"][0]["correct"] == 1


def _campaign(root):
    train = [_row(f"train-{i}") for i in range(400)]
    heldout = [_row(f"test-{i}") for i in range(2)]
    proofs = {}
    token_hash = hashlib.sha256((json.dumps([10], indent=2) + "\n").encode()).hexdigest()
    for name, rows in (("train", train), ("eval", heldout)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        proofs[name] = {
            "rows": [{"prepared_sample_id": audit.identity(row), "token_ids_sha256": token_hash} for row in rows]
        }
    (root / "preparation.json").write_text(
        json.dumps(
            {
                "files": {f"{name}.jsonl": audit.digest(root / f"{name}.jsonl") for name in ("train", "eval")},
                "partitions": proofs,
            }
        )
    )
    for backend, offset in (("core", 0), ("megatron", 1)):
        folder = root / backend / ("rollouts" if backend == "core" else "rollout_data")
        for index in range(100):
            samples = [
                _sample(row, correct=i % 2 == 0, version=index + offset)
                for row in train[index * 4 : (index + 1) * 4]
                for i in range(4)
            ]
            _write_dump(folder / f"{index}.pt", samples)
        for step in audit.EVAL_STEPS:
            # Deliberately reverse sample order between arms: pairing must use IDs.
            samples = [
                _sample(row, correct=(i == 0 if step == 0 else backend == "core" or i == 1), version=step + offset)
                for i, row in enumerate(heldout)
            ]
            if backend == "megatron":
                samples.reverse()
            _write_dump(folder / f"eval_{step - 1 if step else 0}.pt", samples)
    publication = root / "core/metrics/publication.jsonl"
    publication.parent.mkdir()
    publication.write_text("".join(json.dumps({"version": step, "total_seconds": 0.5}) + "\n" for step in range(101)))


def test_full_100_update_audits_and_id_paired_learning_curves(tmp_path):
    _campaign(tmp_path)
    core, megatron = [audit.audit(tmp_path, backend) for backend in ("core", "megatron")]
    assert core["valid"] and megatron["valid"]
    assert [row["completed_steps"] for row in core["evaluation"]] == [0, 20, 40, 60, 80, 100]
    assert core["evaluation"][-1]["policy_version"] == 100
    assert megatron["evaluation"][-1]["policy_version"] == 101
    assert core["training_summary"]["accuracy"] == 0.5
    assert core["training_summary"]["samples"] == 1600
    assert core["training_summary"]["mixed_reward_groups"] == 400
    assert core["publication"]["total_seconds"] == 50.5
    compared = audit.compare(core, megatron)
    assert compared["valid"]
    assert compared["learning_curves"][0]["core_minus_megatron_accuracy"] == 0
    final = compared["learning_curves"][-1]
    assert final["core_minus_megatron_accuracy"] == 0.5
    assert (final["both_correct"], final["core_only_correct"], final["megatron_only_correct"]) == (1, 1, 0)
    assert final["pairs"] == [
        {"id": "test-0", "core_correct": 1, "megatron_correct": 0},
        {"id": "test-1", "core_correct": 1, "megatron_correct": 1},
    ]
    assert "PRIVATE RESPONSE" not in json.dumps(core) + json.dumps(compared)
    invalid = copy.deepcopy(megatron)
    invalid["evaluation"][-1]["samples"].append(invalid["evaluation"][-1]["samples"][0])
    assert not audit.compare(core, invalid)["valid"]
    invalid = copy.deepcopy(megatron)
    invalid["prepared_sha256"]["train"] = "changed"
    assert not audit.compare(core, invalid)["valid"]
    (tmp_path / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Preparation artifact changed"):
        audit.audit(tmp_path, "core")


def test_truncated_correct_answer_remains_correct_and_is_counted_separately(tmp_path):
    row = _row("truncated")
    sample = _sample(row)
    sample["status"] = "truncated"
    path = tmp_path / "eval_0.pt"
    _write_dump(path, [sample])
    result = audit.audit_dump(path, [row], version=0, multiplicity=1, response_cap=2)
    assert result["valid"]
    assert result["summary"]["accuracy"] == 1
    assert result["summary"]["truncation_rate"] == 1
    assert result["summary"]["at_response_cap"] == 1


def test_timing_parser_separates_setup_eval_and_warm_phases(tmp_path):
    path = tmp_path / "core.log"
    path.write_text(
        "\n".join(
            [
                "[2026-09-10 00:00:00.000] setup started",
                "[2026-09-10 00:01:00.000] - eval 0: {'perf/rollout_time': 2000.0}",
                r"\u001b[36m(RolloutManager pid=1)\u001b[0m [2026-09-10 00:01:01.000 rollout_manager] metrics.py:89 - perf 0: {'perf/rollout_time': 100.0}",
                "INFO:metrics:perf 5: {'perf/rollout_time': 10.0, 'unrelated': tensor(0.5)}",
                "INFO:metrics:perf 5: {'perf/rollout_time': 10.0}",
                'Core optimizer step 1: {"train/step_seconds": 50.0}',
                "Core optimizer step 6: {'train/step_seconds': 3.0}",
                'Core weight publication: {"version": 0, "total_seconds": 12.0}',
                'Core weight publication: {"version": 6, "total_seconds": 0.5}',
                'Core weight publication: {"version": 6, "total_seconds": 0.8, "repeated_version": true}',
                'GSM8K_PARITY_CORE_COMPLETED {"elapsed_seconds": 600.0}',
                "[2026-09-10 00:10:00.000] finished",
            ]
        )
    )
    result = audit.parse_timing_log(path)
    assert not result["warnings"]
    assert result["phases"]["generation"]["all"]["count"] == 2
    assert result["phases"]["generation"]["warm"]["mean_seconds"] == 10
    assert result["phases"]["training"]["warm"]["mean_seconds"] == 3
    assert result["phases"]["publication"]["warm"]["mean_seconds"] == 0.5
    assert result["run_elapsed_seconds"] == 600
    assert result["observed_log_span_seconds"] == 600


def test_baseline_perf_logs_merge_fields_and_drop_conflicting_points(tmp_path):
    path = tmp_path / "baseline.log"
    path.write_text(
        "\n".join(
            [
                "perf 5: {'perf/rollout_time': 10.0}",
                "perf 5: {'perf/actor_train_time': 3.0, 'perf/update_weights_time': 0.7, 'perf/log_probs_time': 1.0}",
                "perf 6: {'perf/actor_train_time': 4.0}",
                "perf 6: {'perf/actor_train_time': 5.0}",
                "Timer update_weights end (elapsed: 0.6s)",
                "Timer eval_rollout end (elapsed: 100.0s)",
            ]
        )
    )
    result = audit.parse_timing_log(path)
    assert result["phases"]["training"]["warm"]["mean_seconds"] == 3
    assert result["phases"]["publication"]["warm"]["mean_seconds"] == 0.7
    assert result["phases"]["generation"]["warm"]["mean_seconds"] == 10
    assert result["phases"]["scoring"]["warm"]["mean_seconds"] == 1
    assert result["warnings"] == ["line 4: conflicting training duration at index 6"]
    assert result["rounded_unindexed_timers"]["eval_rollout"]["sum_seconds"] == 100
    assert result["run_elapsed_seconds"] is None
    assert result["observed_log_span_seconds"] is None


def test_metric_parser_never_executes_python_calls(tmp_path):
    sentinel = tmp_path / "never-created"
    result = audit._metric_dict(
        "{'perf/rollout_time': 1.5, 'other': __import__('pathlib').Path(" + repr(str(sentinel)) + ").touch()}"
    )
    assert result == {"perf/rollout_time": 1.5}
    assert not sentinel.exists()


def test_export_comparison_plot(tmp_path):
    _campaign(tmp_path)
    result = audit.compare(audit.audit(tmp_path, "core"), audit.audit(tmp_path, "megatron"))
    assert result["accuracy_gain_0_to_100"] == {"core": 0.5, "megatron": 0.0}
    result["timing"] = {
        backend: {
            "phases": {
                phase: {"points": [{"index": 5, "seconds": value}]}
                for phase, value in (("generation", 8), ("collection_boundary_cycle", 14))
            }
        }
        for backend in ("core", "megatron")
    }
    path = tmp_path / "comparison.png"
    audit.plot_comparison(result, path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _generation_line(index, seconds, *, minute=0, pid=7):
    return (
        f"2026-09-10T09:00:00Z (RolloutManager pid={pid}) "
        f"[2026-09-10 00:{minute:02d}:{seconds:02d}.000 rollout_manager] metrics.py:89 - "
        f"perf {index}: {{'perf/rollout_time': 3.0}}"
    )


def test_collection_cycles_use_native_clock_and_exclude_eval_and_missing_boundaries(tmp_path):
    path = tmp_path / "boundaries.log"
    path.write_text(
        "\n".join(
            [
                _generation_line(4, 0),
                _generation_line(5, 10),
                _generation_line(6, 22),
                # Duplicate capture is harmless; no interval is fabricated for missing7..18.
                _generation_line(6, 22),
                _generation_line(19, 0, minute=1),
                _generation_line(20, 0, minute=5),
                _generation_line(21, 15, minute=5),
            ]
        )
    )
    result = audit.parse_timing_log(path)
    cycles = result["phases"]["collection_boundary_cycle"]
    assert result["collection_boundaries_observed"] == 6
    assert result["excluded_eval_intervals"] == [[19, 20]]
    assert [(point["index"], point["next_rollout"], point["seconds"]) for point in cycles["points"]] == [
        (4, 5, 10),
        (5, 6, 12),
        (20, 21, 15),
    ]
    assert cycles["warm"]["count"] == 2
    assert cycles["warm"]["mean_seconds"] == 13.5
    assert not result["warnings"]


def test_ambiguous_or_restarted_boundaries_never_form_cycles(tmp_path):
    path = tmp_path / "restart.log"
    path.write_text(
        "\n".join(
            [
                _generation_line(5, 0),
                _generation_line(6, 10),
                _generation_line(6, 11),
                _generation_line(7, 20),
                _generation_line(8, 30, pid=8),
            ]
        )
    )
    result = audit.parse_timing_log(path)
    assert result["phases"]["collection_boundary_cycle"]["all"]["count"] == 0
    assert any("conflicting collection boundary" in error for error in result["warnings"])
    assert any("changed manager" in error for error in result["warnings"])


def test_cadence_comparison_uses_common_indices_and_omits_diagnostic_training_scope():
    timing = {
        backend: {
            "warmup_updates_excluded": 5,
            "phases": {
                "generation": {"points": [{"index": index, "seconds": seconds} for index, seconds in points]},
                "training": {"points": [{"index": 5, "seconds": 999}]},
            },
        }
        for backend, points in (("core", [(4, 999), (5, 10), (6, 30)]), ("megatron", [(6, 60), (7, 70)]))
    }
    compared = audit.comparable_timing(timing)
    assert set(compared) == {"generation", "collection_boundary_cycle"}
    assert compared["generation"]["indices"] == [6]
    assert compared["generation"]["core"]["mean_seconds"] == 30
    assert compared["generation"]["megatron"]["mean_seconds"] == 60
    assert compared["collection_boundary_cycle"]["core"]["count"] == 0


def test_retry_directory_is_explicit_and_failed_attempt_stays_untouched(tmp_path, monkeypatch):
    _campaign(tmp_path)
    (tmp_path / "megatron").rename(tmp_path / "megatron-r2")
    (tmp_path / "megatron").mkdir()
    failure = tmp_path / "megatron/failure.json"
    failure.write_text('{"completed_updates": 0, "failure": "missing router API"}\n')
    before = failure.read_bytes()
    assert not audit.audit(tmp_path, "megatron")["valid"]
    retry = audit.audit(tmp_path, "megatron", megatron_directory="megatron-r2")
    assert retry["valid"]
    assert retry["artifact_directory"] == "megatron-r2"
    core = audit.audit(tmp_path, "core")
    (tmp_path / "core/audit.json").write_text(json.dumps(core))
    (tmp_path / "megatron-r2/audit.json").write_text(json.dumps(retry))
    monkeypatch.setattr(sys, "argv", ["audit", "compare", str(tmp_path), "--megatron-directory", "megatron-r2"])
    audit.main()
    compared = json.loads((tmp_path / "comparison.json").read_text())
    assert compared["valid"]
    assert compared["artifact_directories"] == {"core": "core", "megatron": "megatron-r2"}
    assert failure.read_bytes() == before
    assert not (tmp_path / "megatron/audit.json").exists()


def test_cpu_audit_launcher_passes_explicit_retry_directory():
    task = launch_gsm8k_parity.specification("test-image", "audit", megatron_directory="megatron-r2")["tasks"][0]
    script = task["arguments"][0]
    assert "export MEGATRON_DIRECTORY=megatron-r2" in script
    assert script.count('--megatron-directory "$MEGATRON_DIRECTORY"') == 2
    assert 'cp "$RUN_ROOT/$MEGATRON_DIRECTORY/audit.json"' in script
    assert task["resources"]["gpuCount"] == 0
    assert task["constraints"]["cluster"] == ["ai2/saturn"]
    original = launch_gsm8k_parity.specification("test-image", "audit")["tasks"][0]
    assert "export MEGATRON_DIRECTORY=megatron\n" in original["arguments"][0]
    with pytest.raises(ValueError, match="audit artifacts only"):
        launch_gsm8k_parity.specification("test-image", "core", megatron_directory="megatron-r2")


@pytest.mark.parametrize("directory", ["", "..", "../megatron-r2", "/tmp/other"])
def test_retry_directory_must_stay_beneath_campaign_root(directory):
    with pytest.raises(ValueError, match="one directory name"):
        audit.arm_directory("megatron", directory)
    with pytest.raises(ValueError, match="one directory name"):
        launch_gsm8k_parity.specification("test-image", "audit", megatron_directory=directory)
