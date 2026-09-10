"""Offline audit fixtures exercise score, identity, version, and pairing contracts."""

import copy
import hashlib
import json

import pytest
import torch
from scripts.miles import analyze_gsm8k_parity as audit


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
                phase: {"warm": {"mean_seconds": value}}
                for phase, value in (("generation", 8), ("training", 4), ("publication", 0.5))
            }
        }
        for backend in ("core", "megatron")
    }
    path = tmp_path / "comparison.png"
    audit.plot_comparison(result, path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
