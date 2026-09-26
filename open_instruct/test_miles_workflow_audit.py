"""Failure-oriented checks of the bounded retained-workflow auditor."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.miles import audit_workflow as audit


def counters():
    contracts = {}
    for rank in range(2):
        contracts[str(rank)] = [
            {
                "event": "optimizer",
                "rank": rank,
                "step": step,
                "rollout_id": step - 1,
                "optimizer_skipped": False,
                "normalization": {"samples": 64, "world_size": 2},
                "local_microbatches": 32,
                "published_step": step - 1,
                "local_behavior_versions": [max(0, step - 2)],
                "local_policy_objective": 0.01,
                "local_auxiliary_objective": {"router": 0.01},
                "local_pre_optimizer_gradients": {"router": {"local_l2": 0.01, "missing_parameters": 0}},
                "sampled_model_updates": {"router": {"sampled_update_l2": 0.00001}},
                "lr_used": [1e-6],
            }
            for step in range(1, 5)
        ]
    publications = [{"version": 0, "repeated_version": False}] + [
        {"version": step, "repeated_version": repeated} for step in range(1, 5) for repeated in (False, True)
    ]
    stages = [
        {"stage": name, "rollout_id": rollout, "passed": True, "seconds": 1.0}
        for rollout in range(4)
        for name in ("generation_wait", "training", "publication")
    ]
    stages.extend(
        [
            {"stage": "evaluation", "rollout_id": rollout, "details": {"phase": phase}, "passed": True, "seconds": 2.0}
            for rollout, phase in ((0, "initial"), (3, "periodic"))
        ]
    )
    return contracts, publications, stages


def validate(values):
    return audit.validate_counters(*values, updates=4, batch_size=64, world=2)


def test_exact_counter_gate():
    report = validate(counters())
    assert report["publication_count"] == 9
    assert report["cycle_seconds"] == [3.0] * 4
    assert len(report["evaluation_timings"]) == 2


@pytest.mark.parametrize(
    "problem",
    [
        "missing_step",
        "wrong_normalization",
        "nonfinite_gradient",
        "zero_update",
        "missing_publication",
        "bad_repeat",
        "failed_stage",
        "missing_final_eval",
        "stale",
    ],
)
def test_corrupt_counter_contract_fails(problem):
    contracts, publications, stages = counters()
    if problem == "missing_step":
        contracts["1"].pop()
    elif problem == "wrong_normalization":
        contracts["1"][0]["normalization"]["samples"] = 16
    elif problem == "nonfinite_gradient":
        contracts["0"][0]["local_pre_optimizer_gradients"]["router"]["local_l2"] = float("nan")
    elif problem == "zero_update":
        contracts["1"][1]["sampled_model_updates"]["router"]["sampled_update_l2"] = 0
    elif problem == "missing_publication":
        publications.pop()
    elif problem == "bad_repeat":
        publications[-1]["version"] = 3
    elif problem == "failed_stage":
        stages[0]["passed"] = False
    elif problem == "missing_final_eval":
        stages.pop()
    else:
        contracts["1"][3]["local_behavior_versions"] = [0]
    with pytest.raises(ValueError):
        validate((contracts, publications, stages))


def samples():
    prompt_ids = [1, 2, 3]
    prepared = {
        "p": {
            "input": "question",
            "label": "42",
            "metadata": {
                "prepared_sample_id": "p",
                "verifiers": [{"name": "gsm8k", "target": "42"}],
                "run_prompt_token_ids_sha256": hashlib.sha256((json.dumps(prompt_ids) + "\n").encode()).hexdigest(),
            },
        }
    }
    sample = {
        "metadata": copy.deepcopy(prepared["p"]["metadata"]),
        "prompt": "question",
        "label": "42",
        "group_index": 7,
        "weight_versions": [0],
        "response_length": 2,
        "tokens": [*prompt_ids, 4, 5],
        "rollout_log_probs": [-1.0, -2.0],
        "status": "completed",
        "response": "42",
        "reward": 1.0,
    }
    return prepared, [copy.deepcopy(sample) for _ in range(8)]


def check_samples(prepared, rows, **kwargs):
    return audit.audit_samples(
        rows,
        prepared,
        rollout=1,
        version_limit=1,
        multiplicity=8,
        group_count=1,
        consumed=set(),
        groups_seen=set(),
        response_cap=4096,
        verifier=lambda *args: SimpleNamespace(score=1.0),
        **kwargs,
    )


def test_sample_gate_and_eval_without_group_index():
    prepared, rows = samples()
    assert check_samples(prepared, rows)["samples"] == 8
    for row in rows:
        row["group_index"] = None
    assert check_samples(prepared, rows, evaluation=True)["ids"] == ["p"]


@pytest.mark.parametrize("problem", ["mixed_versions", "tokens", "reward", "count", "metadata"])
def test_sample_identity_and_reward_corruption_rejected(problem):
    prepared, rows = samples()
    if problem == "mixed_versions":
        rows[0]["weight_versions"] = [1]
    elif problem == "tokens":
        rows[0]["tokens"][0] = 999
    elif problem == "reward":
        rows[0]["reward"] = 0.0
    elif problem == "count":
        rows.pop()
    else:
        rows[0]["metadata"]["verifiers"][0]["target"] = "43"
    with pytest.raises(ValueError):
        check_samples(prepared, rows)


def test_counters_only_rebases_downloaded_run_and_labels_missing_sources(tmp_path):
    origin = "/weka/oe-training-default/test/run"
    spec = {"output": {"root": origin}, "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 16}]}}
    (tmp_path / "run-spec.json").write_text(json.dumps(spec))
    (tmp_path / "workflow.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            }
        )
    )
    miles = {
        "num_rollout": 4,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "global_batch_size": 64,
        "rollout_batch_size": 8,
        "n_samples_per_prompt": 8,
        "fully_async": True,
        "use_tis": True,
        "use_rollout_logprobs": False,
        "check_weight_update_equal": True,
        "save": origin + "/checkpoints",
    }
    (tmp_path / "resolved-plan.json").write_text(
        json.dumps({"miles": miles, "core": {"diagnostic_interval": 1, "max_policy_lag": 1}})
    )
    prepared = tmp_path / "prepared/data"
    prepared.mkdir(parents=True)
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "contract": {"data": spec["data"]},
                "inputs": {origin + "/prepared/hf/config.json": "absent"},
                "outputs": {},
            }
        )
    )
    metrics = tmp_path / "checkpoints"
    metrics.mkdir()
    contracts, publications, stages = counters()
    files = {f"training_contract_rank{rank}.jsonl": rows for rank, rows in contracts.items()} | {
        "publication.jsonl": publications,
        "driver_timing.jsonl": stages,
    }
    for name, rows in files.items():
        (metrics / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit.audit(tmp_path, counters_only=True)
    assert result["passed"] and result["qualification"] == "counters_only"
    assert result["full_sample_audit"] is False
    assert result["unavailable_prepared_sources"] == [origin + "/prepared/hf/config.json"]


def test_repeated_prompt_in_new_group_is_valid_across_async_epochs():
    prepared, rows = samples()
    report = audit.audit_samples(
        rows,
        prepared,
        rollout=1,
        version_limit=1,
        multiplicity=8,
        group_count=1,
        consumed={"p"},
        groups_seen={6},
        response_cap=4096,
        verifier=lambda *args: SimpleNamespace(score=1.0),
    )
    assert report["ids"] == ["p"] and report["groups"] == [7]
    with pytest.raises(ValueError, match="group identity"):
        audit.audit_samples(
            rows,
            prepared,
            rollout=1,
            version_limit=1,
            multiplicity=8,
            group_count=1,
            consumed={"p"},
            groups_seen={7},
            response_cap=4096,
            verifier=lambda *args: SimpleNamespace(score=1.0),
        )


def test_same_prompt_in_distinct_groups_can_have_different_versions():
    prepared, first = samples()
    second = copy.deepcopy(first)
    for row in second:
        row["group_index"] = 8
        row["weight_versions"] = [1]
    report = audit.audit_samples(
        first + second,
        prepared,
        rollout=1,
        version_limit=1,
        multiplicity=8,
        group_count=2,
        consumed=set(),
        groups_seen=set(),
        response_cap=4096,
        verifier=lambda *args: SimpleNamespace(score=1.0),
    )
    assert report["versions"] == {7: 0, 8: 1}
    assert report["ids"] == ["p"]


def test_packed_counter_gate_checks_schedule_and_token_accounting():
    contracts, publications, stages = counters()
    for rows in contracts.values():
        events = []
        for step in rows:
            step["local_microbatches"] = 8
            step["normalization"]["model_tokens"] = 2048
            events.append(
                {
                    "event": "packing",
                    "step": step["step"],
                    "rollout_id": step["rollout_id"],
                    "samples": 32,
                    "packs": 8,
                    "model_tokens": 1024,
                    "token_budget": 256,
                    "max_pack_tokens": 128,
                }
            )
        rows.extend(events)

    def check():
        return audit.validate_counters(
            contracts, publications, stages, updates=4, batch_size=64, world=2, packing_token_budget=256
        )

    assert check()["publication_count"] == 9
    packed = contracts["1"][-1]
    packed["packs"] = 7
    with pytest.raises(ValueError, match="microbatch count"):
        check()
    contracts["1"][3]["local_microbatches"] = 7
    packed["max_pack_tokens"] = 256
    with pytest.raises(ValueError, match="different microbatch"):
        check()
    contracts["1"][3]["local_microbatches"] = packed["packs"] = 8
    packed["model_tokens"] -= 1
    with pytest.raises(ValueError, match="token counts differ"):
        check()
