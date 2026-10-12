"""Qualification objective and screened-cohort contracts without serving or GPUs."""

import copy
from pathlib import Path

import pytest
import torch

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.distillation import cancellation, qualification, qualification_report


def test_cancellation_preserves_corrections_and_never_reverses_cancelled_boosts():
    student = torch.full((5,), -10.0, requires_grad=True)
    full = student.detach() + torch.tensor([7.2, 5.4, -0.9, 2.0, -2.0])
    answer = student.detach() + torch.tensor([8.0, 4.1, -1.0, -1.0, 3.0])
    pressure, metrics = cancellation.token_pressure(student, full, answer)
    assert torch.allclose(pressure, torch.tensor([0.0, 1.3, -0.9, 2.0, -2.0]), atol=1e-6)
    assert not pressure.requires_grad
    assert metrics["cancelled_positive_sum"] == pytest.approx(11.3)
    assert metrics["retained_negative_sum"] == pytest.approx(2.9)
    assert metrics["zero_tokens"] == 1


def test_native_precomputed_sign_and_pressure_scale_give_expected_gradient():
    old = torch.tensor([-2.0, -2.0, -2.0])
    pressure, _ = cancellation.token_pressure(old, [-1.0, -3.0, -1.0], [-2.0, -2.0, -1.0])
    transported_reverse_kl = -pressure
    beta = 0.5
    advantage = torch.zeros_like(old) - beta * transported_reverse_kl
    current = old.clone().requires_grad_(True)
    loss = -(torch.exp(current - old) * advantage).mean()
    loss.backward()
    assert current.grad.tolist() == pytest.approx([-1 / 6, 1 / 6, 0])


@pytest.mark.parametrize("bad", [[float("nan")], [float("inf")], [0.1], [], [-1.0, -2.0]])
def test_invalid_scores_fail_closed(bad):
    with pytest.raises(ValueError):
        cancellation.token_pressure([-2.0], [-1.0], bad)


def test_optional_clipping_is_explicit_and_measured():
    pressure, metrics = cancellation.token_pressure([-10.0, -1.0], [-1.0, -10.0], [-10.0, -1.0], clip=2.0)
    assert pressure.tolist() == [2.0, -2.0]
    assert metrics["clipped_tokens"] == 2


def row(identity, reward=0):
    return {
        "sample_id": identity,
        "checkpoint": "/h045",
        "gate": {
            "weight_version": 0,
            "initial_responses": [{"reward": reward} for _ in range(8)],
            "failure_kind": "all_truncated",
        },
        "row": {
            "ground_truth": "42",
            "metadata": {
                "prepared_sample_id": identity,
                "query": identity + " question",
                "bread": {"sections": ["reasoning", "42"], "reference_index": 0},
            },
        },
    }


def test_screened_split_is_stable_disjoint_and_rejects_nonzero_rewards():
    rows = [row(str(i)) for i in range(7)]
    a, b = qualification.select_cohort(rows, "/h045", 4, 1)
    assert (a, b) == qualification.select_cohort(list(reversed(rows)), "/h045", 4, 1)
    assert len(a) == 4 and len(b) == 3
    assert not {r["sample_id"] for r in a} & {r["sample_id"] for r in b}
    with pytest.raises(ValueError, match="zero-of-eight"):
        qualification.select_cohort([row("bad", 1), row("ok")], "/h045", 1, 1)
    bad = copy.deepcopy(rows)
    bad[0]["gate"]["weight_version"] = 1
    with pytest.raises(ValueError, match="weight version"):
        qualification.select_cohort(bad, "/h045", 4, 1)


def test_reference_is_only_in_teacher_inputs_and_diagnostic_removes_both_contexts():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return type("Encoding", (), {"ids": list(text.encode())})()

    template = {"input": "USER OLD ASSISTANT <think>", "metadata": {"query": "OLD"}}
    train = qualification.render_row(row("a"), template, Tokenizer(), True)
    diagnostic = qualification.render_row(row("a"), template, Tokenizer(), False)
    assert "reasoning" not in train["input"] and "42" not in train["input"]
    assert "reasoning" in train["metadata"]["opd_teacher_prompt"]
    assert "reasoning\n\n42" not in train["metadata"]["opd_answer_prompt"]
    assert "The final answer is 42." in train["metadata"]["opd_answer_prompt"]
    assert diagnostic["input"] == train["input"]
    assert "opd_teacher_prompt" not in diagnostic["metadata"]
    assert "opd_answer_prompt" not in diagnostic["metadata"]


def test_qualification_recipe_roundtrips_cancellation_settings():
    document = {
        "schema_version": 1,
        "name": "qualification-test",
        "model": {"source": "/models/h045"},
        "output": {"root": "/outputs/q"},
        "launch": {"gpus_per_replica": 5, "auto_resume": False},
        "data": {
            "prompt_data": "/data/train.jsonl",
            "reward_config": "/data/verifiers.json",
            "eval_prompt_data": ["train_diagnostic", "/data/diag.jsonl"],
            "in_sample_eval": ["train_diagnostic"],
        },
        "training": {"filter_zero_std_groups": False},
        "trainer": {"gpus": 2},
        "inference": {"placement_mode": "disaggregated", "gpus": 2, "samples_per_prompt": 2},
        "teacher": {"source": "/models/frozen", "objective": "cancel_answer_positive", "kl_coef": 1.0},
    }
    spec = RunSpec.from_dict(document, config_path=Path("/tmp/run.json"))
    assert spec.teacher["objective"] == "cancel_answer_positive"
    assert spec.compile().miles["opd_kl_coef"] == 1.0
    assert not spec.compile().miles["normalize_advantages"]
    assert RunSpec.from_dict(spec.to_dict()).data["in_sample_eval"] == ["train_diagnostic"]


def test_report_requires_independent_baseline_and_preserves_truncation_split():
    rows = []
    for step in [-1, 7]:
        for i in range(4):
            rows.append(
                dict(
                    rollout_id=step,
                    dataset="train_diagnostic",
                    sample_id=str(i // 2),
                    sample_index=i,
                    reward=int(step == 7 and i == 0),
                    response_length=100,
                    status="TRUNCATED" if i == 3 else "COMPLETED",
                    weight_versions=[[{"version": "0" if step == -1 else "8", "abs_start": 1, "abs_end": 101}]],
                )
            )
    result = qualification_report.summarize(rows, [])
    assert result["changes_from_independent_baseline"][0]["accuracy_change"] == 0.25
    assert result["evaluation"][1]["truncation_rate"] == 0.25
    assert result["evaluation"][1]["finished_accuracy"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="unchanged H045"):
        qualification_report.summarize(rows[4:], [])
    with pytest.raises(ValueError, match="Duplicate"):
        qualification_report.summarize(rows + rows[:1], [])
