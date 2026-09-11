"""Meaningful corruption checks for the read-only response-length audit."""

import copy

import pytest
from scripts.miles import analyze_response_lengths as analysis
from scripts.miles import response_length_transitions as conditional


def fixture():
    prepared = {"input": "Question", "label": "12", "metadata": {"prepared_sample_id": "question-1"}}
    sample = {
        **copy.deepcopy(prepared),
        "prompt": prepared["input"],
        "weight_versions": [3],
        "response_length": 2,
        "tokens": [1, 2, 3, 4],
        "rollout_log_probs": [-0.5, -0.8],
        "status": "completed",
        "response": "Answer: 12",
        "reward": 1.0,
    }
    return sample, prepared, {"token_ids_sha256": analysis.token_digest([1, 2])}


def test_valid_sample():
    sample, prepared, proof = fixture()
    row = analysis.validate_sample(sample, prepared, proof, version=3, rollout=2)
    assert row["reward"] == 1 and row["response_tokens"] == 2


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("weight_versions", [2], "Policy version"),
        ("tokens", [2, 1, 3, 4], "Prompt token proof"),
        ("response_length", 0, "response length"),
        ("reward", 0.0, "reward mismatch"),
        ("rollout_log_probs", [float("nan"), -0.8], "log probabilities"),
        ("status", "aborted", "Failed sample"),
    ],
)
def test_corrupted_sample_is_rejected(field, value, error):
    sample, prepared, proof = fixture()
    sample[field] = value
    with pytest.raises(AssertionError, match=error):
        analysis.validate_sample(sample, prepared, proof, version=3, rollout=2)


def test_quantile_and_group_centering():
    rows = [
        {"id": "all-correct", "rollout": 0, "reward": 1.0, "response_tokens": n, "at_cap": False, "truncated": False}
        for n in (10, 20, 30, 40)
    ]
    rows += [
        {
            "id": "mixed",
            "rollout": 0,
            "reward": float(n < 3),
            "response_tokens": 100,
            "at_cap": True,
            "truncated": True,
        }
        for n in range(4)
    ]
    result = analysis.summarize(rows, training=True)
    assert result["length"] == {"count": 8, "mean": 62.5, "median": 70.0, "p90": 100.0}
    assert result["zero_policy_advantage_sample_fraction"] == 0.5
    assert result["mean_reward"] == 7 / 8
    assert result["wrong_length"]["count"] == 1
    assert analysis.percentile([1, 2, 3, 4], 0.9) == pytest.approx(3.7)


def test_paired_transition_keeps_question_identity_and_excludes_caps():
    before = [
        {"id": "a", "reward": 1, "at_cap": False, "response_tokens": 100},
        {"id": "b", "reward": 1, "at_cap": True, "response_tokens": 4096},
    ]
    after = [
        {"id": "b", "reward": 0, "at_cap": False, "response_tokens": 50},
        {"id": "a", "reward": 1, "at_cap": False, "response_tokens": 80},
    ]
    rows = conditional.transitions(before, after)
    assert rows["correct_both_and_uncapped_both"]["count"] == 1
    assert rows["correct_both_and_uncapped_both"]["paired_change"]["mean"] == -20
    assert rows["correct_to_wrong"]["pairs"][0]["id"] == "b"
    with pytest.raises(AssertionError):
        conditional.transitions(before, [after[0], after[0]])
