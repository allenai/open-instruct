"""Run with the pinned olmo-eval Python environment; no GPU or network required."""

import pytest

development_eval = pytest.importorskip(
    "development_eval", reason="Requires the pinned olmo-eval environment", exc_type=ModuleNotFoundError
)


@pytest.mark.parametrize("label", ["42", ["42"], ["41", "42"]])
@pytest.mark.parametrize("verifier", ["math", "gsm8k"])
def test_real_gold_shapes_and_completion_gates(label, verifier):
    item = dict(prompt="What is six times seven?", label=label, verifier=verifier)
    row = dict(text="</think> \\boxed{42}", finish={"type": "stop"})
    assert development_eval.score_one(row, item)["completed_final_score"] == 1
    assert development_eval.score_one(dict(row, text="</think> \\boxed{43}"), item)["score"] == 0
    capped = development_eval.score_one(dict(row, finish={"type": "length"}), item)
    assert capped["score"] == 1 and capped["completed_final_score"] == 0
    unfinished = development_eval.score_one(dict(row, text="\\boxed{42}"), item)
    assert unfinished["score"] == 1 and unfinished["completed_final_score"] == 0


@pytest.mark.parametrize("label", [[], [["42"]], None, 42, ["42", None], "", [""]])
def test_invalid_gold_fails_loudly(label):
    with pytest.raises(ValueError, match="gold"):
        development_eval.gold_answers(label)
