"""Pairing must survive completion reordering and distinguish correctness from stopping."""

import json

import pytest
from scripts.miles import compare_opd_evaluations


def row(prompt, reward, length, status="completed"):
    return dict(
        dataset="math",
        prompt=prompt,
        label="42",
        reward=reward,
        response_length=length,
        status=status,
        response="\\boxed{42}",
    )


def write(path, rows):
    path.write_text("".join(json.dumps(value) + "\n" for value in rows))


def test_pairs_prompts_not_completion_order_and_retains_capped_correct_answers(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write(left, [row("a", 0, 100), row("b", 1, 200, "truncated")])
    write(right, [row("b", 1, 50), row("a", 1, 150)])
    result = compare_opd_evaluations.compare(left, right)["datasets"]["math"]
    assert result["accuracy_delta_right_minus_left"] == 0.5
    assert result["left"]["correct_while_truncated"] == 1
    assert result["outcomes"]["right_gains"]["mean_length_delta"] == 50
    assert result["outcomes"]["both_correct"]["left_truncated_right_stopped"] == 1


def test_missing_questions_are_not_silently_dropped(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write(left, [row("a", 1, 10), row("b", 1, 20)])
    write(right, [row("b", 1, 20)])
    with pytest.raises(ValueError, match="exactly the same"):
        compare_opd_evaluations.compare(left, right)
