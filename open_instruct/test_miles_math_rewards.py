"""Math-Verify math scoring, the boxed-answer prompt instruction and math target checks."""

import pytest

from open_instruct.miles.rewards import math_rewards


def score(prediction, label):
    return math_rewards.MathVerifyVerifier()([], prediction, label).score


@pytest.mark.parametrize(
    ("prediction", "label"),
    [
        ("So the answer is \\boxed{\\frac{1}{2}}.", "0.5"),
        ("\\boxed{3, 1, 2}", "1, 2, 3"),
        ("It costs \\boxed{\\$12.50}.", "12.5"),
        ("The choice is \\boxed{\\text{(B)}}", "B"),
        ("So \\boxed{9}.", " N = 9 "),
        ("\\boxed{2\\sqrt{2}}", "\\sqrt{8}"),
        ("\\boxed{7}", ["6", "7"]),
        ("<think>maybe \\boxed{1}</think> So \\boxed{2}.", "2"),
    ],
)
def test_credited(prediction, label):
    assert score(prediction, label) == 1.0


@pytest.mark.parametrize(
    ("prediction", "label"),
    [
        # Strict extraction: no bare-number fallback.
        ("The answer is 42", "42"),
        ("I think it's 41, no, 42", "42"),
        ("Step 1: 42 apples", "42"),
        # The model cannot hedge by boxing several answers.
        ("\\boxed{41} or maybe \\boxed{42}", "42"),
        ("\\boxed{41}", "42"),
        ("", "42"),
    ],
)
def test_not_credited(prediction, label):
    assert score(prediction, label) == 0.0


def test_add_boxed_instruction():
    prompt = "What is 6 * 7?"
    assert math_rewards.add_boxed_instruction(prompt) == f"{prompt}\n\n{math_rewards.MATH_BOXED_INSTRUCTION}"
    for already in ["Put your answer in \\boxed{}.", "End with 'The final answer is X'.", "Give the boxed answer."]:
        assert math_rewards.add_boxed_instruction(already) == already


@pytest.mark.parametrize(
    ("ground_truth", "problems"),
    [
        (["42"], []),
        ("42", []),
        (["-1"], []),
        ([" "], ["empty_target"]),
        ([], ["not_single_target"]),
        (None, ["empty_target"]),
    ],
)
def test_math_row_problems(ground_truth, problems):
    assert math_rewards.math_row_problems({"ground_truth": ground_truth}) == problems


@pytest.mark.parametrize(
    ("first", "second", "equal"),
    [("60", "60^\\circ", True), ("2\\sqrt 2", "2\\sqrt{2}", True), (" 1000 ", "1000", True), ("3", "4", False)],
)
def test_targets_equivalent(first, second, equal):
    assert math_rewards.targets_equivalent(first, second) is equal
