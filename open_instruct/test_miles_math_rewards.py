"""Math answer extraction, the boxed-answer prompt instruction and math target checks."""

import pytest

from open_instruct import ground_truth_utils, math_utils
from open_instruct.miles.rewards import math_rewards


def score(prediction, label):
    return ground_truth_utils.MathVerifier()([], prediction, label).score


@pytest.mark.parametrize(
    ("prediction", "label"),
    [
        ("So the answer is $$42$$", "42"),
        ("We get $$x = 42$$", "42"),
        ("Thus $$\\frac{1}{2}$$.", "\\frac{1}{2}"),
        ("We get \\[ x = 42 \\]", "42"),
        ("$a$ and $$b$$ then $42$", "42"),
        ("The final answer is 42.", "42"),
        ("the final answer is: 42", "42"),
        ("**Final Answer:** The final answer is **42**.", "42"),
        ("**Final Answer**: 42", "42"),
        ("**Final Answer:**\n\n42", "42"),
        ("The final answer is 3.5.", "3.5"),
        ("Final Answer: The final answer is $42$. I hope it is correct.", "42"),
        ("\\fbox{7}", "7"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("\\boxed 5 was wrong, so \\boxed{42}", "42"),
    ],
)
def test_accepted_formats(prediction, label):
    assert score(prediction, label) == 1.0


@pytest.mark.parametrize(
    ("prediction", "label"),
    [
        # Only the last box counts.
        ("\\boxed{1} and then \\boxed{2}", "1"),
        # A boxed answer takes precedence over a later "final answer" sentence.
        ("\\boxed{41}\n\nThe final answer is 42.", "42"),
        # No bare-number fallback.
        ("I think it's 41, no, 42", "42"),
        ("Step 1: 42 apples", "42"),
        ("The final answer is not determined yet; we have 42 apples", "42"),
        ("Price is \\$5 and \\$42", "42"),
    ],
)
def test_rejected_formats(prediction, label):
    assert score(prediction, label) == 0.0


def test_last_math_span_skips_empty_and_reads_display_math():
    assert math_utils.last_math_span("$$42$$") == "42"
    assert math_utils.last_math_span("$1$ then \\(2\\)") == "2"
    assert math_utils.last_math_span("no math here") is None


def test_final_answer_phrase_has_no_fallback():
    assert math_utils.get_final_answer_phrase("The answer is 42") is None
    assert math_utils.get_final_answer_phrase("The final answer is $42$. I hope it is correct.") == "$42$"


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
        (["-1"], ["placeholder_target"]),
        ([" "], ["empty_target"]),
        ([], ["not_single_target"]),
        (None, ["empty_target"]),
    ],
)
def test_math_row_problems(ground_truth, problems):
    assert math_rewards.math_row_problems({"ground_truth": ground_truth}) == problems
