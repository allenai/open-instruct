"""The lightweight extractor preserves the existing training utility API."""

import pytest

from open_instruct import answer_utils, utils


@pytest.mark.parametrize(
    "prediction,expected",
    [
        ("reasoning <answer> 42\n </answer> ignored", "42"),
        ("reasoning </think> 17 ", "17"),
        ("<|assistant|> 9 ", "9"),
        (" unchanged ", " unchanged "),
        ("</think> outside <answer>inside</answer>", "inside"),
        ("<answer></answer>", ""),
    ],
)
def test_existing_api_and_extraction(prediction, expected):
    assert utils.extract_final_answer is answer_utils.extract_final_answer
    assert answer_utils.extract_final_answer(prediction) == expected
