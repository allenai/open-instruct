"""Regressions for instruction/checker mismatches in the standard IF reward path."""

from unittest.mock import patch

import pytest

from open_instruct import ground_truth_utils
from open_instruct.IFEvalG import instructions


@pytest.mark.parametrize(
    "start,end,answer,expected",
    [
        (1, 2, "bC", True),
        (1, 2, "b", False),
        (0, 3, "AbCd", True),
        (0, 0, "A", True),
        (3, 3, "d", True),
        (1, 2, "Bc", True),
        (1, 2, " bC ", True),
        (1, 2, "bCd", False),
    ],
)
def test_copy_inclusive_explicit_bounds(start, end, answer, expected):
    checker = instructions.CopySpanIdxChecker("new:copy_span_idx")
    with patch.object(instructions.random, "randint", side_effect=AssertionError("Explicit bounds must not draw")):
        checker.build_description(prompt_to_repeat="AbCd", n_start=start, n_end=end)
        assert checker.check_following(answer) is expected
        assert checker.get_instruction_args()["n_start"] == start
        assert checker.get_instruction_args()["n_end"] == end


@pytest.mark.parametrize("kwargs", [{}, {"n_start": 0}, {"n_end": 0}])
def test_single_character_copy_defaults(kwargs):
    checker = instructions.CopySpanIdxChecker("new:copy_span_idx")
    checker.build_description(prompt_to_repeat="A", **kwargs)
    assert checker.check_following("A")


@pytest.mark.parametrize(
    "keyword,answer,expected",
    [
        ("computer", "computer", True),
        ("computer", "computer computer", True),
        ("computer", "COMPUTER", True),
        ("computer", "desk", False),
        ("a.b", "axb", False),
        ("a.b", "a.b a.b", True),
        ("[", "[", True),
    ],
)
def test_keyword_inclusion_is_literal_not_exact_frequency(keyword, answer, expected):
    checker = instructions.KeywordAtLeastOnceChecker("keywords:at_least_once")
    checker.build_description(keyword=keyword)
    assert checker.check_following(answer) is expected


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("", False),
        ("hello", False),
        ("<<hello world>> stray", False),
        ("<<hello>> <<world>>", False),
        ("<<one two three>>", False),
        ("<<one two>> <<three>>", False),
        ("<<hello world>>", True),
        (" <<hello world>>\n<<again today>> ", True),
        ("<<hello, world!>>", True),
    ],
)
def test_bigram_wrapping_covers_the_whole_answer(answer, expected):
    checker = instructions.BiGramWrappingChecker("detectable_format:bigram_wrapping")
    checker.build_description()
    assert checker.check_following(answer) is expected


@pytest.mark.parametrize("answer,expected", [("bC", 1.0), ("b", 0.0), ("<think>reasoning</think>bC", 1.0)])
def test_standard_if_verifier_uses_corrected_copy_checker(answer, expected):
    label = repr(
        [{"instruction_id": ["new:copy_span_idx"], "kwargs": [{"prompt_to_repeat": "AbCd", "n_start": 1, "n_end": 2}]}]
    )
    assert ground_truth_utils.IFEvalVerifier()([], answer, label).score == expected


def test_standard_if_verifier_preserves_fractional_constraint_rewards():
    label = repr(
        [
            {
                "instruction_id": ["keywords:at_least_once", "detectable_format:bigram_wrapping"],
                "kwargs": [{"keyword": "computer"}, None],
            }
        ]
    )
    verifier = ground_truth_utils.IFEvalVerifier()
    assert verifier([], "<<computer computer>>", label).score == 1.0
    assert verifier([], "computer computer", label).score == 0.5
    assert verifier([], "", label).score == 0.0


def test_explicit_keyword_frequency_rule_is_unchanged():
    checker = instructions.KeywordFrequencyChecker("keywords:frequency")
    checker.build_description(keyword="computer", frequency=2, relation="less than")
    assert checker.check_following("computer")
    assert not checker.check_following("computer computer")


def test_legacy_keyword_generation_and_scores():
    checker = instructions.KeywordFrequencyOnceChecker("keywords:word_once")
    assert checker.build_description(keyword="computer") == "Include keyword computer exactly once in your response."
    assert checker.check_following("COMPUTER")
    assert not checker.check_following("computer computer")
    checker.build_description(keyword="(?i)apple")
    assert checker.check_following("APPLE")
    inclusion = instructions.KeywordAtLeastOnceChecker("keywords:at_least_once")
    assert "at least once" in inclusion.build_description(keyword="computer")
