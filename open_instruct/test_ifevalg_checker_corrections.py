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
    ],
)
def test_existing_frequency_checker_supports_at_least_once(keyword, answer, expected):
    checker = instructions.KeywordFrequencyChecker("keywords:frequency")
    checker.build_description(keyword=keyword, frequency=1, relation="at least")
    assert checker.check_following(answer) is expected


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("", False),
        ("hello", False),
        ("<<hello world>> stray", False),
        ("<<hello>> <<world>>", False),
        ("<<one two three>>", False),
        ("<<one two>> <<three>>", True),
        ("<<hello>>", True),
        ("<<one>> <<two three>>", False),
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
                "instruction_id": ["keywords:frequency", "detectable_format:bigram_wrapping"],
                "kwargs": [{"keyword": "computer", "frequency": 1, "relation": "at least"}, None],
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


CAPITAL_ENGLISH = (
    "SONS: ROBERT, JOHN, TAZEWELL, DAVID, JOHN ALEXANDER, LACHLAN, LYON, ROBERT FITZWALTER. "
    "DAUGHTERS: MARY, LETITIA, ELIZABETH, ANNE, ALICE, JULIA, MARGARET PEARL."
)


def test_capital_checker_detects_language_on_lowercased_text():
    checker = instructions.CapitalLettersEnglishChecker("change_case:english_capital")
    checker.build_description()
    with patch.object(instructions.langdetect, "detect", return_value="en") as detect:
        assert checker.check_following("THE ANSWER IS YES.")
        assert not checker.check_following("The answer is yes.")
    detect.assert_called_once_with("the answer is yes.")


def test_capital_checker_accepts_capitalised_english_and_rejects_other_languages():
    checker = instructions.CapitalLettersEnglishChecker("change_case:english_capital")
    checker.build_description()
    # Unseeded langdetect on the capitalised text reports German for this published answer.
    assert checker.check_following(CAPITAL_ENGLISH)
    assert not checker.check_following("DIE BESPRECHUNG BEGINNT HEUTE UM ZWÖLF UHR IM GROSSEN SAAL.")
    assert not checker.check_following(CAPITAL_ENGLISH.lower())


def test_language_verdicts_do_not_depend_on_the_random_state():
    # Capitalised text leaves langdetect near a tie, which is where unseeded calls disagree.
    text = "THE MEETING WILL START AT NOON AND EVERYONE SHOULD BRING THEIR NOTES."
    instructions.langdetect.DetectorFactory.seed = None
    verdicts = {instructions._detect_language(text) for _ in range(20)}
    assert len(verdicts) == 1
    assert instructions.langdetect.DetectorFactory.seed == 0
