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


def _bullet_checker(num_bullets):
    checker = instructions.BulletListChecker("detectable_format:number_bullet_lists")
    checker.build_description(num_bullets=num_bullets)
    return checker


@pytest.mark.parametrize(
    "answer,num_bullets,expected",
    [
        ("* one\n* two\n* three", 3, True),
        ("- one\n- two\n- three", 3, True),
        ("Intro\n  * one\n  - two", 2, True),
        ("* one\n* two\n* three\n---\n*italic note*", 3, True),
        ("* one\n* two\n* three\n---\n*italic note*", 5, False),
        ("* one\n* two\n\n---\n\n* three\n*Note:* done", 3, True),
        ("* one\n* two\n* three\n*", 3, True),
        ("* one\n* two\n- ", 2, True),
        ("* one\n* two", 3, False),
        ("* one\n* two\n* three\n* four", 3, False),
        ("---\n***\n* * *\n*italic*\n**bold**", 0, True),
        ("*item without space\n* real", 1, True),
    ],
)
def test_bullet_count_ignores_rules_italics_and_empty_markers(answer, num_bullets, expected):
    assert _bullet_checker(num_bullets).check_following(answer) is expected


def _palindrome(answer):
    checker = instructions.PalindromeBasicChecker("keywords:palindrome")
    checker.build_description()
    return checker.check_following(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "The level is high.",
        "Racecar, anyone?",
        'She said "Wow!" loudly',
        "Madam",
        "Step on no pets.",
        "Was it a car or a cat I saw?",
        "We saw 202",
        "121",
        'Here is one: "A man, a plan, a canal: Panama!" enjoy',
        "Here is one: A man, a plan, a canal: Panama! enjoy",
        "(Cue the palindrome: 'A man, a plan, a canal: Panama!')",
        "a palindrome\u2014racecar\u2014appears",
        "palindrome \u2014*racecar*\u2014 appears",
        "Here's one you'll remember: \u201cStep on no pets\u201d, enjoy it",
        "Try this (Was it a car or a cat I saw) today",
        "see:level;radar,done",
        "The phrase A man a plan a canal Panama embodies this idea",
        "the palindrome *A man, a plan, a canal: Panama* is a classic",
        "[*A man, a plan, a canal: Panama*] phrase, a classic",
        "so step on no pets as a call to move forward",
        "Anna\u2019s Exit",
        "at noon\u2019s closing",
    ],
)
def test_palindrome_accepts_real_palindromes(answer):
    assert _palindrome(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "I saw a cat.",
        "a",
        "---",
        "### Title",
        "+ = ~",
        "aa bb",
        "No palindrome here",
        "",
        "ab, ba",
        "II",
        "11.",
        "1, 2; 1",
        "so a b a b a is cool",
        "now 1 2 1 2 1 is odd",
        "go on now",
        "a non-profit group",
        "a level-headed person",
        "a woman\u2019s eyes",
        "see a, b, a, b, a here",
        "Ayumi\u2019s eyes narrowed",
        "see [a] [b] [a] [c] [a] [b] [a] here",
        "so a t a c a t a is it",
        "He saw it was a good day",
    ],
)
def test_palindrome_rejects_single_characters_symbols_and_non_palindromes(answer):
    assert not _palindrome(answer)


def _first_word(cls_name, key, word, answer, **kwargs):
    checker = getattr(instructions, cls_name)(key)
    checker.build_description(first_word=word, **kwargs)
    return checker.check_following(answer)


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Building a house", True),
        ('"Building a house"', True),
        ("Building, a house", True),
        ("**Building** a house", True),
        ("[building] a house", True),
        ("<<Building a house", True),
        ("Buildings are tall", False),
        ("Rebuilding", False),
        ("", False),
        ("# Building a house", False),
    ],
)
def test_first_word_answer_ignores_edge_punctuation_and_markup(answer, expected):
    assert _first_word("FirstWordAnswerChecker", "first_word:first_word_answer", "building", answer) is expected


@pytest.mark.parametrize(
    "answer,expected",
    [('**Yes** it is. Yes, again! "Yes" too?', True), ("Yes it is. Maybe not.", False), ("Yesterday it is.", False)],
)
def test_first_word_of_each_sentence_ignores_edge_punctuation_and_markup(answer, expected):
    assert _first_word("FirstWordSentChecker", "first_word:first_word_sent", "yes", answer) is expected


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("Intro text.\n\n**Hello,** world\n\nlast", True),
        ("Intro text.\n\nHello, world\n\nlast", True),
        ('Intro text.\n\n"Hello world"\n\nlast', True),
        ("Intro text.\n\nHelloes world\n\nlast", False),
        ("Intro text.\n\nWorld hello\n\nlast", False),
    ],
)
def test_nth_paragraph_first_word_ignores_edge_markup(answer, expected):
    assert (
        _first_word(
            "ParagraphFirstWordCheck",
            "length_constraints:nth_paragraph_first_word",
            "hello",
            answer,
            num_paragraphs=3,
            nth_paragraph=2,
        )
        is expected
    )


def test_nth_paragraph_first_word_keeps_apostrophe_cut():
    answer = "Intro.\n\nyesterday's news\n\nlast"
    assert _first_word(
        "ParagraphFirstWordCheck",
        "length_constraints:nth_paragraph_first_word",
        "yesterday",
        answer,
        num_paragraphs=3,
        nth_paragraph=2,
    )
