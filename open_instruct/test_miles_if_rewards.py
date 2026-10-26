"""Strict IF scoring, the fixed IFEvalG checkers and the IF row screens."""

import json
import random

import nltk
import pytest

from open_instruct.IFEvalG import instructions_registry
from open_instruct.miles.rewards import if_rewards


def _has_punkt_tab():
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        return False
    return True


# keywords:start_end tokenizes with NLTK; the runtime image ships punkt_tab.
needs_punkt = pytest.mark.skipif(not _has_punkt_tab(), reason="NLTK punkt_tab data is not installed")


def target(*constraints):
    """An Open Instruct IF target: a list literal holding the constraint dict as JSON."""
    ids = [instruction_id for instruction_id, _ in constraints]
    kwargs = [kwargs for _, kwargs in constraints]
    return str([json.dumps({"instruction_id": ids, "kwargs": kwargs})])


def check(instruction_id, text, **kwargs):
    checker = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    checker.build_description(**kwargs)
    return checker.check_following(text)


def score(prediction, label, query=None, **config):
    verifier = if_rewards.StrictIFVerifier(if_rewards.StrictIFConfig(**config))
    return verifier([], prediction, label, query=query).score


# Checker fixes.


def test_word_count_different_numbers_supports_an_exact_relation():
    kwargs = {"keyword": "cat", "frequency": 3}
    assert check("keywords:word_count_different_numbers", "cat cat cat", relation="exactly", **kwargs)
    assert not check("keywords:word_count_different_numbers", "cat cat", relation="exactly", **kwargs)
    # The stored relation still applies when given.
    assert not check("keywords:word_count_different_numbers", "cat cat cat", relation="less than", **kwargs)


def test_nth_paragraph_out_of_range_fails_deterministically():
    results = set()
    for seed in range(20):
        random.seed(seed)
        results.add(
            check(
                "length_constraints:nth_paragraph_first_word",
                "hello there\n\nhello again",
                num_paragraphs=2,
                nth_paragraph=5,
                first_word="hello",
            )
        )
    assert results == {False}
    assert check(
        "length_constraints:nth_paragraph_first_word",
        "one\n\nhello again",
        num_paragraphs=2,
        nth_paragraph=2,
        first_word="hello",
    )


def test_language_checks():
    assert check("change_case:english_capital", "HELLO WORLD")
    assert check("change_case:english_capital", "THIS IS A SHORT REPLY")
    assert not check("change_case:english_capital", "This is not all caps")
    assert not check("language:response_language", "123 456 !!!", language="de")


def test_palindrome_needs_a_real_palindrome():
    assert not check("keywords:palindrome", "I have a cat")
    assert check("keywords:palindrome", "The racecar was fast")
    assert check("keywords:palindrome", "Level, said Anna.")


def test_sentence_hyphens_needs_two_joined_sentences():
    assert not check("detectable_format:sentence_hyphens", "This is one sentence.")
    assert check("detectable_format:sentence_hyphens", "This is one.-This is two.")
    assert not check("detectable_format:sentence_hyphens", "This is one. This is two.")


def test_exclude_word_harder_matches_whole_words_in_any_case_and_position():
    assert not check("keywords:exclude_word_harder", "Cat, dog.", keyword="cat")
    assert not check("keywords:exclude_word_harder", "cat dog", keyword="cat")
    assert not check("keywords:exclude_word_harder", "the dog and the cat", keyword="cat")
    assert check("keywords:exclude_word_harder", "a category", keyword="cat")


def test_word_keywords_match_whole_words_and_literal_text():
    assert not check("keywords:word_once", "category", keyword="cat")
    assert check("keywords:word_once", "The cat sat.", keyword="cat")
    assert check("count:count_increment_word", "c++ and go, go", keyword1="c++", keyword2="go")
    assert check("keywords:forbidden_words", "I like python", forbidden_words=["c++"])
    assert not check("keywords:forbidden_words", "I like c++ a lot", forbidden_words=["c++"])
    # keywords:existence keeps upstream IFEval substring matching, now escaped.
    assert check("keywords:existence", "category", keywords=["cat"])
    assert not check("keywords:existence", "I like c", keywords=["c++"])


def test_section_splitter_is_literal():
    text = "PART* 1 alpha PART* 2 beta"
    assert check("detectable_format:multiple_sections", text, section_spliter="PART*", num_sections=2)


# Strict verifier.


def test_all_or_nothing_and_partial_credit():
    label = target(("punctuation:no_comma", {}), ("change_case:english_lowercase", {}))
    assert score("hello there, friend", label) == 0.0
    assert score("hello there friend", label) == 1.0
    assert score("hello there, friend", label, all_or_nothing=False) == 0.5


def test_empty_and_thinking_only_answers_score_zero():
    label = target(("punctuation:no_comma", {}))
    assert score("", label) == 0.0
    assert score("<think>reasoning</think>", label) == 0.0
    assert score("<think>a, b</think>no commas here", label) == 1.0


def test_hidden_relation_reads_as_exact_unless_the_prompt_states_it():
    label = target(
        ("keywords:word_count_different_numbers", {"keyword": "cat", "frequency": 3, "relation": "less than"})
    )
    hidden = "Write about pets. In your response, the word cat should appear 3 times."
    stated = "Write about pets. The word cat should appear less than 3 times."
    assert score("cat cat cat", label, query=hidden) == 1.0
    assert score("cat", label, query=hidden) == 0.0
    assert score("cat", label, query=stated) == 1.0
    assert score("cat", label, query=hidden, infer_hidden_relation=False) == 1.0


def test_label_encodings():
    bundle = {"instruction_id": ["punctuation:no_comma"], "kwargs": [{}]}
    for label in (str([json.dumps(bundle)]), str([bundle]), bundle, [str([bundle])]):
        assert score("no commas", label) == 1.0


def test_normalize_if_target_rewrites_only_hidden_relations():
    kwargs = {"keyword": "cat", "frequency": 3, "relation": "at least"}
    label = target(("keywords:word_count_different_numbers", kwargs), ("punctuation:no_comma", None))
    rewritten = if_rewards.normalize_if_target("The word cat should appear 3 times.", label)
    bundle = if_rewards.constraint_bundle(rewritten)
    assert bundle["kwargs"][0]["relation"] == "exactly"
    assert bundle["kwargs"][1] is None
    assert if_rewards.normalize_if_target("Use the word cat at least 3 times.", label) == label


# Row screens.


def problems(prompt, *constraints):
    return if_rewards.if_row_problems({"prompt": prompt, "ground_truth": [target(*constraints)]})


def test_usable_row_has_no_problems():
    assert (
        problems("Write a story.", ("length_constraints:number_words", {"num_words": 50, "relation": "at least"}))
        == []
    )


def test_trivial_answer_rows_are_flagged():
    assert "trivial_answer_passes" in problems("Write a story.", ("punctuation:no_comma", {}))


def test_known_unsat_and_invalid_kwargs():
    found = problems(
        "Write a story.",
        ("keywords:keyword_specific_position", {"keyword": "dog", "n": 2, "m": 3}),
        ("length_constraints:nth_paragraph_first_word", {"num_paragraphs": 2, "nth_paragraph": 4, "first_word": "so"}),
    )
    assert "known_unsat_type:keywords:keyword_specific_position" in found
    assert "invalid_kwargs:length_constraints:nth_paragraph_first_word" in found


def test_missing_kwargs_are_nondeterministic():
    found = problems("Write a story.", ("language:response_language", {}))
    assert "nondeterministic_kwargs:language:response_language" in found


def test_unknown_instruction():
    assert problems("x", ("made:up", {})) == ["unknown_instruction:made:up"]


@pytest.mark.parametrize(
    ("constraints", "rule"),
    [
        (
            (("change_case:english_capital", {}), ("change_case:english_lowercase", {})),
            "english_capital+english_lowercase",
        ),
        (
            (("change_case:english_lowercase", {}), ("language:response_language", {"language": "fr"})),
            "english_case+response_language",
        ),
        (
            (
                ("length_constraints:number_words", {"num_words": 10, "relation": "less than"}),
                ("length_constraints:number_words", {"num_words": 20, "relation": "at least"}),
            ),
            "words_count",
        ),
        (
            (
                ("length_constraints:number_words", {"num_words": 3, "relation": "less than"}),
                ("length_constraints:number_paragraphs", {"num_paragraphs": 4}),
            ),
            "words+star_paragraphs",
        ),
        (
            (("keywords:word_once", {"keyword": "cat"}), ("keywords:forbidden_words", {"forbidden_words": ["Cat"]})),
            "keywords:forbidden_words+keywords:word_once:cat",
        ),
        (
            (
                ("first_word:first_word_answer", {"first_word": "yes"}),
                ("first_word:first_word_sent", {"first_word": "no"}),
            ),
            "first_word",
        ),
        (
            (
                ("last_word:last_word_answer", {"last_word": "done"}),
                ("startend:end_checker", {"end_phrase": "Any other questions?"}),
            ),
            "last_word",
        ),
        (
            (("punctuation:punctuation_dot", {}), ("detectable_content:postscript", {"postscript_marker": "P.S."})),
            "punctuation_dot+required_dot",
        ),
        (
            (("startend:quotation", {}), ("first_word:first_word_answer", {"first_word": "hello"})),
            "quotation+first_word_answer",
        ),
        (
            (
                ("change_case:english_lowercase", {}),
                ("detectable_format:multiple_sections", {"section_spliter": "SECTION", "num_sections": 2}),
            ),
            "english_lowercase+case_sensitive_literal",
        ),
        (
            (("copy:copying_simple", {"prompt_to_repeat": "Write a story, please."}), ("punctuation:no_comma", {})),
            "copy+other",
        ),
    ],
)
def test_contradictions(constraints, rule):
    assert f"contradiction:{rule}" in problems("Write a story.", *constraints)


def test_copy_with_compatible_constraint_is_not_a_contradiction():
    found = problems(
        "Write a story.",
        ("copy:copying_simple", {"prompt_to_repeat": "write a story please"}),
        ("change_case:english_lowercase", {}),
    )
    assert not any(problem.startswith("contradiction:") for problem in found)


def test_empty_copy_span_is_a_contradiction_not_a_crash():
    # The v3 prep crashed here: LastWordAnswerChecker indexes the last word of the empty span.
    found = problems(
        "Write a story.",
        ("new:copy_span_idx", {"prompt_to_repeat": "write a story", "n_start": 3, "n_end": 3}),
        ("last_word:last_word_answer", {"last_word": "story"}),
    )
    assert "contradiction:copy+other" in found


def test_quotation_contradiction_holds_on_the_checkers():
    """The quotation rule matches what the checkers do on plausible attempts."""
    label = target(("startend:quotation", {}), ("first_word:first_word_answer", {"first_word": "hello"}))
    for attempt in ('"hello world hello"', '" hello world hello "', '"hello"'):
        assert score(attempt, label) == 0.0


@needs_punkt
def test_quotation_with_start_end_is_satisfiable():
    label = target(("startend:quotation", {}), ("keywords:start_end", {}))
    assert score('" hello world hello "', label) == 1.0
    assert not any(
        p.startswith("contradiction:")
        for p in problems("Write.", ("startend:quotation", {}), ("keywords:start_end", {}))
    )
