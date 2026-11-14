"""Tests for IFEval constraint verifiers in open_instruct.if_functions.

Covers regression fixes from PRs #1615 / #1646 (validate_choice operand
direction) and #1646 (validate_frequency_capital_words "around" tolerance).
"""

import unittest

from parameterized import parameterized

from open_instruct import if_functions
from open_instruct.IFEvalG import instructions


class TestValidateChoice(unittest.TestCase):
    @parameterized.expand(
        [
            ("option_in_text", "I believe the answer is B", ["A", "B", "C", "D"], True),
            ("option_appears", "I choose red", ["red", "blue"], True),
            ("no_option_present", "I choose green", ["red", "blue"], False),
            ("exact_match", "red", ["red", "blue"], True),
        ]
    )
    def test_validate_choice(self, _name, text, options, expected):
        self.assertEqual(if_functions.validate_choice(text, options), expected)


class TestValidateFrequencyCapitalWords(unittest.TestCase):
    @parameterized.expand(
        [
            ("around_exact", "AAA BBB CCC DDD EEE", 5, "around", True),
            ("around_off_by_one_low", "AAA BBB CCC DDD", 5, "around", True),
            ("around_off_by_one_high", "AAA BBB CCC DDD EEE FFF", 5, "around", True),
            ("around_far_off", "AAA BBB", 5, "around", False),
            ("at_least_satisfied", "AAA BBB CCC", 2, "at least", True),
            ("at_most_violated", "AAA BBB CCC DDD", 2, "at most", False),
        ]
    )
    def test_validate_frequency_capital_words(self, _name, text, n, quantifier, expected):
        self.assertEqual(if_functions.validate_frequency_capital_words(text, n, quantifier), expected)


class TestVerifySentenceConstraint(unittest.TestCase):
    @parameterized.expand(
        [
            # Trailing whitespace / newline must not add a phantom sentence: each
            # of these has the correct number of sentences for "at most".
            ("trailing_space_at_most", "One sentence here. ", 1, "at most", True),
            ("trailing_newline_at_most", "First. Second.\n", 2, "at most", True),
            ("trailing_blank_lines_at_most", "Only one.\n\n", 1, "at most", True),
            # Normal cases (no trailing terminator+space) are unaffected.
            ("no_trailing_at_most", "First. Second.", 2, "at most", True),
            ("three_sentences_at_least", "One. Two. Three.", 3, "at least", True),
            ("around_exact", "One. Two. Three.", 3, "around", True),
            # The count is still real: genuinely too many sentences is rejected.
            ("too_many_at_most", "One. Two. Three. ", 2, "at most", False),
            ("too_few_at_least", "Only one.", 2, "at least", False),
        ]
    )
    def test_verify_sentence_constraint(self, _name, text, n, quantifier, expected):
        self.assertEqual(if_functions.verify_sentence_constraint(text, n, quantifier), expected)


class TestExcludeWordHarderChecker(unittest.TestCase):
    @parameterized.expand(
        [
            ("between_spaces", "I drove my car to work.", False),
            ("before_comma", "I drove my car, then walked.", False),
            ("start_of_text", "Car prices rose.", False),
            ("end_of_text", "I sold the car", False),
            ("in_brackets", "The (car) was red.", False),
            ("before_newline", "We need a car\nfor it.", False),
            ("upper_case", "I drove my CAR.", False),
            ("absent", "I took the bus.", True),
            ("inside_longer_word", "A scar and a carpet and two cars.", True),
        ]
    )
    def test_check_following(self, _name, text, expected):
        checker = instructions.ExcludeWordHarderChecker("keywords:exclude_word_harder")
        checker.build_description(keyword="car")
        self.assertEqual(checker.check_following(text), expected)

    def test_keyword_with_punctuation(self):
        checker = instructions.ExcludeWordHarderChecker("keywords:exclude_word_harder")
        checker.build_description(keyword="piano.")
        self.assertFalse(checker.check_following("She plays the piano."))
        self.assertTrue(checker.check_following("She plays the pianos."))


if __name__ == "__main__":
    unittest.main()
