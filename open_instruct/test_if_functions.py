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


class TestCopySpanIdxChecker(unittest.TestCase):
    PROMPT = "The quick brown fox jumps over the lazy dog"

    def _checker(self, n_start, n_end):
        checker = instructions.CopySpanIdxChecker("new:copy_span_idx")
        checker.build_description(prompt_to_repeat=self.PROMPT, n_start=n_start, n_end=n_end)
        return checker

    def test_span_includes_end_index(self):
        checker = self._checker(4, 8)
        self.assertTrue(checker.check_following("quick"))
        self.assertFalse(checker.check_following("quic"))

    def test_zero_start_is_kept(self):
        checker = self._checker(0, 2)
        self.assertEqual(checker.get_instruction_args()["n_start"], 0)
        self.assertTrue(checker.check_following("The"))


if __name__ == "__main__":
    unittest.main()
