"""Tests for IFEval constraint verifiers in open_instruct.if_functions.

Covers regression fixes from PRs #1615 / #1646 (validate_choice operand
direction) and #1646 (validate_frequency_capital_words "around" tolerance).
"""

import unittest

from parameterized import parameterized

from open_instruct import if_functions


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


if __name__ == "__main__":
    unittest.main()


class TestVerifyBulletPoints(unittest.TestCase):
    @parameterized.expand(
        [
            ("exact_stars", "* one\n* two", 2, True),
            ("bold_not_bullet", "**Introduction**\n* one\n* two", 2, True),
            ("bold_false_positive_old", "**Introduction**\n* one\n* two", 3, False),
            ("dash_bullets", "- a\n- b\n- c", 3, True),
            ("mixed", "* a\n- b", 2, True),
            ("lone_star", "*", 1, True),
        ]
    )
    def test_verify_bullet_points(self, _name, text, n, expected):
        self.assertEqual(if_functions.verify_bullet_points(text, n), expected)


class TestVerifyParagraphCount(unittest.TestCase):
    @parameterized.expand(
        [
            ("triple_star", "Para1\n***\nPara2\n***\nPara3", 3, True),
            ("spaced_stars_not_divider", "Para1\n* * *\nPara2\n* * *\nPara3", 3, False),
            ("inline", "a***b", 2, True),
            ("wrong_count", "only one", 2, False),
        ]
    )
    def test_verify_paragraph_count(self, _name, text, n, expected):
        self.assertEqual(if_functions.verify_paragraph_count(text, n), expected)
