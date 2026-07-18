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



class TestVerifyPostscript(unittest.TestCase):
    @parameterized.expand(
        [
            ("ps_end", "Body text.\nP.S. extra note", "P.S.", True),
            ("case_insensitive", "Body.\np.s. note", "P.S.", True),
            ("marker_only_mid", "P.S. early\nstill going", "P.S.", True),
            ("missing", "No postscript here", "P.S.", False),
            ("case_miss_old", "Body p.s. note", "P.S.", True),
        ]
    )
    def test_verify_postscript(self, _name, text, marker, expected):
        self.assertEqual(if_functions.verify_postscript(text, marker), expected)


if __name__ == "__main__":
    unittest.main()
