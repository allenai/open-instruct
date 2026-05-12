"""Tests for IFEval constraint verifiers in open_instruct.if_functions.

Covers regression fixes from PRs #1615 / #1646 (validate_choice operand
direction) and #1646 (validate_frequency_capital_words "around" tolerance).
"""

import importlib.util
import pathlib
import unittest

from parameterized import parameterized

from open_instruct import if_functions

_SCRIPTS_IF_FUNCTIONS_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "eval_constraints" / "if_functions.py"
)


def _load_scripts_if_functions():
    spec = importlib.util.spec_from_file_location("scripts_eval_constraints_if_functions", _SCRIPTS_IF_FUNCTIONS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestValidateChoice(unittest.TestCase):
    @parameterized.expand(
        [
            ("option_in_text", "I believe the answer is B", ["A", "B", "C", "D"], True),
            ("option_appears", "I choose red", ["red", "blue"], True),
            ("no_option_present", "I choose green", ["red", "blue"], False),
            ("exact_match", "red", ["red", "blue"], True),
        ]
    )
    def test_open_instruct_module(self, _name, text, options, expected):
        self.assertEqual(if_functions.validate_choice(text, options), expected)

    @parameterized.expand(
        [
            ("option_in_text", "I believe the answer is B", ["A", "B", "C", "D"], True),
            ("no_option_present", "I choose green", ["red", "blue"], False),
        ]
    )
    def test_scripts_module(self, _name, text, options, expected):
        scripts_mod = _load_scripts_if_functions()
        self.assertEqual(scripts_mod.validate_choice(text, options), expected)


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
    def test_open_instruct_module(self, _name, text, n, quantifier, expected):
        self.assertEqual(if_functions.validate_frequency_capital_words(text, n, quantifier), expected)

    def test_scripts_around_tolerance_allows_off_by_one(self):
        scripts_mod = _load_scripts_if_functions()
        self.assertTrue(scripts_mod.validate_frequency_capital_words("AAA BBB CCC DDD", 5, "around"))


if __name__ == "__main__":
    unittest.main()
