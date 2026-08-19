#!/usr/bin/env python3
"""
Tests for judge output score extraction
"""

import unittest

from parameterized import parameterized

from open_instruct.judge_utils import extract_json_score_with_fallback

try:
    from open_instruct.judge_utils import coerce_score
except ImportError:  # pre-fix versions of judge_utils
    coerce_score = None


class TestExtractJsonScoreWithFallback(unittest.TestCase):
    """The factuality and refusal templates instruct the judge to answer exactly
    "True" or "False"; a compliant verdict must not collapse to the 0.0 error
    sentinel, or correct and incorrect answers become indistinguishable."""

    @parameterized.expand(
        [
            ("template_compliant_true", '{"REASONING": "Dates match.", "SCORE": "True"}', 1.0),
            ("template_compliant_false", '{"REASONING": "Wrong person.", "SCORE": "False"}', 0.0),
            ("lowercase_true", '{"REASONING": "ok", "SCORE": "true"}', 1.0),
            ("json_boolean_literal", '{"REASONING": "ok", "SCORE": true}', 1.0),
            ("numeric_string", '{"REASONING": "ok", "SCORE": "1"}', 1.0),
            ("numeric_float", '{"REASONING": "ok", "SCORE": 0.5}', 0.5),
            ("fenced_json", '```json\n{"REASONING": "ok", "SCORE": "True"}\n```', 1.0),
        ]
    )
    def test_json_path(self, _name, payload, expected):
        _reasoning, score = extract_json_score_with_fallback(payload)
        self.assertEqual(score, expected)

    @parameterized.expand(
        [
            ("regex_fallback_true", 'prose before {"REASONING": "ok", "SCORE": "True" trailing junk', 1.0),
            ("regex_fallback_false", 'prose before {"REASONING": "ok", "SCORE": "False" trailing junk', 0.0),
            ("regex_fallback_numeric", 'prose before {"REASONING": "ok", "SCORE": "0.5" trailing junk', 0.5),
        ]
    )
    def test_regex_fallback_path(self, _name, payload, expected):
        _reasoning, score = extract_json_score_with_fallback(payload)
        self.assertEqual(score, expected)

    def test_unparseable_still_returns_zero(self):
        _reasoning, score = extract_json_score_with_fallback("no score here at all")
        self.assertEqual(score, 0.0)

    def test_true_and_false_are_distinguishable(self):
        # The regression this file exists for: both verdicts scored 0.0.
        _r, true_score = extract_json_score_with_fallback('{"REASONING": "r", "SCORE": "True"}')
        _r, false_score = extract_json_score_with_fallback('{"REASONING": "r", "SCORE": "False"}')
        self.assertNotEqual(true_score, false_score)


@unittest.skipIf(coerce_score is None, "coerce_score not present in this judge_utils version")
class TestCoerceScore(unittest.TestCase):
    @parameterized.expand(
        [
            ("true_string", "True", 1.0),
            ("false_string", "False", 0.0),
            ("padded_case", "  FALSE  ", 0.0),
            ("python_bool", True, 1.0),
            ("int", 7, 7.0),
            ("numeric_string", "0.25", 0.25),
        ]
    )
    def test_coercions(self, _name, raw, expected):
        self.assertEqual(coerce_score(raw), expected)

    def test_non_numeric_string_raises(self):
        with self.assertRaises(ValueError):
            coerce_score("maybe")


if __name__ == "__main__":
    unittest.main()
