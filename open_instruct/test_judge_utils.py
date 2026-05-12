"""Tests for open_instruct.judge_utils.

Regression coverage for PR #1618: the gpt-4o output price was missing a
zero ($1/1M instead of $10/1M).
"""

import unittest

from parameterized import parameterized

from open_instruct import judge_utils


class TestPricePerToken(unittest.TestCase):
    @parameterized.expand([("gpt-4o",), ("gpt-4o-standard",)])
    def test_gpt_4o_output_price_matches_published_rate(self, model):
        self.assertAlmostEqual(judge_utils.PRICE_PER_TOKEN[model]["input"], 2.5e-6)
        self.assertAlmostEqual(judge_utils.PRICE_PER_TOKEN[model]["output"], 1.0e-5)

    @parameterized.expand(
        [
            ("gpt-4",),
            ("gpt-3.5-turbo",),
            ("gpt-4-1106-preview",),
            ("gpt-4o",),
            ("gpt-4o-mini",),
            ("gpt-4o-standard",),
            ("gpt-4.1",),
        ]
    )
    def test_output_price_at_least_input_price(self, model):
        rates = judge_utils.PRICE_PER_TOKEN[model]
        self.assertGreaterEqual(rates["output"], rates["input"])


if __name__ == "__main__":
    unittest.main()
