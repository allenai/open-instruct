"""Tests for MiniMax provider support in judge_utils and context_window_checker."""

import unittest

from open_instruct.context_window_checker import get_encoding_for_model
from open_instruct.judge_utils import PRICE_PER_TOKEN


class TestMiniMaxPricing(unittest.TestCase):
    """Test MiniMax models are present in PRICE_PER_TOKEN with correct values."""

    def test_minimax_m2_7_in_price_table(self):
        self.assertIn("MiniMax-M2.7", PRICE_PER_TOKEN)

    def test_minimax_m2_7_highspeed_in_price_table(self):
        self.assertIn("MiniMax-M2.7-highspeed", PRICE_PER_TOKEN)

    def test_minimax_m2_7_input_price(self):
        price = PRICE_PER_TOKEN["MiniMax-M2.7"]["input"]
        # $0.30 per million tokens = 0.0000003 per token
        self.assertAlmostEqual(price, 0.0000003)

    def test_minimax_m2_7_output_price(self):
        price = PRICE_PER_TOKEN["MiniMax-M2.7"]["output"]
        # $1.20 per million tokens = 0.0000012 per token
        self.assertAlmostEqual(price, 0.0000012)

    def test_minimax_m2_7_highspeed_input_price(self):
        price = PRICE_PER_TOKEN["MiniMax-M2.7-highspeed"]["input"]
        # $0.60 per million tokens = 0.0000006 per token
        self.assertAlmostEqual(price, 0.0000006)

    def test_minimax_m2_7_highspeed_output_price(self):
        price = PRICE_PER_TOKEN["MiniMax-M2.7-highspeed"]["output"]
        # $2.40 per million tokens = 0.0000024 per token
        self.assertAlmostEqual(price, 0.0000024)


class TestMiniMaxEncoding(unittest.TestCase):
    """Test MiniMax models use the correct tiktoken encoding."""

    def test_minimax_m2_7_uses_cl100k_base(self):
        encoding = get_encoding_for_model("MiniMax-M2.7")
        self.assertEqual(encoding.name, "cl100k_base")

    def test_minimax_m2_7_highspeed_uses_cl100k_base(self):
        encoding = get_encoding_for_model("MiniMax-M2.7-highspeed")
        self.assertEqual(encoding.name, "cl100k_base")

    def test_minimax_case_insensitive(self):
        # Model names passed via litellm may have different casing
        encoding = get_encoding_for_model("minimax-m2.7")
        self.assertEqual(encoding.name, "cl100k_base")


if __name__ == "__main__":
    unittest.main()
