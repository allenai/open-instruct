import unittest

import parameterized

from open_instruct import math_utils


class TestStripString(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (r"\frac{1}{2}", r"\frac{1}{2}"),
            (r"\frac 1 2", r"\frac{1}{2}"),
            (r"  \frac{1}{2}  ", r"\frac{1}{2}"),
            (r"\left(\frac{1}{2}\right)", r"(\frac{1}{2})"),
            (r"50\%", "50"),
            (r"0.5", r"\frac{1}{2}"),
            (r"\tfrac{3}{4}", r"\frac{3}{4}"),
            (r"\sqrt2", r"\sqrt{2}"),
            (r"1 + 2", "1+2"),
        ]
    )
    def test_latex_normalization(self, input_string, expected_output):
        result = math_utils.strip_string(input_string)
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
