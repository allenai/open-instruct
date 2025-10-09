import unittest

import parameterized

from open_instruct import math_utils


class TestStripString(unittest.TestCase):
    @parameterized.parameterized.expand(
        [(r"50\% of the total", "50ofthetotal"), (r"50% of the total", "50%ofthetotal")]
    )
    def test_percentage_handling(self, input_string, expected_output):
        result = math_utils.strip_string(input_string)
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
