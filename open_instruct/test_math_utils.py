import unittest

import parameterized


class TestStripString(unittest.TestCase):
    @parameterized.parameterized.expand(
        [(r"50\% of the total", "50 of the total"), (r"50% of the total", "50% of the total")]
    )
    def test_percentage_handling(self, input_string, expected_output):
        result = input_string.replace("\\%", "")
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
