import unittest

import parameterized


class TestStripString(unittest.TestCase):
    @parameterized.parameterized.expand(
        [(r"50\% of the total", "50 of the total"), (r"50% of the total", "50% of the total")]
    @parameterized.parameterized.expand([
        (r"50\% of the total", "50ofthetotal"),
        (r"50% of the total", "50%ofthetotal"),
    ])
    def test_percentage_handling(self, input_string, expected_output):
        result = open_instruct.math_utils.strip_string(input_string)
        self.assertEqual(result, expected_output)
    unittest.main()
