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


class TestLastBoxedOnlyString(unittest.TestCase):
    def test_brace_form(self):
        self.assertEqual(math_utils.last_boxed_only_string(r"ans \boxed{42}"), r"\boxed{42}")

    def test_space_form(self):
        self.assertEqual(math_utils.last_boxed_only_string(r"ans \boxed 7$."), r"\boxed 7")

    def test_earlier_space_mention_does_not_override_later_brace(self):
        s = r"Hint: write \boxed as a command. The answer is \boxed{42}."
        self.assertEqual(math_utils.last_boxed_only_string(s), r"\boxed{42}")

    def test_last_space_wins_over_earlier_brace(self):
        s = r"first \boxed{1} then \boxed 2$ done"
        self.assertEqual(math_utils.last_boxed_only_string(s), r"\boxed 2")

    def test_nested_braces(self):
        s = r"\boxed{\frac{1}{2}}"
        self.assertEqual(math_utils.last_boxed_only_string(s), r"\boxed{\frac{1}{2}}")

    def test_missing_returns_none(self):
        self.assertIsNone(math_utils.last_boxed_only_string("no box here"))


if __name__ == "__main__":
    unittest.main()
