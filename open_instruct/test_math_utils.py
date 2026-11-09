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


class TestStripOuterMathDelimiters(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("inline_paren", r"\(2^4\)", "2^4"),
            ("inline_paren_padded", r"\( x^4 \)", "x^4"),
            ("display_bracket", r"\[ 98 \]", "98"),
            ("display_bracket_multiline", "\\[\n172890\n\\]", "172890"),
            ("dollar", r"$\pi$", r"\pi"),
            ("double_dollar", r"$$\frac{1}{2}$$", r"\frac{1}{2}"),
            ("surrounding_whitespace", "  \\(\\frac{\\pi}{2}\\) ", r"\frac{\pi}{2}"),
            ("only_one_pair_removed", r"\(\(2^4\)\)", r"\(\(2^4\)\)"),
        ]
    )
    def test_removes_one_pair_around_whole_answer(self, _name, text, expected):
        self.assertEqual(math_utils.strip_outer_math_delimiters(text), expected)

    @parameterized.parameterized.expand(
        [
            ("plain", "2^4"),
            ("two_spans", r"\(a\) and \(b\)"),
            ("two_dollar_spans", r"$a$, $b$"),
            ("prose_with_span", r"The set \( X \) exists."),
            ("unbalanced_open", r"\(2^4"),
            ("unbalanced_close", r"2^4\)"),
            ("mismatched_pair", r"\(2^4\]"),
            ("empty_pair", r"\(\)"),
            ("whitespace_pair", r"$ $"),
            ("escaped_dollar_amount", r"\$5"),
            ("escaped_closing_dollar", r"$5\$"),
            ("boxed_is_not_a_delimiter", r"\boxed{16}"),
            ("left_right_parentheses", r"\left(\frac{1}{2}\right)"),
        ]
    )
    def test_leaves_other_text_unchanged(self, _name, text):
        self.assertEqual(math_utils.strip_outer_math_delimiters(text), text)

    def test_non_string_is_returned_unchanged(self):
        self.assertEqual(math_utils.strip_outer_math_delimiters(16), 16)
        self.assertIsNone(math_utils.strip_outer_math_delimiters(None))


if __name__ == "__main__":
    unittest.main()
