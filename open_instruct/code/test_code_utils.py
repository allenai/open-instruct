import unittest
from unittest.mock import MagicMock, patch


from open_instruct.code.code_utils import get_successful_tests_fast


class TestCodeUtils(unittest.TestCase):
    def test_get_successful_tests_fast_basic(self):
        """Test basic functionality with simple program and tests"""
        program = "a = 1"
        bad_test = "assert False"
        good_test = "assert True"

        test_case_status = get_successful_tests_fast(
            program=program,
            tests=[
                bad_test,
                good_test,
                bad_test,
                good_test,
                good_test,
                good_test,
            ],
        )
        expected = [0, 1, 0, 1, 1, 1]
        self.assertEqual(test_case_status, expected)

    def test_get_successful_tests_fast_timeout(self):
        """Test timeout handling with infinite loop"""
        program = "a = 1"
        bad_test = "assert False"
        time_out_test = """
for i in range(9999999999999999999):
    for k in range(99999999999999999999):
        print("hello world")
"""

        test_case_status = get_successful_tests_fast(
            program=program,
            tests=[
                bad_test,
                bad_test,
                time_out_test,
                time_out_test,
                time_out_test,
                time_out_test,
            ],
        )
        expected = [0, 0, 0, 0, 0, 0]
        self.assertEqual(test_case_status, expected)

    def test_get_successful_tests_fast_math_function(self):
        """Test with a simple math function"""
        program = "\ndef add(a, b):\n    return a + b\n"
        tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"]

        test_case_status = get_successful_tests_fast(
            program=program,
            tests=tests,
        )
        expected = [1, 1, 0]  # Third test should fail: add(0, 0) == 0, not 1
        self.assertEqual(test_case_status, expected)

    @patch("open_instruct.code.code_utils.load_dataset")
    def test_get_successful_tests_fast_with_dataset(self, mock_load_dataset):
        """Test with mocked dataset"""
        # Mock dataset structure
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = {
            "question": "Write a function to add two numbers",
            "inferences": [
                {"completion": "def add(a, b):\n    return a + b", "model_name": "test-model", "pass_rate": 0.8}
            ],
            "test_cases": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
        }
        mock_load_dataset.return_value = mock_dataset

        # This test would normally require the actual dataset
        # but we're just testing that the function works with the expected data structure
        program = "def add(a, b):\n    return a + b"
        tests = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]

        test_case_status = get_successful_tests_fast(
            program=program,
            tests=tests,
        )
        expected = [1, 1]
        self.assertEqual(test_case_status, expected)

    def test_empty_tests_list(self):
        """Test with empty tests list"""
        program = "a = 1"
        tests = []

        result = get_successful_tests_fast(program=program, tests=tests)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
