"""Unit-tests for `get_successful_tests_fast`."""

import unittest

from datasets import load_dataset

from open_instruct.code.code_utils import get_successful_tests_fast


class GetSuccessfulTestsFastTests(unittest.TestCase):
    """Tests mirroring the interactive checks from the original script."""

    # ---------- helpers & fixtures -------------------------------------------------
    @classmethod
    def setUpClass(cls):
        # shared objects that never mutate
        cls.program = "a = 1"
        cls.bad_test = "assert False"
        cls.good_test = "assert True"
        cls.time_out_test = (
            "for i in range(9999999999999999999):\n"
            "    for k in range(99999999999999999999):\n"
            '        print("hello world")'
        )

    # ---------- simple synthetic cases --------------------------------------------
    def test_mixed_pass_and_fail(self):
        """Bad + good + timeout tests: expect [F, T, F, T, T, F, T]."""
        tests = [
            self.bad_test,
            self.good_test,
            self.bad_test,
            self.good_test,
            self.good_test,
            self.time_out_test,
            self.good_test,
        ]
        expected = [0, 1, 0, 1, 1, 0, 1]

        result = get_successful_tests_fast(program=self.program, tests=tests)

        # The API most commonly returns a list[int] the same length as `tests`.
        # If your implementation instead returns e.g. indices or a set,
        # adapt the assertion below.
        self.assertEqual(
            result,
            expected,
            msg=f"Expected {expected} for mixed pass/fail case, got {result}",
        )

    def test_all_fail_or_timeout(self):
        """All failing or timing-out tests: expect a full-False result."""
        tests = [
            self.bad_test,
            self.bad_test,
            self.time_out_test,
            self.time_out_test,
            self.time_out_test,
            self.time_out_test,
        ]
        expected = [0] * len(tests)

        result = get_successful_tests_fast(program=self.program, tests=tests)
        self.assertEqual(
            result,
            expected,
            msg="All tests should fail or time-out, but result differed",
        )

    def test_tiger_lab_acecode_sample(self):
        """
        Replicates the sample in the script against an actual AceCode record.
        """
        ds = load_dataset("TIGER-Lab/AceCode-87K", split="train")

        # Choose the same sample index used in the original snippet.
        i = 1
        program = ds[i]["inferences"][-1]["completion"]
        tests = ds[i]["test_cases"]

        # The dataset also stores a pass-rate; we can use it to sanity-check.
        expected_passes = int(len(tests) * ds[i]["inferences"][-1]["pass_rate"])

        result = get_successful_tests_fast(program=program, tests=tests)

        # Robustly handle either a list[int] return or a collection of indices.
        if all(isinstance(x, int) for x in result):
            actual_passes = sum(result)
        else:
            actual_passes = len(result)

        self.assertEqual(
            actual_passes,
            expected_passes,
            msg="Pass count does not match `pass_rate` in dataset sample",
        )

    def test_add_function_example(self):
        """Small ‘add’ sample from the original script; two pass, one fail."""
        program = "\n\ndef add(a, b):\n    return a + b\n"
        tests = [
            "assert add(1, 2) == 3",  # pass
            "assert add(-1, 1) == 0",  # pass
            "assert add(0, 0) == 1",  # fail
        ]
        expected = [1, 1, 0]

        result = get_successful_tests_fast(program=program, tests=tests)
        self.assertEqual(result, expected, "Unexpected outcome for add() example")


if __name__ == "__main__":
    unittest.main()
