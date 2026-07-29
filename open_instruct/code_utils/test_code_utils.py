"""Unit-tests for `get_successful_tests_fast`."""

import gc
import json
import multiprocessing
import pathlib
import unittest

import parameterized

from open_instruct.code_utils import code_utils

TEST_DATA_DIR = pathlib.Path(__file__).parent.parent / "test_data"

SIMPLE_PROGRAM = "a = 1"
FAILING_TEST = "assert False"
PASSING_TEST = "assert True"
TIMEOUT_TEST = "import time\ntime.sleep(10)"


class BaseCodeTestCase(unittest.TestCase):
    """Base test class with cleanup for multiprocessing resources."""

    def tearDown(self):
        """Clean up multiprocessing resources after each test."""
        gc.collect()
        for child in multiprocessing.active_children():
            child.terminate()
            child.join(timeout=1)
            child.close()


class GetSuccessfulTestsFastTests(BaseCodeTestCase):
    def test_mixed_pass_and_fail(self):
        """Bad + good + timeout tests: expect [F, T, F, T, T, F, T]."""
        tests = [FAILING_TEST, PASSING_TEST, FAILING_TEST, PASSING_TEST, PASSING_TEST, TIMEOUT_TEST, PASSING_TEST]
        expected = [0, 1, 0, 1, 1, 0, 1]
        result, _ = code_utils.get_successful_tests_fast(program=SIMPLE_PROGRAM, tests=tests)
        self.assertEqual(result, expected)

    def test_all_fail_or_timeout(self):
        """All failing or timing-out tests: expect a full-False result."""
        tests = [FAILING_TEST, FAILING_TEST, TIMEOUT_TEST, TIMEOUT_TEST, TIMEOUT_TEST, TIMEOUT_TEST]
        expected = [0] * len(tests)

        result, _ = code_utils.get_successful_tests_fast(program=SIMPLE_PROGRAM, tests=tests)
        self.assertEqual(result, expected)

    def test_tiger_lab_acecode_sample(self):
        """Tests the script against an actual AceCode record."""
        with open(TEST_DATA_DIR / "acecode_sample.jsonl") as f:
            record = json.loads(f.readline())
        program = record["inferences"][-1]["completion"]
        tests = code_utils.decode_tests(record["test_cases"])

        expected_passes = int(len(tests) * record["inferences"][-1]["pass_rate"])

        result, _ = code_utils.get_successful_tests_fast(program=program, tests=tests)
        self.assertEqual(sum(result), expected_passes)

    def test_add_function_example(self):
        """Small ‘add’ sample; two pass, one fail."""
        program = "\n\ndef add(a, b):\n    return a + b\n"
        tests = [
            "assert add(1, 2) == 3",  # pass
            "assert add(-1, 1) == 0",  # pass
            "assert add(0, 0) == 1",  # fail
        ]
        expected = [1, 1, 0]

        result, _ = code_utils.get_successful_tests_fast(program=program, tests=tests)
        self.assertEqual(result, expected)


class GetSuccessfulTestsFunctionalTests(BaseCodeTestCase):
    """Tests for the LiveCodeBench call-based ("functional") format."""

    # LiveCodeBench encodes one JSON argument per line and the expected return value as JSON.
    TWO_SUM_TESTS = [
        {"input": "[2,7,11,15]\n9", "output": "[0,1]", "fn_name": "twoSum"},
        {"input": "[3,2,4]\n6", "output": "[1,2]", "fn_name": "twoSum"},
        {"input": "[3,3]\n6", "output": "[0,1]", "fn_name": "twoSum"},
    ]

    SOLUTION_CLASS = (
        "class Solution:\n"
        "    def twoSum(self, nums: List[int], target: int) -> List[int]:\n"
        "        seen = {}\n"
        "        for i, n in enumerate(nums):\n"
        "            if target - n in seen:\n"
        "                return [seen[target - n], i]\n"
        "            seen[n] = i\n"
    )

    def test_correct_solution_passes(self):
        result, _ = code_utils.get_successful_tests_functional(
            program=self.SOLUTION_CLASS, tests=self.TWO_SUM_TESTS, max_execution_time=2.0
        )
        self.assertEqual(result, [1, 1, 1])

    def test_bare_function_passes(self):
        """The solution may define a plain function instead of a `class Solution` method."""
        program = (
            "def twoSum(nums, target):\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        if target - n in seen:\n"
            "            return [seen[target - n], i]\n"
            "        seen[n] = i\n"
        )
        result, _ = code_utils.get_successful_tests_functional(
            program=program, tests=self.TWO_SUM_TESTS, max_execution_time=2.0
        )
        self.assertEqual(result, [1, 1, 1])

    def test_wrong_answer_fails(self):
        program = "class Solution:\n    def twoSum(self, nums, target):\n        return [0, 0]\n"
        result, _ = code_utils.get_successful_tests_functional(
            program=program, tests=self.TWO_SUM_TESTS, max_execution_time=2.0
        )
        self.assertEqual(result, [0, 0, 0])

    def test_tuple_return_is_not_penalized(self):
        """Ground truth sequences are never tuples, so a tuple return still counts as correct."""
        program = (
            "class Solution:\n"
            "    def twoSum(self, nums, target):\n"
            "        seen = {}\n"
            "        for i, n in enumerate(nums):\n"
            "            if target - n in seen:\n"
            "                return (seen[target - n], i)\n"
            "            seen[n] = i\n"
        )
        result, _ = code_utils.get_successful_tests_functional(
            program=program, tests=self.TWO_SUM_TESTS, max_execution_time=2.0
        )
        self.assertEqual(result, [1, 1, 1])

    def test_timeout_fails(self):
        program = "class Solution:\n    def twoSum(self, nums, target):\n        import time\n        time.sleep(10)\n"
        result, _ = code_utils.get_successful_tests_functional(
            program=program, tests=self.TWO_SUM_TESTS, max_execution_time=1.0
        )
        self.assertEqual(result, [0, 0, 0])

    def test_uncompilable_program_fails(self):
        result, _ = code_utils.get_successful_tests_functional(
            program="def f(:", tests=self.TWO_SUM_TESTS, max_execution_time=2.0
        )
        self.assertEqual(result, [0, 0, 0])

    def test_missing_function_fails(self):
        """The solution compiles but never defines the function the problem asks for."""
        program = "class Solution:\n    def somethingElse(self, nums, target):\n        return [0, 1]\n"
        result, _ = code_utils.get_successful_tests_functional(
            program=program, tests=self.TWO_SUM_TESTS, max_execution_time=2.0
        )
        self.assertEqual(result, [0, 0, 0])

    def test_missing_fn_name_fails(self):
        tests = [{"input": "[3,3]\n6", "output": "[0,1]"}]
        result, _ = code_utils.get_successful_tests_functional(program=self.SOLUTION_CLASS, tests=tests)
        self.assertEqual(result, [0])

    def test_no_tests(self):
        result, runtimes = code_utils.get_successful_tests_functional(program=self.SOLUTION_CLASS, tests=[])
        self.assertEqual(result, [])
        self.assertEqual(runtimes, [])

    def test_partial_pass_rate(self):
        """A solution correct on some tests only: the pass rate must be per-test, not all-or-nothing."""
        program = (
            "class Solution:\n"
            "    def twoSum(self, nums, target):\n"
            "        if nums == [3, 3]:\n"
            "            return [9, 9]\n"
            "        seen = {}\n"
            "        for i, n in enumerate(nums):\n"
            "            if target - n in seen:\n"
            "                return [seen[target - n], i]\n"
            "            seen[n] = i\n"
        )
        result, _ = code_utils.get_successful_tests_functional(
            program=program, tests=self.TWO_SUM_TESTS, max_execution_time=2.0
        )
        self.assertEqual(result, [1, 1, 0])


class GetSuccessfulTestsFunctionalWrappedOutputTests(BaseCodeTestCase):
    """primeintellect/taco wrap the expected return value in a single-element list (not always
    consistently within the same problem), unlike LiveCodeBench's bare encoding. `grade_call_based`
    must accept the value either way -- see the APPS-style unwrap fallback in testing_util.py.
    """

    IS_ANAGRAM_PROGRAM = (
        "def is_anagram(test, original):\n    return sorted(original.lower()) == sorted(test.lower())\n"
    )

    def test_wrapped_output_passes(self):
        tests = [{"input": '"foefet"\n"toffee"', "output": "[true]", "fn_name": "is_anagram"}]
        result, _ = code_utils.get_successful_tests_functional(
            program=self.IS_ANAGRAM_PROGRAM, tests=tests, max_execution_time=2.0
        )
        self.assertEqual(result, [1])

    def test_bare_output_still_passes(self):
        """Not every source wraps consistently, so a bare expected value must still work."""
        program = "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n"
        tests = [{"input": '"the sky is blue"', "output": '"blue is sky the"', "fn_name": "reverse_words"}]
        result, _ = code_utils.get_successful_tests_functional(program=program, tests=tests, max_execution_time=2.0)
        self.assertEqual(result, [1])

    def test_wrapped_wrong_answer_fails(self):
        tests = [{"input": '"foefet"\n"toffee2"', "output": "[true]", "fn_name": "is_anagram"}]
        result, _ = code_utils.get_successful_tests_functional(
            program=self.IS_ANAGRAM_PROGRAM, tests=tests, max_execution_time=2.0
        )
        self.assertEqual(result, [0])

    def test_genuine_single_element_list_answer_passes(self):
        """A prediction that itself returns the (unwrapped) list still matches directly."""
        program = "def duplicate(x):\n    return [x]\n"
        tests = [{"input": "5", "output": "[5]", "fn_name": "duplicate"}]
        result, _ = code_utils.get_successful_tests_functional(program=program, tests=tests, max_execution_time=2.0)
        self.assertEqual(result, [1])

    def test_fallback_is_lenient_for_a_genuine_single_element_list_answer(self):
        """Known tradeoff: since we can't tell "wrapped scalar" from "genuine 1-element list answer"
        apart at grading time, a solution returning the bare scalar is accepted even when the real
        answer is the list itself. This mirrors APPS's own grading behavior.
        """
        program = "def duplicate(x):\n    return x\n"
        tests = [{"input": "5", "output": "[5]", "fn_name": "duplicate"}]
        result, _ = code_utils.get_successful_tests_functional(program=program, tests=tests, max_execution_time=2.0)
        self.assertEqual(result, [1])


class ReliabilityGuardNetworkTests(BaseCodeTestCase):
    """Tests to verify network operations behavior under reliability_guard."""

    def test_socket_operations_allowed(self):
        """Test that socket operations are not blocked by reliability_guard."""
        program = """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Don't actually bind to avoid port conflicts
    socket_created = True
except:
    socket_created = False
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert socket_created == True"], max_execution_time=0.5
        )
        self.assertEqual(result, [1])

    @parameterized.parameterized.expand(
        [
            ("threading_with_socket", "import threading\nimport socket\ns = socket.socket()"),
            ("multiprocessing_with_socket", "from multiprocessing import Pool\nimport socket\ns = socket.socket()"),
        ]
    )
    def test_dangerous_imports_with_networking(self, name, program):
        """Test that certain networking-related imports are blocked."""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert True"], max_execution_time=0.5
        )
        self.assertEqual(result, [0])


class ReliabilityGuardFileSystemTests(BaseCodeTestCase):
    """Tests to ensure file system operations are properly sandboxed."""

    @parameterized.parameterized.expand(
        [
            (
                "file_operations",
                """
try:
    with open('test_file.txt', 'w') as f:
        f.write('test')
    with open('test_file.txt', 'r') as f:
        content = f.read()
    file_ops_work = content == 'test'
except:
    file_ops_work = False
""",
                ["assert file_ops_work == True"],
                1,
            ),
            (
                "os_chdir_disabled",
                """
try:
    import os
    os.chdir('..')
    chdir_works = True
except:
    chdir_works = False
""",
                ["assert chdir_works == False"],
                0,
            ),
        ]
    )
    def test_file_operations_sandboxed(self, name, program, tests, expected):
        """Test that file operations are properly sandboxed."""
        result, _ = code_utils.get_successful_tests_fast(program=program, tests=tests, max_execution_time=0.5)
        self.assertEqual(result, [expected])

    @parameterized.parameterized.expand(
        [
            (
                "os_remove",
                """
try:
    import os
    os.remove('some_file.txt')
    remove_worked = True
except:
    remove_worked = False
""",
            ),
            (
                "os_unlink",
                """
try:
    import os
    os.unlink('some_file.txt')
    unlink_worked = True
except:
    unlink_worked = False
""",
            ),
        ]
    )
    def test_file_deletion_blocked(self, name, program):
        """Test that file deletion operations are blocked."""
        # These will be blocked by should_execute due to "import os"
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert remove_worked == False"], max_execution_time=0.5
        )
        # Programs with "import os" are blocked entirely
        self.assertEqual(result, [0])


class ReliabilityGuardProcessTests(BaseCodeTestCase):
    """Tests to ensure process/subprocess operations are blocked."""

    def test_subprocess_blocked(self):
        """Test that subprocess.Popen is disabled."""
        program = """
import subprocess
try:
    result = subprocess.Popen(['echo', 'hello'], stdout=subprocess.PIPE)
    popen_worked = True
except:
    popen_worked = False
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert popen_worked == False"], max_execution_time=0.5
        )
        self.assertEqual(result, [1])

    def test_os_system_blocked(self):
        """Test that os.system is disabled (but import os is blocked)."""
        # This program will be blocked by should_execute due to "import os"
        program = """
import os
exit_code = os.system('echo hello')
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert True"], max_execution_time=0.5
        )
        self.assertEqual(result, [0])

    def test_os_operations_without_import(self):
        """Test os operations when already imported."""
        # Test that os.system is None when reliability_guard is applied
        program = """
# os is already in builtins from the execution environment
try:
    import os as operating_system
    result = operating_system.system is None
except:
    result = False
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert result == True"], max_execution_time=0.5
        )
        # This will be blocked due to "import os"
        self.assertEqual(result, [0])


class ReliabilityGuardResourceTests(BaseCodeTestCase):
    """Tests to ensure resource exhaustion attacks are prevented."""

    def test_infinite_recursion_protection(self):
        """Test that infinite recursion is caught."""
        program = """
def recursive():
    return recursive()
try:
    recursive()
    recursion_failed = False
except RecursionError:
    recursion_failed = True
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert recursion_failed == True"], max_execution_time=0.5
        )
        self.assertEqual(result, [1])

    def test_cpu_intensive_timeout(self):
        """Test that CPU-intensive operations timeout properly."""
        program = """
# CPU intensive operation
result = 0
for i in range(10**10):
    result += i
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program,
            tests=["assert result > 0"],
            max_execution_time=0.1,  # Very short timeout
        )
        self.assertEqual(result, [0])

    def test_faulthandler_disabled(self):
        """Test that faulthandler is disabled."""
        program = """
import faulthandler
try:
    is_enabled = faulthandler.is_enabled()
except:
    is_enabled = None
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert is_enabled == False"], max_execution_time=0.5
        )
        self.assertEqual(result, [1])


class ReliabilityGuardModuleImportTests(BaseCodeTestCase):
    """Tests to ensure dangerous module imports are blocked."""

    @parameterized.parameterized.expand(
        [
            ("threading", "import threading\nthreading.Thread(target=lambda: None).start()"),
            ("multiprocessing", "from multiprocessing import Pool\nPool()"),
            ("os_system", "import os\nos.system('ls')"),
            ("shutil_rmtree", "import shutil\nshutil.rmtree('/')"),
            ("torch", "import torch\ntensor = torch.zeros(10)"),
            ("sklearn", "from sklearn import datasets\ndata = datasets.load_iris()"),
        ]
    )
    def test_dangerous_imports_blocked(self, name, program):
        """Test that imports of dangerous modules fail."""
        # These are checked by should_execute function
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert True"], max_execution_time=0.5
        )
        self.assertEqual(result, [0])

    @parameterized.parameterized.expand(
        [
            ("math_sqrt", "import math\nresult = math.sqrt(16)", "assert result == 4.0"),
            ("json_dumps", "import json\ndata = json.dumps({'a': 1})", "assert data == '{\"a\": 1}'"),
            ("re_compile", "import re\npattern = re.compile(r'\\d+')", "assert pattern is not None"),
        ]
    )
    def test_safe_imports_allowed(self, name, program, test):
        """Test that safe imports are allowed."""
        result, _ = code_utils.get_successful_tests_fast(program=program, tests=[test], max_execution_time=0.5)
        self.assertEqual(result, [1])


class ReliabilityGuardEdgeCaseTests(BaseCodeTestCase):
    """Tests for edge cases and combined attack vectors."""

    def test_builtins_modifications(self):
        """Test that certain builtins are modified."""
        program = """
# Check if exit and quit are disabled
exit_disabled = exit is None
quit_disabled = quit is None
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program,
            tests=["assert exit_disabled == True", "assert quit_disabled == True"],
            max_execution_time=0.5,
        )
        self.assertEqual(result, [1, 1])

    def test_working_directory_sandboxed(self):
        """Test that working directory is changed to cache."""
        program = """
# Since os.getcwd is None, we can't directly check the directory
# But we can verify that we can create files in the current directory
try:
    with open('sandbox_test.txt', 'w') as f:
        f.write('test')
    with open('sandbox_test.txt', 'r') as f:
        content = f.read()
    sandboxed = content == 'test'
except:
    sandboxed = False
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert sandboxed == True"], max_execution_time=0.5
        )
        self.assertEqual(result, [1])

    def test_shutil_operations_blocked(self):
        """Test that shutil operations are blocked."""
        # Note: "shutil" in program triggers should_execute to block it
        program = """
import shutil
shutil.rmtree('/tmp/test')
"""
        result, _ = code_utils.get_successful_tests_fast(
            program=program, tests=["assert True"], max_execution_time=0.5
        )
        self.assertEqual(result, [0])


if __name__ == "__main__":
    unittest.main()
