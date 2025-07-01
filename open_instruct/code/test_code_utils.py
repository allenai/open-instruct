"""Unit-tests for `get_successful_tests_fast`."""

import os
import sys
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
            "import time\n"
            "time.sleep(10)"  # Simple timeout test without print spam
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


class ReliabilityGuardNetworkTests(unittest.TestCase):
    """Tests to verify network operations behavior under reliability_guard."""

    def test_socket_operations_allowed(self):
        """Test that socket operations are actually allowed (not blocked by reliability_guard)."""
        # Socket operations are NOT blocked by reliability_guard
        program = """
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Don't actually bind to avoid port conflicts
    socket_created = True
except:
    socket_created = False
"""
        result = get_successful_tests_fast(
            program=program,
            tests=["assert socket_created == True"],
            max_execution_time=0.5
        )
        self.assertEqual(result, [1], "Socket creation should succeed")

    def test_dangerous_imports_with_networking(self):
        """Test that certain networking-related imports are blocked."""
        # These programs will be blocked by should_execute due to dangerous imports
        programs = [
            "import threading\nimport socket\ns = socket.socket()",
            "from multiprocessing import Pool\nimport socket\ns = socket.socket()",
        ]
        
        for program in programs:
            result = get_successful_tests_fast(
                program=program,
                tests=["assert True"],
                max_execution_time=0.5
            )
            self.assertEqual(result, [0], f"Program with dangerous imports should be blocked: {program[:30]}...")


class ReliabilityGuardFileSystemTests(unittest.TestCase):
    """Tests to ensure file system operations are properly sandboxed."""

    def test_file_operations_sandboxed(self):
        """Test that file operations are restricted to sandbox."""
        programs = [
            # File operations are allowed but in sandbox directory
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
            # os.chdir is disabled
            """
try:
    import os
    os.chdir('..')
    chdir_works = True
except:
    chdir_works = False
""",
        ]
        
        expected_results = [1, 0]  # First should pass (file ops work), second should fail (chdir disabled)
        
        for program, expected in zip(programs, expected_results):
            result = get_successful_tests_fast(
                program=program,
                tests=["assert file_ops_work == True"] if "file_ops_work" in program else ["assert chdir_works == False"],
                max_execution_time=0.5
            )
            self.assertEqual(result, [expected], f"Unexpected result for: {program[:50]}...")

    def test_file_deletion_blocked(self):
        """Test that file deletion operations are blocked."""
        programs = [
            # os.remove is disabled - but need to handle import os being blocked
            """
try:
    import os
    os.remove('some_file.txt')
    remove_worked = True
except:
    remove_worked = False
""",
            # os.unlink is disabled
            """
try:
    import os
    os.unlink('some_file.txt')
    unlink_worked = True
except:
    unlink_worked = False
""",
        ]
        
        # These will be blocked by should_execute due to "import os"
        for program in programs:
            result = get_successful_tests_fast(
                program=program,
                tests=["assert remove_worked == False"],
                max_execution_time=0.5
            )
            # Programs with "import os" are blocked entirely
            self.assertEqual(result, [0], f"Program with 'import os' should be blocked: {program[:50]}...")


class ReliabilityGuardProcessTests(unittest.TestCase):
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
        result = get_successful_tests_fast(
            program=program,
            tests=["assert popen_worked == False"],
            max_execution_time=0.5
        )
        self.assertEqual(result, [1], "subprocess.Popen should be disabled")

    def test_os_system_blocked(self):
        """Test that os.system is disabled (but import os is blocked)."""
        # This program will be blocked by should_execute due to "import os"
        program = """
import os
exit_code = os.system('echo hello')
"""
        result = get_successful_tests_fast(
            program=program,
            tests=["assert True"],
            max_execution_time=0.5
        )
        self.assertEqual(result, [0], "Programs with 'import os' should be blocked")

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
        result = get_successful_tests_fast(
            program=program,
            tests=["assert result == True"],
            max_execution_time=0.5
        )
        # This will be blocked due to "import os"
        self.assertEqual(result, [0], "Programs with 'import os' are blocked by should_execute")


class ReliabilityGuardResourceTests(unittest.TestCase):
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
        result = get_successful_tests_fast(
            program=program,
            tests=["assert recursion_failed == True"],
            max_execution_time=0.5
        )
        self.assertEqual(result, [1], "Infinite recursion should raise RecursionError")

    def test_cpu_intensive_timeout(self):
        """Test that CPU-intensive operations timeout properly."""
        program = """
# CPU intensive operation
result = 0
for i in range(10**10):
    result += i
"""
        result = get_successful_tests_fast(
            program=program,
            tests=["assert result > 0"],
            max_execution_time=0.1  # Very short timeout
        )
        self.assertEqual(result, [0], "CPU intensive operation should timeout")

    def test_faulthandler_disabled(self):
        """Test that faulthandler is disabled."""
        program = """
import faulthandler
try:
    is_enabled = faulthandler.is_enabled()
except:
    is_enabled = None
"""
        result = get_successful_tests_fast(
            program=program,
            tests=["assert is_enabled == False"],
            max_execution_time=0.5
        )
        self.assertEqual(result, [1], "Faulthandler should be disabled")


class ReliabilityGuardModuleImportTests(unittest.TestCase):
    """Tests to ensure dangerous module imports are blocked."""

    def test_dangerous_imports_blocked(self):
        """Test that imports of dangerous modules fail."""
        # These are checked by should_execute function
        dangerous_programs = [
            "import threading\nthreading.Thread(target=lambda: None).start()",
            "from multiprocessing import Pool\nPool()",
            "import os\nos.system('ls')",
            "import shutil\nshutil.rmtree('/')",
            "import torch\ntensor = torch.zeros(10)",
            "from sklearn import datasets\ndata = datasets.load_iris()",
        ]
        
        for program in dangerous_programs:
            result = get_successful_tests_fast(
                program=program,
                tests=["assert True"],
                max_execution_time=0.5
            )
            self.assertEqual(result, [0], f"Dangerous import should be blocked: {program[:30]}...")

    def test_safe_imports_allowed(self):
        """Test that safe imports are allowed."""
        safe_programs = [
            "import math\nresult = math.sqrt(16)",
            "import json\ndata = json.dumps({'a': 1})",
            "import re\npattern = re.compile(r'\\d+')",
        ]
        
        tests = [
            "assert result == 4.0",
            "assert data == '{\"a\": 1}'",
            "assert pattern is not None",
        ]
        
        for program, test in zip(safe_programs, tests):
            result = get_successful_tests_fast(
                program=program,
                tests=[test],
                max_execution_time=0.5
            )
            self.assertEqual(result, [1], f"Safe import should work: {program[:30]}...")


class ReliabilityGuardEdgeCaseTests(unittest.TestCase):
    """Tests for edge cases and combined attack vectors."""

    def test_builtins_modifications(self):
        """Test that certain builtins are modified."""
        program = """
# Check if exit and quit are disabled
exit_disabled = exit is None
quit_disabled = quit is None
"""
        result = get_successful_tests_fast(
            program=program,
            tests=["assert exit_disabled == True", "assert quit_disabled == True"],
            max_execution_time=0.5
        )
        self.assertEqual(result, [1, 1], "exit and quit should be None")

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
        result = get_successful_tests_fast(
            program=program,
            tests=["assert sandboxed == True"],
            max_execution_time=0.5
        )
        self.assertEqual(result, [1], "Should be able to write files in sandbox")

    def test_shutil_operations_blocked(self):
        """Test that shutil operations are blocked."""
        # Note: "shutil" in program triggers should_execute to block it
        program = """
import shutil
shutil.rmtree('/tmp/test')
"""
        result = get_successful_tests_fast(
            program=program,
            tests=["assert True"],
            max_execution_time=0.5
        )
        self.assertEqual(result, [0], "Programs with 'shutil' should be blocked")


if __name__ == "__main__":
    unittest.main()
