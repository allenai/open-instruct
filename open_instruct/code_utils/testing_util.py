"""
copied directly from livecodebench
https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py
"""

import ast
import faulthandler

# to run the solution files we're using a timing based approach
import signal
import sys
import time

# used for debugging to time steps
from decimal import Decimal
from enum import Enum
from io import StringIO

# from pyext import RuntimeModule
from types import ModuleType

# used for testing the code that reads from input
from unittest.mock import mock_open, patch

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    print("timeout occured: alarm went off")
    raise TimeoutException


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# Custom mock for sys.stdin that supports buffer attribute
class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        # Delegate other attributes to StringIO
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")  # Convert to bytes

    def read(self, *args):
        # Return as byte strings that can be split
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
    except Exception:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, ast.Import | ast.ImportFrom):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception:
        return code


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # Create custom stdin mock with buffer support
    mock_stdin = MockStdinWithBuffer(inputs)

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)  # Use our custom mock instead of StringIO
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        compiled_sol = tmp_sol.Solution() if "class Solution" in code else tmp_sol

        assert compiled_sol is not None
    except Exception:
        compiled_sol = None
    finally:
        signal.alarm(0)

    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except Exception:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    # you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def grade_stdio(code: str, all_inputs: list, all_outputs: list, timeout: int):
    signal.signal(signal.SIGALRM, timeout_handler)
    # runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)

    # we wrap the given code inside another function
    code = make_function(code)
    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return ([-1] * len(all_outputs), [-1.0] * len(all_outputs))

    method = get_function(compiled_sol, "wrapped_function")
    if method is None:
        return ([-1] * len(all_outputs), [-1.0] * len(all_outputs))

    all_results = []
    all_runtimes = []
    total_execution_time = 0
    first_failure_info = None
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()

        prediction = None
        try:
            with Capturing() as captured_output:
                start = time.time()
                call_method(method, gt_inp)
                end_time = time.time()
                total_execution_time += end_time - start
                all_runtimes.append(end_time - start)
                # reset the alarm
            signal.alarm(0)
            prediction = captured_output[0] if captured_output else ""

        except Exception as e:
            signal.alarm(0)
            all_runtimes.append(-1.0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                if not first_failure_info:
                    first_failure_info = {"error_code": -3, "error_message": "Time Limit Exceeded"}
                continue
            else:
                all_results.append(-4)
                if not first_failure_info:
                    first_failure_info = {"error_code": -4, "error_message": f"Runtime Error: {e}"}
                continue
        finally:
            signal.alarm(0)
            faulthandler.disable()

        if prediction is None:
            # Should not happen if there was no exception, but as a safeguard.
            all_results.append(-4)  # Runtime error
            if not first_failure_info:
                first_failure_info = {"error_code": -4, "error_message": "Runtime Error: No output produced"}
            continue

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            if not first_failure_info:
                first_failure_info = {"error_code": -2, "error_message": "Wrong answer: mismatched output length"}
            continue

        test_case_failed = False
        for stripped_prediction_line, stripped_gt_out_line in zip(stripped_prediction_lines, stripped_gt_out_lines):
            if stripped_prediction_line == stripped_gt_out_line:
                continue

            success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
            if not success:
                test_case_failed = True
                break
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                test_case_failed = True
                break

            if decimal_prediction_line == decimal_gtout_line:
                continue

            test_case_failed = True
            break

        if test_case_failed:
            all_results.append(-2)
            if not first_failure_info:
                first_failure_info = {"error_code": -2, "error_message": "Wrong Answer"}
            continue

        all_results.append(True)

    if first_failure_info:
        print(f"first_failure_info: {first_failure_info}")
        return all_results, all_runtimes

    return all_results, all_runtimes
