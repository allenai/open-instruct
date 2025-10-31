import base64
import json
import math
import multiprocessing
import os
import pickle
import shutil
import sys
import time
import zlib
from typing import Any

from open_instruct import logger_utils

from .testing_util import grade_stdio

# taken from https://github.com/TIGER-AI-Lab/AceCoder/blob/62bb7fc25d694fed04a5270c89bf2cdc282804f7/data/inference/EvaluateInferencedCode.py#L372
# DANGEROUS_MODULES = ["os", "sys", "shutil", "subprocess", "socket", "urllib", "requests", "pathlib", "glob", "cgi", "cgitb", "xml", "pickle", "eval", "exec"]
# we save the current working directory and restore them later
cwd = os.getcwd()
cache_wd = cwd + "/cache"

tmp_chmod = os.chmod
tmp_fchmod = os.fchmod
tmp_chdir = os.chdir
tmp_rmdir = os.rmdir
tmp_getcwd = os.getcwd
# tmp_open = open
tmp_print = print
tmp_rm_tree = shutil.rmtree
tmp_unlink = os.unlink

logger = logger_utils.setup_logger(__name__)


# -------------------------------------------------------------
# The slow but  accurate version
# -------------------------------------------------------------
original_builtins = __builtins__


def encode_tests(tests: list) -> str:
    if not tests:
        return ""
    pickled_data = pickle.dumps(tests)
    compressed_data = zlib.compress(pickled_data)
    b64_encoded_data = base64.b64encode(compressed_data)
    return b64_encoded_data.decode("utf-8")


def decode_tests(tests: Any) -> list:
    if not tests:
        return []
    if isinstance(tests, list):
        return tests
    if isinstance(tests, str):
        try:
            # First, try to decode as a plain JSON string
            return json.loads(tests)
        except json.JSONDecodeError:
            # If that fails, try to decode from the compressed format
            try:
                b64_decoded = base64.b64decode(tests.encode("utf-8"))

                # Use a streaming decompressor to handle potentially very large test cases
                # without allocating a massive buffer upfront. This is more memory-efficient.
                decompressor = zlib.decompressobj()
                decompressed_chunks = []
                total_decompressed_size = 0

                # Process in chunks to avoid holding the entire decompressed data in memory at once.
                chunk_size = 256 * 1024  # 256KB chunks
                for i in range(0, len(b64_decoded), chunk_size):
                    chunk = b64_decoded[i : i + chunk_size]
                    decompressed_chunk = decompressor.decompress(chunk)
                    total_decompressed_size += len(decompressed_chunk)
                    decompressed_chunks.append(decompressed_chunk)

                decompressed_chunks.append(decompressor.flush())

                decompressed_data = b"".join(decompressed_chunks)
                return pickle.loads(decompressed_data)
            except Exception:
                # Log the problematic data before returning an empty list
                logger.error(f"Failed to decode test data: {tests}")
                return []
    return []


# -------------------------------------------------------------
# The fast but more readable version
# -------------------------------------------------------------


def run_tests_against_program_helper_2(func: str, tests: list[str], shared_results) -> None:
    """Run all tests against the program and store results in shared array"""
    # Apply reliability guard in the child process
    reliability_guard()

    try:
        execution_context: dict[str, Any] = {}
        execution_context.update({"__builtins__": __builtins__})

        try:
            exec(func, execution_context)
        except Exception:
            for i in range(len(tests)):
                shared_results[i] = 0
            return

        for idx, test in enumerate(tests):
            try:
                exec(test, execution_context)
                shared_results[idx] = 1
            except Exception:
                shared_results[idx] = 0
    finally:
        # Restore in child process (though it will terminate anyway)
        partial_undo_reliability_guard()


def run_individual_test_helper(func: str, test: str, result_array, index: int, runtimes_array) -> None:
    """Run a single test and store result in shared array at given index"""
    # Apply reliability guard in the child process
    reliability_guard()

    try:
        execution_context = {}
        execution_context.update({"__builtins__": __builtins__})
        try:
            exec(func, execution_context)
            start_time = time.time()
            exec(test, execution_context)
            end_time = time.time()
            result_array[index] = 1
            runtimes_array[index] = end_time - start_time
        except Exception:
            result_array[index] = 0
            runtimes_array[index] = -1.0
    finally:
        # Restore in child process (though it will terminate anyway)
        partial_undo_reliability_guard()


def get_successful_tests_fast(
    program: str, tests: list[str], max_execution_time: float = 1.0
) -> tuple[list[int], list[float]]:
    """Run a program against a list of tests, if the program exited successfully then we consider
    the test to be passed. Note that you SHOULD ONLY RUN THIS FUNCTION IN A VIRTUAL ENVIRONMENT
    as we do not guarantee the safety of the program provided.

    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of assert statements which are considered to be the test cases
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated

    Return:
        a tuple of (results, runtimes). results is a list of 0/1 indicating
        passed or not, runtimes is a list of execution times for each test."""
    test_ct = len(tests)
    if test_ct == 0:
        return [], []
    if not should_execute(program=program, tests=tests):
        logger.info("Not executing program %s", program)
        return [0] * len(tests), [-1.0] * len(tests)

    # Run each test individually to handle timeouts properly
    shared_test_results = multiprocessing.Array("i", len(tests))
    shared_runtimes = multiprocessing.Array("d", len(tests))

    # Initialize results
    for i in range(len(tests)):
        shared_test_results[i] = 0
        shared_runtimes[i] = -1.0

    # Run each test in its own process
    for idx, test in enumerate(tests):
        p = multiprocessing.Process(
            target=run_individual_test_helper, args=(program, test, shared_test_results, idx, shared_runtimes)
        )
        p.start()
        p.join(timeout=max_execution_time)
        if p.is_alive():
            p.kill()
            p.join()
        p.close()

    return [shared_test_results[i] for i in range(len(tests))], [shared_runtimes[i] for i in range(len(tests))]


# -------------------------------------------------------------
# Stdio format - mostly copied from livecodebench
# -------------------------------------------------------------


def run_tests_stdio_helper(
    program: str, tests: list[Any], max_execution_time: float, stdio_test_results, stdio_runtimes
):
    """Helper to run stdio tests in a separate process."""
    reliability_guard()
    try:
        all_inputs = [test["input"] for test in tests]
        all_outputs = [test["output"] for test in tests]
        timeout = math.ceil(max_execution_time)
        results, runtimes = grade_stdio(program, all_inputs, all_outputs, timeout)

        if results is not None:
            processed_results = [1 if r is True else 0 for r in results]
            for i, res in enumerate(processed_results):
                if i < len(stdio_test_results):
                    stdio_test_results[i] = res
                    stdio_runtimes[i] = runtimes[i]
    except Exception:
        # On any failure, results in the shared array will remain as they were initialized (0), indicating failure.
        pass
    finally:
        partial_undo_reliability_guard()


def get_successful_tests_stdio(
    program: str, tests: list[Any], max_execution_time: float = 1.0
) -> tuple[list[int], list[float]]:
    """Same as above but for stdio format.
    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of (input, output) pairs
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated
    Return:
        a tuple of (results, runtimes). results is a list of 0/1 indicating
        passed or not, runtimes is a list of execution times for each test.
    """
    test_ct = len(tests)
    if test_ct == 0:
        return [], []
    if not should_execute(program=program, tests=tests):
        logger.info("Not executing program %s", program)
        return [0] * len(tests), [-1.0] * len(tests)

    stdio_test_results = multiprocessing.Array("i", test_ct)
    stdio_runtimes = multiprocessing.Array("d", test_ct)

    for i in range(test_ct):
        stdio_test_results[i] = 0  # Initialize results to 0 (failure)
        stdio_runtimes[i] = -1.0

    # Total timeout needs to account for all tests running sequentially.
    total_timeout = max_execution_time * test_ct + 5.0

    p = multiprocessing.Process(
        target=run_tests_stdio_helper, args=(program, tests, max_execution_time, stdio_test_results, stdio_runtimes)
    )
    p.start()
    p.join(timeout=total_timeout)

    if p.is_alive():
        p.kill()
        p.join()
    p.close()

    return [stdio_test_results[i] for i in range(test_ct)], [stdio_runtimes[i] for i in range(test_ct)]


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------


def should_execute(program: str, tests: list[Any]) -> bool:
    """Determine if we should try to execute this program at all for safety
    reasons."""
    dangerous_commands = [
        "threading",
        "multiprocess",
        "multiprocessing",
        "import os",
        "from os",
        "shutil",
        "import torch",
        "from torch",
        "import sklearn",
        "from sklearn",
    ]
    return all(comm not in program for comm in dangerous_commands)


# -------------------------------------------------------------
# For safety handling
# -------------------------------------------------------------


def partial_undo_reliability_guard():
    """Undo the chmod, fchmod, print and open operation"""
    import builtins

    os.chmod = tmp_chmod
    os.fchmod = tmp_fchmod
    os.chdir = tmp_chdir
    os.unlink = tmp_unlink
    os.rmdir = tmp_rmdir
    os.getcwd = tmp_getcwd
    # shutil.rmtree = tmp_rmtree
    # builtins.open = tmp_open
    builtins.print = tmp_print

    # restore working directory
    os.chdir(cwd)
    # shutil.rmtree(cache_wd)
    shutil.rmtree = tmp_rm_tree


def reliability_guard(maximum_memory_bytes: int | None = None):
    """
    This function is copied from https://github.com/openai/human-eval/blob/master/human_eval/execution.py.
    It disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """
    import faulthandler
    import platform

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None
    # builtins.open = None
    # builtins.print = lambda *args, **kwargs: None

    import os

    # we save the current working directory and restore them later
    os.makedirs(cache_wd, exist_ok=True)
    os.chdir(cache_wd)

    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    # os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    # __builtins__['help'] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
