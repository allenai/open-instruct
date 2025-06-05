import logging
import multiprocessing
import os
import shutil
from ctypes import c_int
from typing import Any, Dict, List, Optional

# taken from https://github.com/TIGER-AI-Lab/AceCoder/blob/62bb7fc25d694fed04a5270c89bf2cdc282804f7/data/inference/EvaluateInferencedCode.py#L372
# we save the current working directory and restore them later
cwd = os.getcwd()
cache_wd = cwd + "/cache"

tmp_chmod = os.chmod
tmp_fchmod = os.fchmod
tmp_chdir = os.chdir
tmp_rmdir = os.rmdir
# tmp_open = open
tmp_print = print
tmp_rm_tree = shutil.rmtree
tmp_unlink = os.unlink

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# The slow but  accurate version
# -------------------------------------------------------------
return_var = multiprocessing.Value(c_int, 0)


def run_single_test_against_program_helper(func: str, test: str) -> int:
    """Return 1 if func finish running, 0 otherwise"""
    execution_context = {}
    execution_context.update({"__builtins__": __builtins__})
    try:
        exec(func, execution_context)
        exec(test, execution_context)
        return_var.value = 1
        return 1
    except Exception:
        return_var.value = 0
        return 0


# very unstable, seems to work, seeing there are no uses will be deprecated for now.
def get_successful_tests_slow(program: str, tests: List[str], max_execution_time: float = 1.0) -> List[int]:
    """Run a program against a list of tests, if the program exited successfully then we consider
    the test to be passed. Note that you SHOULD ONLY RUN THIS FUNCTION IN A VIRTUAL ENVIRONMENT
    as we do not gurantee the safety of the program provided.

    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of assert statements which are considered to be the test cases
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated

    Return:
        a list of 0/1 indicating passed or not"""
    test_ct = len(tests)
    if test_ct == 0:
        return []
    if not should_execute(program=program, tests=tests):
        return [0] * len(tests)

    reliability_guard()
    result = []
    for test in tests:
        return_var.value = 0
        p = multiprocessing.Process(target=run_single_test_against_program_helper, args=(program, test))
        p.start()
        p.join(timeout=max_execution_time)
        if p.is_alive():
            p.kill()
        result.append(return_var.value)

    partial_undo_reliability_guard()
    return result


# -------------------------------------------------------------
# The fast but more readable version
# -------------------------------------------------------------
test_results = multiprocessing.Array("i", 1000)  # Support up to 1000 tests


def run_tests_against_program_helper_2(func: str, tests: List[str]) -> None:
    """Run all tests against the program and store results in shared array"""
    execution_context: Dict[str, Any] = {}
    execution_context.update({"__builtins__": __builtins__})

    try:
        exec(func, execution_context)
    except Exception:
        for i in range(len(tests)):
            test_results[i] = 0
        return

    for idx, test in enumerate(tests):
        try:
            exec(test, execution_context)
            test_results[idx] = 1
        except Exception:
            test_results[idx] = 0


def get_successful_tests_fast(program: str, tests: List[str], max_execution_time: float = 1.0) -> List[int]:
    """Run a program against a list of tests, if the program exited successfully then we consider
    the test to be passed. Note that you SHOULD ONLY RUN THIS FUNCTION IN A VIRTUAL ENVIRONMENT
    as we do not guarantee the safety of the program provided.

    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of assert statements which are considered to be the test cases
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated

    Return:
        a list of 0/1 indicating passed or not"""
    test_ct = len(tests)
    if test_ct == 0:
        return []
    if not should_execute(program=program, tests=tests):
        logger.info("Not executing program %s", program)
        return [0] * len(tests)

    reliability_guard()

    # Reset test results
    for i in range(len(tests)):
        test_results[i] = 0

    p = multiprocessing.Process(target=run_tests_against_program_helper_2, args=(program, tests))
    p.start()
    p.join(timeout=max_execution_time)
    if p.is_alive():
        p.kill()

    partial_undo_reliability_guard()

    return [test_results[i] for i in range(len(tests))]


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------


def should_execute(program: str, tests: List[str]) -> bool:
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
    for comm in dangerous_commands:
        if comm in program:
            return False  # assume the program fails
    return True


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
    # shutil.rmtree = tmp_rmtree
    # builtins.open = tmp_open
    builtins.print = tmp_print

    # restore working directory
    os.chdir(cwd)
    # shutil.rmtree(cache_wd)
    shutil.rmtree = tmp_rm_tree


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
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
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None
    # builtins.open = None
    builtins.print = lambda *args, **kwargs: None

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

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


if __name__ == "__main__":
    from datasets import load_dataset

    # for testing purpose
    program = "a = 1"
    bad_test = "assert False"
    good_test = "assert True"
    time_out_test = """
for i in range(9999999999999999999):
    for k in range(99999999999999999999):
        print("hello world")
"""
    test_case_status = get_successful_tests_fast(
        program=program,
        tests=[
            bad_test,
            good_test,
            bad_test,
            good_test,
            good_test,
            time_out_test,
            good_test,
        ],
    )
    print(test_case_status)
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
    print(test_case_status)

    ds = load_dataset("TIGER-Lab/AceCode-87K", split="train")
    i = 1
    print(ds[i]["question"])
    print(ds[i]["inferences"][-1]["completion"])
    print(ds[i]["inferences"][-1]["model_name"], ds[i]["inferences"][-1]["pass_rate"])
    print("=" * 100)
    for item in ds[i]["test_cases"]:
        print(item)
    print("=" * 100)
    test_case_status = get_successful_tests_fast(
        program=ds[i]["inferences"][-1]["completion"],
        tests=ds[i]["test_cases"],
    )
    print(test_case_status)

    test_case_status = get_successful_tests_fast(
        program="\ndef add(a, b):\n    return a + b\n",
        tests=["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"],
    )
    print(test_case_status)
