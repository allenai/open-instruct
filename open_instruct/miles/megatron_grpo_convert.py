"""Check the pinned native switch that preserves PP1 during TP conversion."""

import ast
import contextlib
import hashlib
import os
import signal
import subprocess


def verify_keep_pp1(source, filename):
    """Fail closed if the native converter does not honor CONVERT_KEEP_PP1."""
    tree = ast.parse(source, filename=filename)
    expected = ast.dump(
        ast.parse(
            'args.pipeline_model_parallel_size == 1 and world_size > 1 and not os.environ.get("CONVERT_KEEP_PP1")',
            mode="eval",
        ).body
    )
    matches = [
        node
        for function in tree.body
        if isinstance(function, ast.FunctionDef) and function.name == "get_args"
        for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.dump(node.test) == expected
    ]
    if len(matches) != 1:
        raise ValueError("Native converter CONVERT_KEEP_PP1 guard differs; inspect the image source before conversion")
    return hashlib.sha256(source.encode()).hexdigest()


def run_conversion(command, environment, stream, *, timeout=1800):
    """Bound conversion and terminate its whole torchrun tree on failure.

    subprocess.run kills only the torchrun parent on timeout; its distributed
    workers can keep the Beaker allocation alive after the workflow fails.
    """
    process = subprocess.Popen(
        command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        returncode = process.wait(timeout=timeout)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        finally:
            # A terminated torchrun parent is not evidence that its workers exited.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        raise
