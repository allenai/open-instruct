"""Bounded HF -> Megatron torch_dist learner conversion on the trainer's TP-only mesh."""

import ast
import hashlib
import subprocess
from pathlib import Path

from open_instruct.miles.distillation import opd_runtime

# The operands of the pinned converter's automatic pipeline-parallel guard that must remain for the
# conversion to stay PP1: an explicit PP1 request, a multi-rank world and the CONVERT_KEEP_PP1 escape.
_KEEP_PP1_OPERANDS = (
    "args.pipeline_model_parallel_size == 1",
    "world_size > 1",
    'not os.environ.get("CONVERT_KEEP_PP1")',
)


def conversion_command(executable, miles_root, architecture, model, checkpoint, tensor_parallel):
    """Keep conversion on the same TP-only mesh as the GRPO trainer.

    The native converter may infer pipeline parallelism from the world size even when PP1 is
    explicit; the runtime verifies the native CONVERT_KEEP_PP1 escape and sets it.
    """
    return [
        executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={tensor_parallel}",
        str(Path(miles_root) / "tools/convert_hf_to_torch_dist.py"),
        *architecture,
        "--hf-checkpoint",
        str(model),
        "--save",
        str(checkpoint),
        "--tensor-model-parallel-size",
        str(tensor_parallel),
        "--pipeline-model-parallel-size",
        "1",
    ]


def _keep_pp1_guards(function):
    """Conjunctions in ``function`` holding every ``_KEEP_PP1_OPERANDS`` operand."""
    required = {ast.dump(ast.parse(operand, mode="eval").body) for operand in _KEEP_PP1_OPERANDS}
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.BoolOp)
        and isinstance(node.op, ast.And)
        and required <= {ast.dump(value) for value in node.values}
    ]


def _gates_a_branch(function, guard):
    """True when ``guard`` is an ``if`` test, or is bound to a name that an ``if`` tests."""
    names = {
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign) and node.value is guard
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    for node in ast.walk(function):
        if isinstance(node, ast.If) and (
            node.test is guard or (isinstance(node.test, ast.Name) and node.test.id in names)
        ):
            return True
    return False


def verify_keep_pp1(source, filename):
    """Fail closed if the native converter's PP inference does not honor CONVERT_KEEP_PP1.

    Accepts the guard as a direct ``if`` test (older converters) or as a named conjunction with
    extra parallelism operands (the pinned converter only infers PP when TP=CP=EP=ETP=1). Returns
    the converter's SHA256.
    """
    tree = ast.parse(source, filename=filename)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_args"]
    guards = [
        (function, guard)
        for function in functions
        for guard in _keep_pp1_guards(function)
        if _gates_a_branch(function, guard)
    ]
    if len(guards) != 1:
        raise ValueError("Native converter CONVERT_KEEP_PP1 guard differs; inspect the image source before conversion")
    return hashlib.sha256(source.encode()).hexdigest()


def run_conversion(command, environment, stream, *, timeout=1800):
    """Bound conversion and terminate its whole torchrun tree afterwards (see ``opd_runtime.run_group``)."""
    opd_runtime.run_group(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
