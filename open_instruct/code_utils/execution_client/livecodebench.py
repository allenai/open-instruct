# Client-side code ported from code-execution-service revision 103ca279b35e8e81d118947f82b72c22676c6540.
"""Explicit LiveCodeBench-shaped labels for the existing assert/stdio wire API."""

import inspect
import json
import keyword

from open_instruct.code_utils.execution_client import nemo_comparison, protocol


def prepare_nemotron_tests(inputs, outputs, fn_name=None):
    """Dataset-owned NeMo semantics; no training configuration or tolerance knobs."""
    return prepare_tests(inputs, outputs, fn_name, comparison="nemo_v1")


def _json(value):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    if not isinstance(value, str):
        raise ValueError("function inputs and outputs must contain JSON strings")
    return json.loads(value, parse_constant=reject_constant)


def prepare_tests(inputs, outputs, fn_name=None, *, comparison="exact"):
    """Return (mode, tests), preserving every input/output pair in order.

    Without fn_name, inputs are stdin strings (or lists of lines), outputs stdout
    strings. With fn_name, each input is one JSON value per line, each line a
    positional argument; output is one JSON value. Invoke Solution().fn_name when
    Solution is defined, otherwise the module-level function. Compare using Python
    equality after a JSON round trip (tuples therefore normalize to lists).
    """
    if comparison not in ("exact", "nemo_v1"):
        raise ValueError("unknown LiveCodeBench comparison")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise ValueError("inputs and outputs must be equal-length lists")
    if not inputs:
        raise ValueError("LiveCodeBench labels must contain at least one test")
    if fn_name is None:
        tests = [{"input": i, "output": o} for i, o in zip(inputs, outputs, strict=True)]
        if comparison == "nemo_v1":
            tests = [{**test, "comparison": "nemo_v1"} for test in tests]
        protocol.validate_tests("stdio", tests)
        return "stdio", tests
    if not isinstance(fn_name, str) or not fn_name.isidentifier() or keyword.iskeyword(fn_name):
        raise ValueError("fn_name must be a Python identifier or null")
    tests = []
    for source, expected in zip(inputs, outputs, strict=True):
        if not isinstance(source, str):
            raise ValueError("function input must be a string of JSON arguments, one per line")
        args = [_json(line) for line in source.splitlines()] if source else []
        want = _json(expected)
        arguments_json = json.dumps(args, allow_nan=False)
        expected_json = json.dumps(want, allow_nan=False)
        setup = ""
        assertion = (
            "assert (__verifier_json.loads(__verifier_json.dumps(__verifier_actual, allow_nan=False))\n"
            "        == __verifier_expected)"
        )
        if comparison == "nemo_v1":
            comparison_source = (
                "import json, math\n"
                + inspect.getsource(nemo_comparison._nemo_values_close)
                + "\n"
                + inspect.getsource(nemo_comparison._nemo_call_outputs_match)
                + "\n"
            )
            setup = (
                "__verifier_comparison_namespace = {}\n"
                f"exec({comparison_source!r}, __verifier_comparison_namespace)\n"
            )
            assertion = (
                "assert __verifier_comparison_namespace['_nemo_call_outputs_match']"
                "(__verifier_expected, __verifier_actual)"
            )
        tests.append(
            setup + "import json as __verifier_json\n"
            f"__verifier_args = __verifier_json.loads({arguments_json!r})\n"
            f"__verifier_expected = __verifier_json.loads({expected_json!r})\n"
            f"__verifier_fn = (getattr(Solution(), {fn_name!r})\n"
            f"                 if 'Solution' in globals() else globals()[{fn_name!r}])\n"
            "__verifier_actual = __verifier_fn(*__verifier_args)\n" + assertion
        )
    protocol.validate_tests("assert", tests)
    return "assert", tests
