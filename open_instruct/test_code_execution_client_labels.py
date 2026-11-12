"""Client-side generated tests preserve the dataset's comparison semantics."""

import pytest

from open_instruct.code_utils.execution_client import livecodebench, protocol


@pytest.mark.parametrize("actual,expected,passes", [(1.0000005, "1.0", True), (1.01, "1.0", False)])
def test_function_label_conversion_preserves_nemotron_tolerance(actual, expected, passes):
    mode, tests = livecodebench.prepare_nemotron_tests(["0"], [expected], "f")
    assert mode == "assert" and len(tests) == 1
    namespace = {"f": lambda _: actual}
    if passes:
        exec(tests[0], namespace)
    else:
        with pytest.raises(AssertionError):
            exec(tests[0], namespace)


def test_stdio_label_conversion_preserves_order_and_comparison():
    mode, tests = livecodebench.prepare_nemotron_tests(["a", "b"], ["A", "B"])
    assert mode == "stdio"
    assert tests == [
        {"input": "a", "output": "A", "comparison": "nemo_v1"},
        {"input": "b", "output": "B", "comparison": "nemo_v1"},
    ]
    protocol.validate_tests(mode, tests)


def test_unknown_comparison_is_rejected_before_transport():
    with pytest.raises(protocol.BadRequest, match="comparison"):
        protocol.validate_tests("stdio", [{"input": "", "output": "", "comparison": "unknown"}])
