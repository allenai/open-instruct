"""General math grading regressions, including compact saved-rollout examples."""

import asyncio
import time
from unittest import mock

import pytest
from math_verify import errors, grader, parser

from open_instruct import ground_truth_utils, math_verify_utils
from open_instruct.miles.rewards import rewards


@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        ("4", r"\boxed{5}", 0),
        ("0.5", r"\boxed{\frac{1}{2}}", 1),
        ("4", "Answer: 4", 1),
        ("4", "Final Answer: The final answer is 4. I hope it is correct.", 1),
        ("4", "$4$", 1),
        ("4", "4", 1),
        ("4", "", 0),
        ("", r"\boxed{}", 0),
        ("4", "No answer.", 0),
        ("-2, 3", r"\boxed{-2}", 0),
        ("-2, 3", r"\boxed{-2, 4}", 0),
        ("-2, 3", r"\boxed{3, -2}", 1),
        ("1,000", r"\boxed{1000}", 1),
        ("(1,2,3)", r"\boxed{(3,2,1)}", 0),
        (r"\{1,2,3\}", r"\boxed{\{3,2,1\}}", 1),
        (r"x \in (0,1)", r"\boxed{(0,1)}", 1),
        # C039/81481: a wrapped tuple reference previously rejected a correct answer.
        (r"\((4,2,5)\)", r"\boxed{(4,2,5)}", 1),
        # C040/2600: old comparison kept only the first root, accepting omissions.
        (r"\(-\pi,-\frac{\pi}{6},0,\frac{\pi}{6},\pi\)", r"\boxed{-\pi,0,\pi}", 0),
        (
            r"\(-\pi,-\frac{\pi}{6},0,\frac{\pi}{6},\pi\)",
            r"\boxed{-\pi}, \boxed{-\frac{\pi}{6}}, \boxed{0}, \boxed{\frac{\pi}{6}}, \boxed{\pi}",
            1,
        ),
    ],
)
def test_math_verifier(label, prediction, expected):
    result = ground_truth_utils.MathVerifier()([], prediction, label)
    assert result.score == expected
    assert result.diagnostics == {"symbolic_timeouts": 0}


@pytest.mark.parametrize("label", ["4", "$4$", "$$4$$", r"\(4\)", r"\[4\]"])
def test_reference_math_delimiters(label):
    assert ground_truth_utils.MathVerifier()([], r"\boxed{4}", label).score == 1


@pytest.mark.parametrize(
    "label,prediction,expected",
    [
        ("1,000, 2,000", r"\boxed{1000, 2000}", 1),
        ("1,000, 2,000", r"\boxed{0, 1, 2}", 0),
        ("f(1)=1", r"\boxed{f(1)=0}", 0),
    ],
)
@pytest.mark.xfail(strict=True, reason="Known Math-Verify 0.9.0 limitations; review on dependency upgrades")
def test_upstream_limitations(label, prediction, expected):
    assert ground_truth_utils.MathVerifier()([], prediction, label).score == expected


def test_parse_timeout_is_counted_and_resets():
    verifier = ground_truth_utils.MathVerifier()
    with mock.patch.object(math_verify_utils.math_verify, "parse", side_effect=errors.TimeoutException):
        result = verifier([], r"\boxed{4}", "4")
    assert result.score == 0
    assert result.diagnostics == {"symbolic_timeouts": 1}
    assert verifier([], r"\boxed{4}", "4").diagnostics == {"symbolic_timeouts": 0}


def test_each_timed_out_candidate_pair_is_counted():
    with mock.patch.object(math_verify_utils.math_verify, "verify", side_effect=errors.TimeoutException):
        result = ground_truth_utils.MathVerifier()([], r"\boxed{4}", "4")
    assert result.score == 0
    # Each side has a symbolic and a literal candidate.
    assert result.diagnostics == {"symbolic_timeouts": 4}


def test_real_comparison_alarm_is_recorded(monkeypatch, caplog):
    monkeypatch.setattr(math_verify_utils, "_TIMEOUT_SECONDS", 1)
    with mock.patch.object(grader, "sympy_expr_eq", side_effect=lambda *a: time.sleep(5)):
        result = ground_truth_utils.MathVerifier()([], r"\boxed{4}", "4")
    # Math-Verify's own literal candidate still agrees after the symbolic timeout.
    assert result.score == 1
    assert result.diagnostics == {"symbolic_timeouts": 1}
    assert "Math-Verify parsing or comparison exceeded 1 seconds" in caplog.text


def test_real_parse_alarm_is_recorded(monkeypatch):
    monkeypatch.setattr(math_verify_utils, "_TIMEOUT_SECONDS", 1)
    with mock.patch.object(parser, "extract_target_from_pred", side_effect=lambda *a, **kw: time.sleep(5)):
        result = ground_truth_utils.MathVerifier()([], r"\boxed{4}", "4")
    assert result.score == 0
    assert result.diagnostics == {"symbolic_timeouts": 1}


def test_async_math_uses_process_isolation_and_reaps_workers():
    async def run():
        verifier = ground_truth_utils.MathVerifier()
        return await asyncio.gather(
            verifier.async_call([], r"\boxed{-2}", "-2, 3"), verifier.async_call([], r"\boxed{3, -2}", "-2, 3")
        )

    children_before = set(rewards._CHILDREN)
    results = asyncio.run(run())
    assert [result.score for result in results] == [0, 1]
    assert all(isinstance(result, ground_truth_utils.VerificationResult) for result in results)
    assert all(result.diagnostics["status"] == "ok" for result in results)
    assert children_before == rewards._CHILDREN
