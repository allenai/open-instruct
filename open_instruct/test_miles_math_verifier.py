"""General math grading regressions, including compact saved-rollout examples."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest import mock

import pytest
from math_verify import errors, grader, parser
from scripts.miles import prepare_eopd_math_prompts, prepare_qwen35_math_prompts

from open_instruct import ground_truth_utils
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.datasets import run_data, source_preparation
from open_instruct.miles.rewards import math_verifier, rewards


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
    result = math_verifier.MathVerifier()([], prediction, label)
    assert result.score == expected
    assert result.diagnostics == {"symbolic_timeouts": 0}


@pytest.mark.parametrize("label", ["4", "$4$", "$$4$$", r"\(4\)", r"\[4\]"])
def test_reference_math_delimiters(label):
    assert math_verifier.MathVerifier()([], r"\boxed{4}", label).score == 1


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
    assert math_verifier.MathVerifier()([], prediction, label).score == expected


def test_parse_timeout_is_counted_and_resets():
    verifier = math_verifier.MathVerifier()
    with mock.patch.object(math_verifier.math_verify, "parse", side_effect=errors.TimeoutException):
        result = verifier([], r"\boxed{4}", "4")
    assert result.score == 0
    assert result.diagnostics == {"symbolic_timeouts": 1}
    assert verifier([], r"\boxed{4}", "4").diagnostics == {"symbolic_timeouts": 0}


def test_each_timed_out_candidate_pair_is_counted():
    with mock.patch.object(math_verifier.math_verify, "verify", side_effect=errors.TimeoutException):
        result = math_verifier.MathVerifier()([], r"\boxed{4}", "4")
    assert result.score == 0
    # Each side has a symbolic and a literal candidate.
    assert result.diagnostics == {"symbolic_timeouts": 4}


def test_real_comparison_alarm_is_recorded(monkeypatch, caplog):
    monkeypatch.setattr(math_verifier, "_TIMEOUT_SECONDS", 1)
    with mock.patch.object(grader, "sympy_expr_eq", side_effect=lambda *a: time.sleep(5)):
        result = math_verifier.MathVerifier()([], r"\boxed{4}", "4")
    # Math-Verify's own literal candidate still agrees after the symbolic timeout.
    assert result.score == 1
    assert result.diagnostics == {"symbolic_timeouts": 1}
    assert "Math-Verify parsing or comparison exceeded 1 seconds" in caplog.text


def test_real_parse_alarm_is_recorded(monkeypatch):
    monkeypatch.setattr(math_verifier, "_TIMEOUT_SECONDS", 1)
    with mock.patch.object(parser, "extract_target_from_pred", side_effect=lambda *a, **kw: time.sleep(5)):
        result = math_verifier.MathVerifier()([], r"\boxed{4}", "4")
    assert result.score == 0
    assert result.diagnostics == {"symbolic_timeouts": 1}


def test_async_math_uses_process_isolation_and_reaps_workers():
    async def run():
        verifier = math_verifier.MathVerifier()
        return await asyncio.gather(
            verifier.async_call([], r"\boxed{-2}", "-2, 3"), verifier.async_call([], r"\boxed{3, -2}", "-2, 3")
        )

    children_before = set(rewards._CHILDREN)
    results = asyncio.run(run())
    assert [result.score for result in results] == [0, 1]
    assert all(isinstance(result, ground_truth_utils.VerificationResult) for result in results)
    assert all(result.diagnostics["status"] == "ok" for result in results)
    assert children_before == rewards._CHILDREN


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param("open_instruct.ground_truth_utils.MathVerifier", id="existing-prepared-registry"),
        pytest.param(run_data.FACTORIES["math"], id="builtin-math-preparation"),
        pytest.param(source_preparation.FACTORIES["math"], id="offline-source-preparation"),
        pytest.param(prepare_eopd_math_prompts.VERIFIERS["math"]["factory"], id="eopd-preparation"),
        pytest.param(prepare_qwen35_math_prompts.VERIFIERS["math"]["factory"], id="qwen35-preparation"),
    ],
)
def test_miles_math_routes_use_math_verify_through_worker(tmp_path, factory):
    spec = {"factory": factory, "config": {}}
    registry = tmp_path / "verifiers.json"
    registry.write_text(json.dumps({"math": spec}))
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=str(registry)))
    samples = [
        SimpleNamespace(
            tokens=[],
            response_length=0,
            response=response,
            prompt="Find both roots",
            metadata={"verifiers": [{"name": "math", "target": "-2, 3"}]},
        )
        for response in (r"\boxed{-2}", r"\boxed{3, -2}")
    ]
    assert asyncio.run(rewards.registered_reward(args, samples)) == [0, 1]
    assert json.loads(registry.read_text()) == {"math": spec}
    # Offline/direct construction and child worker construction share the mapping.
    assert isinstance(rewards._resolve_verifier(spec), math_verifier.MathVerifier)
    assert spec["factory"] == factory


def test_legacy_math_and_explicit_strict_contract_are_unchanged():
    legacy = ground_truth_utils.MathVerifier()
    assert legacy([], r"\boxed{-2}", "-2, 3").score == 1
    assert legacy([], r"\boxed{3, -2}", "-2, 3").score == 0
    # Legacy build_all_verifiers enumerates direct subclasses: importing the MILES
    # implementation must not add a second verifier that overwrites its "math" key.
    assert math_verifier.MathVerifier not in ground_truth_utils.VerifierFunction.__subclasses__()
    assert isinstance(
        rewards._resolve_verifier({"factory": "open_instruct.ground_truth_utils.StrictMathVerifier"}),
        ground_truth_utils.StrictMathVerifier,
    )
