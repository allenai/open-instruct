"""Internal symbolic timeouts remain observable without changing math rewards."""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import sympy
from math_verify import errors, grader

from open_instruct import ground_truth_utils, math_utils
from open_instruct.miles.configuration import config
from open_instruct.miles.rewards import math_verifier, rewards


@pytest.mark.parametrize("factory", [ground_truth_utils.MathVerifier, ground_truth_utils.StrictMathVerifier])
@pytest.mark.parametrize("answer, score", [("4", 1.0), ("2+2", 0.0), ("5", 0.0)])
def test_timeout_preserves_fallback_and_resets_per_call(factory, answer, score):
    verifier = factory()
    with mock.patch.object(math_utils, "parse_latex", side_effect=TimeoutError):
        result = verifier([], rf"\boxed{{{answer}}}", "4")
    assert result.score == score
    assert result.diagnostics == {"symbolic_timeouts": 1}
    # Reused worker instances must not accumulate counts across requests.
    healthy = verifier([], r"\boxed{4}", "4")
    assert healthy.score == 1.0
    assert healthy.diagnostics == {"symbolic_timeouts": 0}


@pytest.mark.parametrize("answer, score", [("4", 1.0), ("2+2", 0.0), ("5", 0.0)])
def test_math_verify_timeout_preserves_library_literal_fallback(answer, score):
    verifier = math_verifier.MathVerifier()
    with mock.patch.object(grader, "sympy_expr_eq", side_effect=errors.TimeoutException):
        result = verifier([], rf"\boxed{{{answer}}}", "4")
    assert result.score == score
    assert result.diagnostics == {"symbolic_timeouts": 1}
    assert verifier([], r"\boxed{4}", "4").diagnostics == {"symbolic_timeouts": 0}


def test_real_five_second_alarm_is_recorded(caplog):
    diagnostics = {}
    with mock.patch.object(math_utils, "parse_latex", side_effect=lambda _: time.sleep(10)):
        assert math_utils.is_equiv("2+2", "4", diagnostics=diagnostics) is False
    assert diagnostics == {"symbolic_timeouts": 1}
    assert "Symbolic math comparison exceeded 5 seconds" in caplog.text


def test_simplification_timeout_is_recorded():
    diagnostics = {}
    with (
        mock.patch.object(math_utils, "parse_latex", return_value=sympy.Integer(4)),
        mock.patch.object(math_utils.sympy, "simplify", side_effect=TimeoutError),
    ):
        assert math_utils.is_equiv("4", "4", diagnostics=diagnostics) is False
    assert diagnostics == {"symbolic_timeouts": 1}


def test_mismatch_and_parse_failure_are_not_timeouts():
    diagnostics = {"symbolic_timeouts": 0}
    assert math_utils.is_equiv("4", "5", diagnostics=diagnostics) is False
    with mock.patch.object(math_utils, "parse_latex", side_effect=TypeError):
        assert math_utils.is_equiv("bad", "4", diagnostics=diagnostics) is False
    assert diagnostics == {"symbolic_timeouts": 0}
    # Existing callers need not supply diagnostics.
    assert math_utils.is_equiv("4", "4") is True


class TimeoutFixtureVerifier(math_verifier.MathVerifier):
    """Inject a comparison timeout inside the real worker, not in its IPC result."""

    def __call__(self, tokenized_prediction, prediction, label, query=None, rollout_state=None):
        if query == "force-timeout":
            with mock.patch.object(grader, "sympy_expr_eq", side_effect=errors.TimeoutException):
                return super().__call__(tokenized_prediction, prediction, label, query, rollout_state)
        return super().__call__(tokenized_prediction, prediction, label, query, rollout_state)


def test_worker_timeout_diagnostics_reach_sample_and_tracking(tmp_path, monkeypatch, caplog):
    metrics = pytest.importorskip("open_instruct.miles.rollout.rollout_metrics")
    registry = tmp_path / "verifiers.json"
    registry.write_text(
        json.dumps({"math": {"factory": "open_instruct.test_miles_symbolic_math_timeouts.TimeoutFixtureVerifier"}})
    )
    args = SimpleNamespace(olmo_core=config.CoreConfig(reward_config=str(registry)), save=None)
    samples = [
        SimpleNamespace(
            tokens=[],
            response_length=0,
            response=answer,
            prompt=query,
            metadata={"verifiers": [{"name": "math", "target": "4"}]},
        )
        for answer, query in [
            (r"\boxed{4}", "force-timeout"),
            (r"\boxed{5} Final Answer: The final answer is 6. I hope it is correct.", "force-timeout"),
            (r"\boxed{4}", "normal"),
        ]
    ]
    children_before = set(rewards._CHILDREN)
    assert asyncio.run(rewards.registered_reward(args, samples)) == [1.0, 0.0, 1.0]
    assert children_before == rewards._CHILDREN
    records = [sample.metadata["verifier_diagnostics"]["math"] for sample in samples]
    assert [record["symbolic_timeouts"] for record in records] == [1, 1, 0]
    assert all(record["status"] == "ok" for record in records)
    assert records[0]["symbolic_timeout_rewarded"] is True
    assert records[1]["symbolic_timeout_rewarded"] is False
    assert "Math verifier completed with 1 symbolic timeout(s); final score=0.0" in caplog.text

    logged = []
    monkeypatch.setattr(metrics, "compute_rollout_step", lambda args, rollout_id: rollout_id)
    monkeypatch.setattr(metrics.tracking, "log", lambda args, values, **kwargs: logged.append(values))
    assert metrics.log_rollout_data(3, args, samples, None, 1.0) is False
    expected = {
        "rollout/math_verifier/samples": 3,
        "rollout/math_verifier/timeouts": 0,
        "rollout/math_verifier/timeout_fraction": 0.0,
        "rollout/math_verifier/symbolic_timeouts": 2,
        "rollout/math_verifier/symbolic_timeout_samples": 2,
        "rollout/math_verifier/symbolic_timeout_fraction": 2 / 3,
        "rollout/math_verifier/symbolic_timeout_rewarded_samples": 1,
    }
    assert {key: logged[0][key] for key in expected} == expected
    assert logged[0]["rollout/step"] == 3
    extra = {}
    metrics.log_rollout_data(3, args, samples, extra, 1.0)
    assert {key: extra[key] for key in expected} == expected
    assert len(logged) == 1  # Native MILES logs the extra-metrics path itself.


def test_outer_timeout_and_old_records_do_not_count_as_symbolic_timeouts():
    metrics = pytest.importorskip("open_instruct.miles.rollout.rollout_metrics")
    samples = [
        SimpleNamespace(metadata={"verifier_diagnostics": {"math": {"kind": "math", "status": status}}})
        for status in ("timeout", "ok")
    ]
    values = metrics.math_verifier_metrics(samples)
    assert values["rollout/math_verifier/timeouts"] == 1
    assert values["rollout/math_verifier/timeout_fraction"] == 0.5
    assert all(value == 0 for key, value in values.items() if "symbolic_" in key)
    assert metrics.math_verifier_metrics([]) == {}


def test_multiple_candidates_count_each_timeout_and_still_allow_later_match():
    verifier = ground_truth_utils.MathVerifier()
    with mock.patch.object(math_utils, "parse_latex", side_effect=TimeoutError):
        failed = verifier([], r"\boxed{5} Final Answer: The final answer is 6. I hope it is correct.", "4")
        recovered = verifier([], r"\boxed{5} Final Answer: The final answer is 4. I hope it is correct.", "4")
    assert failed.score == 0.0
    assert recovered.score == 1.0
    assert failed.diagnostics == recovered.diagnostics == {"symbolic_timeouts": 2}
