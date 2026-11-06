"""Known zero rewards remain valid evidence through the reward bridge and MILES."""

import asyncio
from types import SimpleNamespace

import pytest

from open_instruct.miles.rewards import rewards


def sample(name):
    return SimpleNamespace(
        response="answer",
        tokens=[1],
        response_length=1,
        status="completed",
        prompt="question",
        metadata={"verifiers": [{"name": name, "target": "target"}]},
    )


def score(monkeypatch, adapter, item):
    monkeypatch.setattr(rewards, "_registry", lambda path: {item.metadata["verifiers"][0]["name"]: adapter})
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(reward_config="unused", reward_final_answer_only=False, reward_zero_truncated=False)
    )
    return asyncio.run(rewards.registered_reward(args, item))


@pytest.mark.parametrize(
    "status,reason,value",
    [
        ("scored", None, 1.0),
        ("scored", None, 0.0),
        ("model_format_failure", "thinking_terminator_count", 0.0),
        ("empty_answer", None, 0.0),
    ],
)
def test_if_completed_outcomes_are_valid_evidence(monkeypatch, status, reason, value):
    records = pytest.importorskip("miles.utils.inference_records")
    original = {
        "kind": "experimental_ifeval_v1",
        "contract": "experimental_ifeval_inclusion_v2",
        "status": status,
        "constraints_scored": status == "scored",
    }
    if reason:
        original["reason"] = reason

    class Adapter:
        async def async_call(self, *args, **kwargs):
            return SimpleNamespace(score=value, cost=0.0, diagnostics=original)

    item = sample("ifeval")
    assert score(monkeypatch, Adapter(), item) == value
    diagnostic = item.metadata["verifier_diagnostics"]["ifeval"]
    assert diagnostic == {**original, "status": "completed", "outcome": status}
    assert original["status"] == status  # Do not mutate the pinned scorer's result.
    assert records.validity(item.metadata)["valid"] is True


@pytest.mark.parametrize(
    "kind,status", [("experimental_ifeval_v1", "service_error"), ("other", "model_format_failure")]
)
def test_unrecognized_diagnostics_are_not_blessed(monkeypatch, kind, status):
    records = pytest.importorskip("miles.utils.inference_records")

    class Adapter:
        async def async_call(self, *args, **kwargs):
            return SimpleNamespace(score=0, cost=0, diagnostics={"kind": kind, "status": status})

    item = sample("ifeval")
    assert score(monkeypatch, Adapter(), item) == 0
    assert item.metadata["verifier_diagnostics"]["ifeval"]["status"] == status
    assert records.validity(item.metadata)["valid"] is False


def test_if_service_failure_stays_unknown(monkeypatch):
    records = pytest.importorskip("miles.utils.inference_records")

    class Adapter:
        async def async_call(self, *args, **kwargs):
            raise RuntimeError("verifier unavailable")

    item = sample("ifeval")
    with pytest.raises(RuntimeError, match="verifier unavailable"):
        score(monkeypatch, Adapter(), item)
    assert records.validity(item.metadata)["valid"] is None


@pytest.mark.parametrize("prediction", ["", "```python\nprint("])
def test_code_adapter_zero_is_valid_through_actual_bridge(monkeypatch, prediction):
    verifier = pytest.importorskip("code_execution_client.verifier")
    records = pytest.importorskip("miles.utils.inference_records")
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")
    item = sample("code")
    item.response = prediction
    item.metadata["verifiers"][0]["target"] = ["assert True"]
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://unused"))
    assert score(monkeypatch, adapter, item) == 0
    assert records.validity(item.metadata)["valid"] is True
    assert item.metadata["verifier_diagnostics"]["code"]["outcome"] == "invalid_code"


def test_code_service_failure_stays_unknown_through_actual_bridge(monkeypatch):
    verifier = pytest.importorskip("code_execution_client.verifier")
    records = pytest.importorskip("miles.utils.inference_records")
    monkeypatch.setenv("CODE_EXECUTION_TOKEN", "test")

    client = pytest.importorskip("code_execution_client")

    def fail(*args, **kwargs):
        raise client.ServiceError("unavailable")

    monkeypatch.setattr(verifier.CodeExecutionClient, "execute", fail)
    item = sample("code")
    item.response = "x=1"
    item.metadata["verifiers"][0]["target"] = ["assert x==1"]
    adapter = verifier.CodeVerifier(verifier.ServiceConfig(api_url="http://unused"))
    with pytest.raises(client.ServiceError):
        score(monkeypatch, adapter, item)
    assert records.validity(item.metadata)["valid"] is None


@pytest.mark.parametrize(
    "response,expected,outcome",
    [
        ("computer", 0, "model_format_failure"),
        ("r</think>computer</think>computer", 0, "model_format_failure"),
        ("r</think>", 0, "empty_answer"),
        ("r</think> \n\t", 0, "empty_answer"),
        ("r</think>desk", 0, "scored"),
        ("r</think>computer computer", 1, "scored"),
    ],
)
def test_actual_qualified_if_contract(monkeypatch, response, expected, outcome):
    qualified = pytest.importorskip("experimental_ifeval_reward_v2")
    records = pytest.importorskip("miles.utils.inference_records")
    target = {"instruction_id": [qualified.INCLUSION], "kwargs": [{"keyword": "computer"}]}
    item = sample("ifeval")
    item.response = response
    item.metadata["verifiers"][0]["target"] = target
    native = qualified.score_response(response, target)
    assert native.score == expected and native.diagnostics["status"] == outcome
    assert score(monkeypatch, qualified.IFVerifier(), item) == native.score
    assert item.metadata["verifier_diagnostics"]["ifeval"]["outcome"] == outcome
    assert records.validity(item.metadata)["valid"] is True


def test_actual_qualified_if_bad_target_stays_unknown(monkeypatch):
    qualified = pytest.importorskip("experimental_ifeval_reward_v2")
    records = pytest.importorskip("miles.utils.inference_records")
    item = sample("ifeval")
    item.response = "malformed response"
    item.metadata["verifiers"][0]["target"] = {"instruction_id": ["unknown"], "kwargs": [{}]}
    with pytest.raises(qualified.base.TargetQualificationError):
        score(monkeypatch, qualified.IFVerifier(), item)
    assert records.validity(item.metadata)["valid"] is None
