"""Final-answer dispatch must not reapply a verifier's raw-response boundary gate."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from open_instruct import ground_truth_utils
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import rewards, task_verifiers


@pytest.fixture
def qualified_if(monkeypatch, tmp_path):
    calls = []

    def grade_answer(prediction, label):
        calls.append(("final", prediction, label))
        if label != "computer":
            raise ValueError("Invalid target")
        return SimpleNamespace(score=float(prediction.strip() == label), cost=0, diagnostics={"status": "scored"})

    class RawIFVerifier:
        @classmethod
        def get_config_class(cls):
            return task_verifiers.RewardConfig

        def __init__(self, verifier_config):
            calls.append(("initialize", verifier_config))

        async def async_call(self, tokens, prediction, label, **kwargs):
            calls.append(("raw", prediction, list(tokens)))
            if prediction.count("</think>") != 1:
                return SimpleNamespace(score=0.0, cost=0, diagnostics={"status": "model_format_failure"})
            return grade_answer(ground_truth_utils.remove_thinking_section(prediction), label)

    module = SimpleNamespace(IFVerifier=RawIFVerifier, grade_answer=grade_answer)
    monkeypatch.setitem(sys.modules, "qualified_if_test_module", module)
    registry = tmp_path / "verifiers.json"
    registry.write_text(
        json.dumps(
            {
                "ifeval": {
                    "factory": "open_instruct.miles.rewards.task_verifiers.QualifiedIFVerifier",
                    "config": {"module": "qualified_if_test_module"},
                }
            }
        )
    )
    return registry, calls


def sample(response, status="completed", label="computer", verifier="ifeval"):
    return SimpleNamespace(
        response=response,
        tokens=[1, 2, 3],
        response_length=2,
        status=status,
        prompt="Include computer.",
        metadata={"verifiers": [{"name": verifier, "target": label}]},
    )


def score(registry, item, final_only):
    args = SimpleNamespace(
        olmo_core=CoreConfig(
            reward_config=str(registry), reward_final_answer_only=final_only, reward_zero_truncated=True
        )
    )
    return asyncio.run(rewards.registered_reward(args, item))


@pytest.mark.parametrize("final_only", [False, True])
@pytest.mark.parametrize(
    "response,expected",
    [
        ("reasoning </think> computer", 1),
        ("computer </think> desk", 0),
        ("reasoning </think><answer>computer</answer>", 1),
        ("reasoning </think> \n<|assistant|><answer> computer </answer> \n", 1),
        ("computer </think><answer>desk</answer>", 0),
        ("reasoning </think><answer></answer>", 0),
    ],
)
def test_qualified_if_grades_selected_contents_and_preserves_raw(qualified_if, final_only, response, expected):
    registry, calls = qualified_if
    item = sample(response)
    assert score(registry, item, final_only) == expected
    assert item.response == response and item.tokens == [1, 2, 3]
    raw_calls = [c for c in calls if c[0] == "raw"]
    assert raw_calls == ([] if final_only else [("raw", response, [2, 3])])
    assert calls[-1] == ("final", ground_truth_utils.remove_thinking_section(response), "computer")
    assert item.metadata["verifier_diagnostics"]["ifeval"]["status"] == "scored"


@pytest.mark.parametrize("response", ["computer", "reasoning </think>", "reasoning </think> \n\t"])
def test_final_only_missing_or_empty_answer_never_reaches_if(qualified_if, response):
    registry, calls = qualified_if
    item = sample(response)
    assert score(registry, item, True) == 0
    assert item.metadata["reward_gate"] == "missing_final_answer"
    assert not [c for c in calls if c[0] in {"raw", "final"}]


@pytest.mark.parametrize("final_only,expected", [(False, 0), (True, 1)])
def test_repeated_marker_keeps_each_modes_registered_contract(qualified_if, final_only, expected):
    registry, _ = qualified_if
    assert score(registry, sample("draft </think> wrong </think> computer"), final_only) == expected


@pytest.mark.parametrize("final_only", [False, True])
def test_truncation_gate_precedes_both_if_entry_points(qualified_if, final_only):
    registry, calls = qualified_if
    item = sample("reasoning </think> computer", "truncated")
    assert score(registry, item, final_only) == 0
    assert item.metadata["reward_gate"] == "truncated"
    assert not [c for c in calls if c[0] in {"raw", "final"}]


def test_final_only_target_error_is_not_silently_zeroed(qualified_if):
    registry, _ = qualified_if
    with pytest.raises(ValueError, match="Invalid target"):
        score(registry, sample("reasoning </think> computer", label="bad target"), True)


def test_qualified_if_requires_explicit_runtime_module():
    with pytest.raises(ValueError, match="explicit module"):
        task_verifiers.QualifiedIFVerifier()


def test_qualified_if_does_not_bypass_source_pin_failure(qualified_if, monkeypatch):
    registry, _ = qualified_if

    def reject_pins(self, verifier_config):
        raise ValueError("Native source changed; requalify instructions")

    monkeypatch.setattr(sys.modules["qualified_if_test_module"].IFVerifier, "__init__", reject_pins)
    with pytest.raises(ValueError, match="Native source changed; requalify instructions"):
        score(registry, sample("reasoning </think><answer>computer</answer>"), True)


def test_legacy_text_verifiers_still_receive_only_final_text(monkeypatch):
    observed = []

    class TextVerifier:
        async def async_call(self, tokens, prediction, label, **kwargs):
            observed.append((tokens, prediction))
            return SimpleNamespace(score=1.0, cost=0)

    monkeypatch.setattr(rewards, "_registry", lambda path: {"ifeval": TextVerifier()})
    assert score("unused", sample("reasoning </think> computer"), True) == 1
    assert observed == [([], " computer")]


@pytest.mark.parametrize("final_only", [False, True])
def test_if_group_retains_reward_contrast_for_zero_variance_filter(qualified_if, final_only):
    registry, _ = qualified_if
    items = [sample("reasoning </think> " + answer) for answer in ["computer", "desk", "computer", "desk"]]
    args = SimpleNamespace(
        olmo_core=CoreConfig(
            reward_config=str(registry), reward_final_answer_only=final_only, reward_zero_truncated=True
        )
    )
    assert asyncio.run(rewards.registered_reward(args, items)) == [1.0, 0.0, 1.0, 0.0]


@pytest.fixture
def r1_format(tmp_path):
    registry = tmp_path / "r1-format.json"
    registry.write_text(
        json.dumps({"r1_format": {"factory": "open_instruct.miles.rewards.task_verifiers.R1FormatVerifier"}})
    )
    return registry


@pytest.mark.parametrize("final_only", [False, True])
@pytest.mark.parametrize(
    "response,raw_expected,final_expected",
    [
        ("reasoning </think><answer>42</answer>", 1, 1),
        ("reasoning </think> \n\t<answer>line one\nline two</answer>", 1, 1),
        ("reasoning </think><answer>42</answer> trailing text", 1, 1),
        # This is the existing format check, not an answer-content validator.
        ("reasoning </think><answer></answer>", 1, 1),
        ("reasoning </think>42</answer>", 0, 0),
        ("reasoning </think><answer>42", 0, 0),
        ("reasoning </think>prefix <answer>42</answer>", 0, 0),
        ("<answer>42</answer> </think>wrong", 0, 0),
        ("draft </think><answer>42</answer> </think>wrong", 1, 0),
        ("draft </think>wrong </think><answer>42</answer>", 1, 1),
    ],
)
def test_r1_format_checks_selected_region_and_preserves_raw(
    r1_format, final_only, response, raw_expected, final_expected
):
    item = sample(response, verifier="r1_format")
    assert score(r1_format, item, final_only) == (final_expected if final_only else raw_expected)
    assert item.response == response and item.tokens == [1, 2, 3]
    assert item.metadata["reward_components"] == [
        {"name": "r1_format", "score": final_expected if final_only else raw_expected, "weight": 1.0, "cost": 0.0}
    ]


@pytest.mark.parametrize("response", ["<answer>42</answer>", "reasoning </think>", "reasoning </think> \n\t"])
def test_r1_format_cannot_bypass_missing_or_empty_final_gate(r1_format, response):
    item = sample(response, verifier="r1_format")
    assert score(r1_format, item, True) == 0
    assert item.metadata["reward_gate"] == "missing_final_answer"


@pytest.mark.parametrize("final_only", [False, True])
def test_r1_format_cannot_bypass_truncation_gate(r1_format, final_only):
    item = sample("reasoning </think><answer>42</answer>", "truncated", verifier="r1_format")
    assert score(r1_format, item, final_only) == 0
    assert item.metadata["reward_gate"] == "truncated"


@pytest.mark.parametrize("final_only", [False, True])
def test_r1_format_retains_group_reward_contrast(r1_format, final_only):
    items = [
        sample("reasoning </think>" + answer, verifier="r1_format")
        for answer in ["<answer>42</answer>", "42", "<answer>wrong</answer>", "<answer>42"]
    ]
    assert score(r1_format, items, final_only) == [1.0, 0.0, 1.0, 0.0]
