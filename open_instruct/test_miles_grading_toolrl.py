"""Regression coverage for original context required by specialized graders."""

import pytest

from open_instruct import test_miles_grading as shared
from open_instruct.miles.rewards import toolrl


@pytest.mark.parametrize("prompt_prefilled", [False, True])
def test_toolrl_boundary_format_survives_final_selection(monkeypatch, prompt_prefilled):
    raw = "work</think>delivered" if prompt_prefilled else "<think>work</think>delivered"
    sample = shared.item(
        raw,
        target={"format": "olmo-toolrl-v1", "tools": [], "calls": []},
        prompt="question<think>" if prompt_prefilled else "question",
    )
    assert shared.score(monkeypatch, sample, toolrl.ToolRLVerifier()) == 1
    diagnostic = sample.metadata["verifier_diagnostics"]["test"]
    assert diagnostic["format_score"] == 1 and diagnostic["correctness_score"] == 0


def test_toolrl_and_other_graders_share_last_marker_action(monkeypatch):
    sample = shared.item(
        "<think>work</think>old action</think>delivered",
        target={"format": "olmo-toolrl-v1", "tools": [], "calls": []},
        prompt="question",
    )
    verifier = toolrl.ToolRLVerifier()
    # Repeated markers still fail ToolRL format; its action selection is now
    # shared instead of parsing a different (first-marker) action region.
    seen = []
    original = verifier._grade

    def record(prediction, label, **kwargs):
        seen.append(kwargs["action"])
        return original(prediction, label, **kwargs)

    monkeypatch.setattr(verifier, "_grade", record)
    assert shared.score(monkeypatch, sample, verifier) == 0
    assert seen == ["delivered"]


@pytest.mark.parametrize("unit_scale,expected,floor", [(True, 1.0, 0.0), (False, 4.0, -3.0)])
def test_toolrl_call_credit_and_gate_floor_through_dispatch(monkeypatch, unit_scale, expected, floor):
    target = {
        "format": "olmo-toolrl-v1",
        "tool_format": "olmo_pythonic_v1",
        "calls": [{"name": "lookup", "parameters": {"item": "apples"}}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {"item": {"type": "string"}}},
                },
            }
        ],
    }
    sample = shared.item('work</think><function_calls>lookup(item="apples")</function_calls>', target=target)
    verifier = toolrl.ToolRLVerifier(toolrl.ToolRLConfig(unit_scale=unit_scale))
    assert shared.score(monkeypatch, sample, verifier) == expected
    assert sample.metadata["verifier_diagnostics"]["test"]["full_success"]
    sample.response = "unfinished work"
    assert shared.score(monkeypatch, sample, verifier) == floor
    assert sample.metadata["grading"]["components"]["test"]["gate"] == "missing_final_answer"


def test_toolrl_nonthinking_mode_is_an_explicit_incompatibility(monkeypatch):
    with pytest.raises(ValueError, match="requires reward_reasoning_mode=required"):
        shared.score(monkeypatch, shared.item("answer", prompt="question"), toolrl.ToolRLVerifier(), reasoning="none")


@pytest.mark.parametrize("prompt_prefilled", [False, True])
def test_actual_prompt_overrides_both_label_and_registry_prefix_flags(monkeypatch, prompt_prefilled):
    raw = "work</think>delivered" if prompt_prefilled else "<think>work</think>delivered"
    sample = shared.item(
        raw,
        target={"format": "olmo-toolrl-v1", "tools": [], "calls": [], "think_prefix_in_prompt": not prompt_prefilled},
        prompt="question<think>" if prompt_prefilled else "question",
    )
    verifier = toolrl.ToolRLVerifier(toolrl.ToolRLConfig(think_prefix_in_prompt=not prompt_prefilled))
    assert shared.score(monkeypatch, sample, verifier) == 1
    assert sample.metadata["verifier_diagnostics"]["test"]["format_score"] == 1
