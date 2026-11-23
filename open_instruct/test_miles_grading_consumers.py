"""Typed grading views are authoritative; consumers only parse task answers."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from open_instruct import answer_utils, ground_truth_utils
from open_instruct import test_miles_grading as shared
from open_instruct.miles.datasets import long_context as lc_targets
from open_instruct.miles.rewards import deterministic_verifiers as common
from open_instruct.miles.rewards import grading, json_schema, litmus, long_context, rewards, task_verifiers, toolrl


@pytest.mark.parametrize(
    "declaration",
    [
        {"scope": "full"},
        {"objective": "answer", "scope": "thinking"},
        {"objective": "process", "scope": "response"},
        {"objective": "composite", "scope": "full"},
        {"objective": "unknown"},
    ],
)
def test_objective_and_view_must_agree(declaration):
    with pytest.raises(ValueError, match="grading"):
        grading.GradingPolicy.from_spec({"grading": declaration})


def test_composite_cannot_hide_undeclared_or_retyped_inputs():
    with pytest.raises(ValueError, match="grading_components"):
        grading.component_policies(SimpleNamespace(), grading.GradingPolicy(objective="composite"))
    with pytest.raises(ValueError, match="relabeled"):
        grading.component_policies(toolrl.ToolRLVerifier(), grading.GradingPolicy())


def test_composite_cannot_give_answer_correctness_access_to_thinking(monkeypatch):
    verifier = toolrl.ToolRLVerifier()
    monkeypatch.setattr(verifier, "grading_components", {"correctness": {"objective": "answer", "scope": "full"}})
    with pytest.raises(ValueError, match="answer/composite"):
        grading.component_policies(verifier, grading.verifier_policy(verifier))


@pytest.mark.parametrize("objective,scope,expected", [("answer", "response", 0), ("format", "full", 1)])
def test_registry_type_selects_input_without_consumer_reselection(tmp_path, objective, scope, expected):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "test": {
                    "factory": "open_instruct.miles.rewards.deterministic_verifiers.DeterministicVerifier",
                    "grading": {"objective": objective, "scope": scope},
                }
            }
        )
    )
    target = {"kind": "format", "value": {"type": "regex", "verify_regex": ["^keep"]}}
    value = shared.item("keep</think>tail", target=target)
    args = SimpleNamespace(olmo_core=shared.CoreConfig(reward_config=str(path), reward_final_answer_only=True))
    assert asyncio.run(rewards.registered_reward(args, value)) == expected
    assert value.metadata["grading"]["components"]["test"]["objective"] == objective
    # Flag-off retains the historical consumer's suffix selection.
    args.olmo_core = shared.CoreConfig(reward_config=str(path), reward_final_answer_only=False)
    assert asyncio.run(rewards.registered_reward(args, value)) == 0


def test_task_extraction_survives_without_thinking_extraction():
    text = "prefix</think>tail"
    assert common.answer_text(text, selected_input=True) == text
    assert common.answer_text(text) == "tail"
    assert answer_utils.extract_final_answer(text, selected_input=True) == text
    assert answer_utils.extract_final_answer(text) == "tail"
    assert answer_utils.extract_final_answer("<answer>42</answer> prose", selected_input=True) == "42"
    assert ground_truth_utils.remove_thinking_section("<answer>42</answer> prose", selected_input=True) == "42 prose"


@pytest.mark.parametrize("factory", [ground_truth_utils.PuzzleMatcherVerifier, ground_truth_utils.F1Verifier])
def test_legacy_text_consumers_respect_selected_view_through_async_bridge(factory):
    verifier = factory()
    text = "keep</think>tail"
    # Native text normalization is still applied to both label and prediction.
    assert asyncio.run(verifier.async_call([], text, text, selected_input=True)).score == 1
    assert asyncio.run(verifier.async_call([], text, text)).score < 1


def test_manifest_and_native_if_keep_task_normalization_without_boundary_selection():
    text = '{"marker":"</think>"}'
    target = {"instruction_id": ["detectable_format:json_format"], "kwargs": [{}]}
    manifest = task_verifiers.ManifestIFVerifier()
    assert asyncio.run(manifest.async_call([], text, target, selected_input=True)).score == 1
    assert asyncio.run(manifest.async_call([], text, target)).score == 0
    old = ground_truth_utils.IFEvalVerifierOld()
    assert asyncio.run(old.async_call([], text, {"func_name": "validate_json_format"}, selected_input=True)).score == 1
    assert asyncio.run(old.async_call([], text, {"func_name": "validate_json_format"})).score == 0


def test_litmus_and_schema_do_not_reselect_inside_supplied_text():
    target = {"answer_type": "string", "expected_answer": "keep", "output_regex": "^(keep)"}
    verifier = litmus.LitmusVerifier()
    assert asyncio.run(verifier.async_call([], "keep</think>tail", target, selected_input=True)).score == 1
    assert asyncio.run(verifier.async_call([], "keep</think>tail", target)).score == 0
    verifier = json_schema.JSONSchemaVerifier()
    text = '{"marker":"</think>"}'
    assert asyncio.run(verifier.async_call([], text, {"type": "object"}, selected_input=True)).score == 1
    assert asyncio.run(verifier.async_call([], text, {"type": "object"})).score == 0


def test_long_context_uses_shared_eligibility_for_literal_opener(monkeypatch):
    text = "The literal <think> token."
    target = lc_targets.target("exact", [text])
    value = shared.item("work</think>" + text, target=target)
    assert shared.score(monkeypatch, value, long_context.LongContextVerifier()) == 1
    assert long_context.score(text, target) == 0  # Historical direct-call behavior.
    value.response = "work</think><think>unfinished"
    assert shared.score(monkeypatch, value, long_context.LongContextVerifier()) == 0
    assert value.metadata["reward_gate"] == "unfinished_reasoning"


def test_legacy_lm_judge_gets_selected_contract_before_building_any_request(monkeypatch):
    observed = []

    class Captured(Exception):
        pass

    def extract(text, *, selected_input=False):
        observed.append((text, selected_input))
        raise Captured

    monkeypatch.setattr(ground_truth_utils, "extract_final_answer", extract)
    verifier = object.__new__(ground_truth_utils.LMJudgeVerifier)
    with pytest.raises(Captured):
        asyncio.run(verifier.async_call([], "keep</think>tail", "target", "query", selected_input=True))
    assert observed == [("keep</think>tail", True)]


def test_toolrl_records_separate_format_and_correctness_views(monkeypatch):
    value = shared.item("work</think>delivered", target={"format": "olmo-toolrl-v1", "tools": [], "calls": []})
    verifier = toolrl.ToolRLVerifier()
    observed = []
    original = verifier._grade

    def record(prediction, label, **kwargs):
        observed.append((prediction, kwargs["action"]))
        return original(prediction, label, **kwargs)

    monkeypatch.setattr(verifier, "_grade", record)
    assert shared.score(monkeypatch, value, verifier) == 1
    assert observed == [("work</think>delivered", "delivered")]
    contract = value.metadata["grading"]["components"]["test"]
    assert contract["objective"] == "composite"
    assert contract["inputs"]["correctness"]["objective"] == "answer"
    assert contract["inputs"]["correctness"]["scope"] == "response"
    assert contract["inputs"]["format"]["objective"] == "format"
    assert contract["inputs"]["format"]["scope"] == "full"


@pytest.mark.parametrize(
    "factory,declaration", [("task_verifiers.R1FormatVerifier", {"scope": "full"}), ("toolrl.ToolRLVerifier", {})]
)
def test_registry_preflight_rejects_reasoning_objectives_before_any_sample(tmp_path, factory, declaration):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({"test": {"factory": "open_instruct.miles.rewards." + factory, "grading": declaration}})
    )
    args = SimpleNamespace(
        olmo_core=shared.CoreConfig(
            reward_config=str(path), reward_final_answer_only=True, reward_reasoning_mode="none"
        )
    )
    with pytest.raises(ValueError, match="requires reward_reasoning_mode=required"):
        rewards.validate_registry(args)
    # Historical dispatch ignores the new reasoning convention.
    args.olmo_core = shared.CoreConfig(reward_config=str(path), reward_reasoning_mode="none")
    assert "test" in rewards.validate_registry(args)


def test_registry_preflight_rejects_undeclared_external_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(rewards, "_resolve_verifier", lambda *a, **kw: SimpleNamespace())
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"test": {"factory": "external.Factory"}}))
    args = SimpleNamespace(olmo_core=shared.CoreConfig(reward_config=str(path), reward_final_answer_only=True))
    with pytest.raises(ValueError, match="must declare registry grading policy"):
        rewards.validate_registry(args)


def test_r1_response_format_remains_usable_without_thinking(monkeypatch):
    sample = shared.item("<answer>42</answer>", prompt="question")
    verifier = task_verifiers.R1FormatVerifier()
    assert shared.score(monkeypatch, sample, verifier, reasoning="none") == 1
    with pytest.raises(ValueError, match="requires reward_reasoning_mode=required"):
        shared.score(
            monkeypatch, sample, verifier, reasoning="none", policy=grading.GradingPolicy("full", objective="format")
        )
