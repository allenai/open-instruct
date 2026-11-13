"""Shared response selection, explicit exceptions and legacy-mode compatibility."""

import asyncio
import copy
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.test import replay_grading_selection

from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import general_judge, grading, rewards, task_verifiers


def item(response, *, target=None, name="test", prompt="question<think>", status="completed"):
    return SimpleNamespace(
        response=response,
        tokens=[101, 102, 103, 104],
        response_length=3,
        prompt=prompt,
        status=status,
        metadata={"verifiers": [{"name": name, "target": target}]},
    )


def score(monkeypatch, sample, verifier, *, final=True, policy=None, reasoning="required", zero_truncated=True):
    registry = rewards._VerifierRegistry()
    registry["test"] = verifier
    if policy is not None:
        registry.grading_policies["test"] = policy
    monkeypatch.setattr(rewards, "_registry", lambda path: registry)
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: False)
    config = CoreConfig(
        reward_config="unused",
        reward_final_answer_only=final,
        reward_reasoning_mode=reasoning,
        reward_zero_truncated=zero_truncated,
    )
    return asyncio.run(rewards.registered_reward(SimpleNamespace(olmo_core=config), sample))


@pytest.mark.parametrize(
    "raw,mode,thinking,response,gate,count",
    [
        ("<think>work</think>answer", "required", "work", "answer", None, 1),
        ("work</think>answer<|endoftext|>", "required", "work", "answer", None, 1),
        ("work", "required", "work", None, "missing_final_answer", 0),
        ("work</think> \n<|im_end|>", "required", "work", " \n", "missing_final_answer", 1),
        ("a</think>b</think>c", "required", "a</think>b", "c", None, 2),
        ("work</think><think>more", "required", "work", "<think>more", "unfinished_reasoning", 1),
        ("answer<|endoftext|>", "none", None, "answer", None, 0),
        ("", "none", None, "", "missing_final_answer", 0),
        ("work</think>answer", "none", None, None, "unexpected_reasoning", 1),
        ("<think>work", "none", None, None, "unexpected_reasoning", 0),
    ],
)
def test_response_parts(raw, mode, thinking, response, gate, count):
    parts = grading.split_response(raw, reasoning_mode=mode)
    assert (parts.thinking, parts.response, parts.gate, parts.closing_markers) == (thinking, response, gate, count)
    assert parts.raw == raw
    if response is not None:
        assert parts.full[parts.response_start :] == response
    assert "raw" not in parts.summary() and "full" not in parts.summary()


@pytest.mark.parametrize(
    "response",
    [
        "\n\n42\n\t",
        "\n<answer>42</answer>\nExplanation outside the wrapper.\n",
        '  {"answer": 42}  \n',
        "\n```python\nprint(42)\n```\n",
    ],
)
@pytest.mark.parametrize("final", [False, True])
def test_flag_selects_full_or_after_thinking_without_answer_cleanup(monkeypatch, response, final):
    seen = []

    async def verify(tokens, prediction, label, **kwargs):
        seen.append(prediction)
        return SimpleNamespace(score=1.0, cost=0.0)

    full = "<think>scratch answer</think>" + response
    sample = item(full)
    assert score(monkeypatch, sample, SimpleNamespace(async_call=verify), final=final) == 1
    assert seen == [response if final else full]
    assert sample.response == full


def test_prompt_state_and_nonthinking_contract_are_explicit():
    parts = grading.split_response("work", prompt="question<think> \n")
    assert parts.prompt_opens_reasoning and parts.gate == "missing_final_answer"
    assert grading.split_response("answer", reasoning_mode="none", prompt="question<think>").gate == (
        "unexpected_reasoning"
    )
    # Message-list prompts cannot certify a prefilled opener; required mode still
    # requires closure, rather than inferring non-thinking from absent tags.
    assert grading.split_response("work", prompt=[{"role": "user", "content": "question"}]).gate
    with pytest.raises(ValueError, match="reasoning_mode"):
        grading.split_response("answer", reasoning_mode="guess")


def test_prompt_prefilled_closed_thinking_is_already_final(monkeypatch):
    sample = item("<answer>42</answer>", target="42", prompt="question<think>\n</think>\n\n")
    assert score(monkeypatch, sample, task_verifiers.MultiplicationVerifier()) == 1
    assert sample.metadata["grading"]["prompt_closes_reasoning"]
    assert sample.metadata["grading"]["thinking_characters"] == 0
    assert grading.split_response("<think>more", prompt=sample.prompt).gate == "unfinished_reasoning"


@pytest.mark.parametrize(
    "value",
    [
        {"scope": "typo"},
        {"require_response": 0},
        {"scope": "response", "require_response": False},
        {"require_response": False},
        {"unknown": True},
        "full",
    ],
)
def test_bad_registry_policies_fail_at_load(tmp_path, value):
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "test": {
                    "factory": "open_instruct.miles.rewards.task_verifiers.MultiplicationVerifier",
                    "grading": value,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="grading"):
        rewards._registry(str(path))


@pytest.mark.parametrize("scope", grading.SCOPES)
@pytest.mark.parametrize("final", [False, True])
def test_declared_view_and_original_tokens(monkeypatch, scope, final):
    seen = []

    async def verify(tokens, prediction, label, **kwargs):
        seen.append((tokens, prediction))
        return SimpleNamespace(score=1.0, cost=0.0)

    sample = item("work</think>answer<|im_end|>")
    original = copy.deepcopy(sample)
    assert (
        score(
            monkeypatch,
            sample,
            SimpleNamespace(async_call=verify),
            final=final,
            policy=grading.GradingPolicy(scope, objective="answer" if scope == "response" else "process"),
        )
        == 1
    )
    expected = {"response": "answer", "thinking": "work", "full": "work</think>answer"}
    assert seen == [
        ([102, 103, 104] if not final or scope == "full" else [], expected[scope] if final else expected["full"])
    ]
    assert sample.response == original.response and sample.tokens == original.tokens
    if final:
        assert sample.metadata["grading"]["components"]["test"] == {
            "scope": scope,
            "objective": "answer" if scope == "response" else "process",
            "require_response": True,
            "gate": None,
        }
    else:
        assert "grading" not in sample.metadata


def test_parts_protocol_delivers_both_pieces_once(monkeypatch):
    seen = []

    async def verify_parts(tokens, parts, label, *, scope, query=None, rollout_state=None):
        seen.append((tokens, parts.thinking, parts.response, scope, parts.raw))
        return SimpleNamespace(score=float(label == parts.thinking), cost=0.0)

    sample = item("work</think>answer<|im_end|>", target="work")
    verifier = SimpleNamespace(async_call_parts=verify_parts, reads_prompt_reasoning=True)
    assert score(monkeypatch, sample, verifier, policy=grading.GradingPolicy("thinking", objective="process")) == 1
    assert seen == [([], "work", "answer", "thinking", sample.response)]


def test_nonthinking_response_can_receive_reward(monkeypatch):
    verifier = task_verifiers.MultiplicationVerifier()
    sample = item("<answer>42</answer>", target="42", prompt="question")
    assert score(monkeypatch, sample, verifier, reasoning="none") == 1
    assert score(monkeypatch, sample, verifier, reasoning="required") == 0


def test_nonthinking_mode_rejects_incompatible_objectives(monkeypatch):
    sample = item("answer", prompt="question")
    with pytest.raises(ValueError, match="requires reward_reasoning_mode=required"):
        score(
            monkeypatch,
            sample,
            task_verifiers.MultiplicationVerifier(),
            reasoning="none",
            policy=grading.GradingPolicy("thinking", objective="process"),
        )
    with pytest.raises(ValueError, match="requires reward_reasoning_mode=required"):
        score(monkeypatch, sample, SimpleNamespace(grading_requires_reasoning=True), reasoning="none")


@pytest.mark.parametrize("scope", ["thinking", "full"])
def test_missing_response_exception_is_separate_from_scope(monkeypatch, scope):
    seen = []

    async def verify(tokens, prediction, label, **kwargs):
        seen.append(prediction)
        return SimpleNamespace(score=0.75, cost=0.0)

    sample = item("unfinished work")
    verifier = SimpleNamespace(async_call=verify, score_floor=lambda label: -3.0)
    assert (
        score(
            monkeypatch,
            sample,
            verifier,
            policy=grading.GradingPolicy(scope, objective="answer" if scope == "response" else "process"),
        )
        == -3
    )
    assert seen == []
    assert score(monkeypatch, sample, verifier, policy=grading.GradingPolicy(scope, False, "process")) == 0.75
    assert seen == ["unfinished work"]
    assert "reward_gate" not in sample.metadata
    assert sample.metadata["grading"]["components"]["test"]["gate"] is None
    sample.status = "truncated"
    assert score(monkeypatch, sample, verifier, policy=grading.GradingPolicy(scope, False, "process")) == -3
    assert seen == ["unfinished work"]


def test_mixed_grader_scopes_gate_each_component(monkeypatch):
    async def should_not_run(*args, **kwargs):
        raise AssertionError("missing response must gate correctness")

    async def process(tokens, prediction, label, **kwargs):
        assert prediction == "unfinished" and tokens == [102, 103, 104]
        return SimpleNamespace(score=0.5, cost=0.0)

    registry = rewards._VerifierRegistry()
    registry.update(
        {"answer": SimpleNamespace(async_call=should_not_run), "process": SimpleNamespace(async_call=process)}
    )
    registry.grading_policies.update(
        {"answer": grading.GradingPolicy(), "process": grading.GradingPolicy("full", False, "process")}
    )
    monkeypatch.setattr(rewards, "_registry", lambda path: registry)
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: False)
    sample = item("unfinished")
    sample.metadata["verifiers"] = [{"name": "answer", "target": "x"}, {"name": "process", "target": "y", "weight": 2}]
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config="unused", reward_final_answer_only=True))
    assert asyncio.run(rewards.registered_reward(args, sample)) == 1
    assert [c["score"] for c in sample.metadata["reward_components"]] == [0, 0.5]
    assert sample.metadata["grading"]["components"]["answer"]["gate"] == "missing_final_answer"
    assert sample.metadata["grading"]["components"]["process"]["gate"] is None


def test_new_reasoning_mode_is_validated():
    assert dataclasses.asdict(CoreConfig(reward_reasoning_mode="none"))["reward_reasoning_mode"] == "none"
    with pytest.raises(ValueError, match="reward_reasoning_mode"):
        CoreConfig(reward_reasoning_mode="auto")


def test_dataset_cannot_change_trusted_scope(monkeypatch):
    async def verify(tokens, prediction, label, **kwargs):
        assert prediction == "wrong"
        return SimpleNamespace(score=0.0, cost=0.0)

    sample = item("correct</think>wrong")
    sample.metadata["verifiers"][0]["grading"] = {"scope": "thinking", "require_response": False}
    assert score(monkeypatch, sample, SimpleNamespace(async_call=verify)) == 0


def test_external_verifier_requires_a_declared_contract_before_gating(tmp_path, monkeypatch):
    async def verify(tokens, prediction, label, **kwargs):
        assert prediction == "work</think>answer"
        return SimpleNamespace(score=1.0, cost=0.0)

    monkeypatch.setattr(rewards, "_resolve_verifier", lambda *args, **kwargs: SimpleNamespace(async_call=verify))
    path = tmp_path / "external.json"
    spec = {"factory": "external_qualified_module.RawVerifier"}
    path.write_text(json.dumps({"test": spec}))
    sample = item("work</think>answer")
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=str(path)))
    assert asyncio.run(rewards.registered_reward(args, sample)) == 1
    args.olmo_core = CoreConfig(reward_config=str(path), reward_final_answer_only=True)
    sample.response = "unfinished"
    with pytest.raises(ValueError, match="must declare registry grading policy"):
        asyncio.run(rewards.registered_reward(args, sample))
    explicit = tmp_path / "explicit.json"
    spec["grading"] = {"objective": "process", "scope": "full"}
    explicit.write_text(json.dumps({"test": spec}))
    args.olmo_core = CoreConfig(reward_config=str(explicit), reward_final_answer_only=True)
    sample.response = "work</think>answer"
    assert asyncio.run(rewards.registered_reward(args, sample)) == 1


def test_reopened_thinking_is_gated_and_legacy_behavior_is_preserved(monkeypatch):
    verifier = task_verifiers.MultiplicationVerifier()
    sample = item("work</think><think><answer>42</answer>", target="42")
    assert score(monkeypatch, sample, verifier, final=False) == 1
    assert score(monkeypatch, sample, verifier, final=True) == 0
    assert sample.metadata["reward_gate"] == "unfinished_reasoning"


def test_literal_opener_in_delivered_prose_is_not_a_reopened_section():
    raw = "work</think>The thought process uses the <think> tag. The answer is 42."
    parts = grading.split_response(raw)
    assert parts.gate is None
    assert parts.response == raw.rpartition("</think>")[2]


@pytest.mark.parametrize("scope", grading.SCOPES)
@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("name", ["general-quality", "general-quality_ref", "general-web_instruct_general_verifier"])
def test_judge_payload_respects_declared_view(monkeypatch, scope, final, name):
    registry = rewards._VerifierRegistry()
    registry[name] = SimpleNamespace()
    registry.grading_policies[name] = grading.GradingPolicy(
        scope, objective="answer" if scope == "response" else "process"
    )
    monkeypatch.setattr(rewards, "_registry", lambda path: registry)
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: True)
    monkeypatch.setattr(rewards.judge_registry, "resolve_request", lambda *args, **kwargs: None)
    monkeypatch.setenv("OI_MILES_JUDGE_API_BASE", "http://unused.invalid")
    prompts = []

    def request(config, prompt):
        prompts.append(prompt)
        return "Final Decision: Yes" if name == "general-web_instruct_general_verifier" else '{"SCORE":10}'

    monkeypatch.setattr(general_judge, "_request", request)
    delivered = "\n\n<answer>delivered answer</answer>\nExplanation outside the wrapper.\n\t"
    raw = "<think><answer>scratch answer</answer>more work</think>" + delivered
    sample = item(raw, name=name)
    sample.metadata["judge_query"] = "question"
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config="unused", reward_final_answer_only=final))
    assert asyncio.run(rewards.registered_reward(args, sample)) == 1
    parts = grading.split_response(raw)
    expected, _ = general_judge.build_judge_prompt(
        name, query="question", prediction=parts.text(scope) if final else raw, target=None, extract_answer=not final
    )
    assert prompts == [expected]
    if not final:
        # Historical generic extraction prefers the first answer wrapper, even
        # when it is inside thinking. This remains legacy behavior, not a new
        # shared input contract.
        assert "scratch answer" in prompts[0] and "delivered answer" not in prompts[0]
    elif scope == "response":
        assert delivered in prompts[0] and "scratch answer" not in prompts[0]
    else:
        assert parts.text(scope) in prompts[0]
    assert sample.response == raw


def test_saved_selection_replay_reports_gate_changes_without_score_claims(tmp_path):
    path = tmp_path / "draws.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"full": text, "id": index})
            for index, text in enumerate(
                [
                    "work</think>answer",
                    "unfinished",
                    "work</think><think>more",
                    "work</think>literal <think> tag",
                    "a</think>b</think>c",
                ]
            )
        )
        + "\n"
    )
    report = replay_grading_selection.replay(path)
    assert report["input"] == "draws.jsonl"
    assert str(tmp_path) not in json.dumps(report)
    assert report["comparison_revision"] == "46d2dbbb5bc13baa7c821afb82cd16a95be52e41"
    assert report["input_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["counts"] == {
        "rows": 5,
        "legacy_input_changes": 0,
        "eligible_input_changes": 0,
        "gate_changes": 1,
        "multiple_closing_markers": 1,
    }
    assert report["new_gates"] == {"eligible": 3, "missing_final_answer": 1, "unfinished_reasoning": 1}


@pytest.mark.parametrize("mutation", ["legacy_helper", "terminal_cleanup"])
def test_saved_selection_replay_detects_candidate_regressions(tmp_path, monkeypatch, mutation):
    path = tmp_path / "draws.jsonl"
    path.write_text(json.dumps({"full": "work</think> answer <|im_end|>"}) + "\n")
    assert replay_grading_selection.replay(path)["counts"]["legacy_input_changes"] == 0
    if mutation == "legacy_helper":
        monkeypatch.setattr(replay_grading_selection.deterministic_verifiers, "answer_text", lambda text: text)
    else:
        # Changing the candidate's shared cleanup must not also change the
        # reference. Otherwise a shared regression still reports zero changes.
        monkeypatch.setattr(grading.response_text, "strip_terminal_special_tokens", lambda text, **kwargs: text)
    report = replay_grading_selection.replay(path)
    assert report["counts"]["legacy_input_changes"] == 1
    if mutation == "terminal_cleanup":
        assert report["counts"]["eligible_input_changes"] == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  plain answer  ", "plain answer"),
        ("work</think> first</think>  last  ", "last"),
        ("work</think> answer <|im_end|>  <|endoftext|>  ", "answer"),
        ("work</think> <|im_end|> ", ""),
        ("literal <|im_end|> inside", "literal <|im_end|> inside"),
    ],
)
def test_saved_selection_replay_frozen_baseline(raw, expected):
    assert replay_grading_selection._baseline_legacy_input(raw) == expected
