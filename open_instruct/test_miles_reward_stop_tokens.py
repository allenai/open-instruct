"""Generation terminators must not become answer content in any reward route."""

import asyncio
from types import SimpleNamespace

import pytest

from open_instruct import ground_truth_utils
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.rewards import (
    deterministic_verifiers,
    nemo_tool_calls,
    response_text,
    rewards,
    task_verifiers,
)

STOP_SEQUENCES = [
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_end|><|endoftext|>",
    "<|endoftext|><|im_end|>",
    "<|endoftext|><|endoftext|>",
    "<|im_end|> \n<|endoftext|>  \n",
]


@pytest.mark.parametrize("stop_token", STOP_SEQUENCES)
@pytest.mark.parametrize(
    "instruction,kwargs,answer",
    [
        ("startend:end_checker", {"end_phrase": "Any other questions?"}, "Hello. Any other questions?"),
        ("last_word:last_word_answer", {"last_word": "spring"}, "The season is spring"),
        ("startend:quotation", {}, '"Hello world"'),
        ("detectable_format:json_format", {}, '{"a": 1}'),
    ],
)
def test_registered_if_reward_ignores_terminal_stop_token(monkeypatch, stop_token, instruction, kwargs, answer):
    target = {"instruction_id": [instruction], "kwargs": [kwargs]}
    verifier = task_verifiers.ManifestIFVerifier()
    monkeypatch.setattr(rewards, "_registry", lambda path: {"ifeval": verifier})
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: False)
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config="unused"))
    original = answer + stop_token
    tokens = [101, 102]
    sample = SimpleNamespace(
        response=original,
        tokens=tokens,
        response_length=len(tokens),
        prompt="Follow the constraint.",
        metadata={"verifiers": [{"name": "ifeval", "target": target}]},
    )
    # Confirm the regression is sensitive to the literal suffix before checking
    # the complete registry adapter path.
    assert ground_truth_utils.IFEvalVerifier()([], original, repr([target])).score == 0.0
    assert asyncio.run(rewards.registered_reward(args, sample)) == 1.0
    assert sample.response == original
    assert sample.tokens is tokens and sample.tokens == [101, 102]


@pytest.mark.parametrize("managed_judge", [False, True])
@pytest.mark.parametrize("stop_token", STOP_SEQUENCES)
def test_all_reward_routes_receive_cleaned_text_and_original_tokens(monkeypatch, managed_judge, stop_token):
    received = []

    async def verify(tokens, prediction, target, **kwargs):
        received.append((tokens, prediction))
        return SimpleNamespace(score=1.0, cost=0.0)

    async def judge(args, sample, *, prediction, **kwargs):
        received.append((sample.tokens, prediction))
        assert sample.response.endswith(stop_token)
        return 1.0

    monkeypatch.setattr(rewards, "_registry", lambda path: {"test": SimpleNamespace(async_call=verify)})
    monkeypatch.setattr(rewards.judge_registry, "bound", lambda name: managed_judge)
    monkeypatch.setattr(rewards.general_judge, "general_judge_score", judge)
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config="unused"))
    answer = '<think>Plan</think><tool_call>{"name": "test"}</tool_call>Answer'
    sample = SimpleNamespace(
        response=answer + stop_token,
        tokens=[101, 102],
        response_length=2,
        prompt="Prompt",
        metadata={"verifiers": [{"name": "test", "target": "Target"}]},
    )
    assert asyncio.run(rewards.registered_reward(args, sample)) == 1.0
    assert received == [([101, 102], answer)]
    assert sample.response == answer + stop_token


@pytest.mark.parametrize("stop_token", STOP_SEQUENCES)
def test_empty_final_answer_with_only_stop_token_stays_gated(monkeypatch, stop_token):
    monkeypatch.setattr(rewards, "_registry", lambda path: {"test": SimpleNamespace()})
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config="unused", reward_final_answer_only=True))
    sample = SimpleNamespace(
        response="Reasoning</think>" + stop_token, metadata={"verifiers": [{"name": "test", "target": "Target"}]}
    )
    assert asyncio.run(rewards.registered_reward(args, sample)) == 0.0
    assert sample.metadata["reward_gate"] == "missing_final_answer"


@pytest.mark.parametrize(
    "text",
    [
        'Mention "<|endoftext|>" literally.',
        "<|im_start|>assistant\nAnswer",
        "<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>middle",
        "Answer<|endofprompt|>",
        "Answer<|pad|>",
        "<think>Reasoning</think><tool_response>Result</tool_response>",
        "  Answer\n\n",
    ],
)
def test_other_content_is_preserved(text):
    assert response_text.strip_terminal_special_tokens(text) == text


def test_selected_special_tokens_are_stripped_by_default_and_can_be_extended():
    text = "  Answer\n<|endoftext|>  \n"
    assert response_text.strip_terminal_special_tokens(text) == "  Answer\n"
    assert response_text.strip_terminal_special_tokens(text, stripped_special_tokens=frozenset()) == text
    selected = response_text.STRIPPED_SPECIAL_TOKENS | {"<custom_stop>"}
    assert (
        response_text.strip_terminal_special_tokens("Answer<custom_stop>", stripped_special_tokens=selected)
        == "Answer"
    )


@pytest.mark.parametrize("stop_token", STOP_SEQUENCES)
def test_shared_cleanup_is_idempotent_and_direct_callers_agree(stop_token):
    answer = 'Mention "<|endoftext|>" literally.'
    original = answer + stop_token
    cleaned = response_text.strip_terminal_special_tokens(original)
    assert cleaned == answer
    assert response_text.strip_terminal_special_tokens(cleaned) == cleaned
    assert deterministic_verifiers.answer_text(original) == cleaned
    assert deterministic_verifiers.answer_text(cleaned) == cleaned
    target = {"tool_format": "olmo_pythonic_v1"}
    assert nemo_tool_calls.parse_action(original, target).content == cleaned
    assert nemo_tool_calls.parse_action(cleaned, target).content == cleaned
