"""Offline next-action preparation and native NeMo comparison through MILES."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers.utils import chat_template_utils

from open_instruct import test_miles_grading as grading_tests
from open_instruct.miles.datasets import run_data, source_preparation
from open_instruct.miles.rewards import nemo_tool_calls as tools
from open_instruct.miles.rewards import rewards


def raw_row():
    return {
        "agent_ref": {"name": "single_step_tool_use_with_argument_comparison_agent"},
        "responses_create_params": {
            "instructions": "Be helpful.",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}, "count": {"type": "integer"}},
                    },
                }
            ],
            "input": [
                {"role": "developer", "content": "Use the provided functions."},
                {"role": "user", "content": "Find apples."},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": '{"item":"apples","count":1}',
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "Two varieties available."},
            ],
        },
        "expected_action": {"type": "function_call", "name": "lookup", "arguments": '{"item":"apples","count":2}'},
    }


def test_preserves_tool_history_and_system_context():
    raw = raw_row()
    original = copy.deepcopy(raw)
    messages, target, name, stratum = source_preparation.adapt(raw, "single_step_tool_v1")
    assert raw == original
    assert messages[0] == {"role": "system", "content": "Be helpful.\n\nUse the provided functions."}
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"item": "apples", "count": 1}
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "lookup",
        "content": "Two varieties available.",
    }
    assert target["expected_action"] == raw["expected_action"]
    assert name == "single_step_tool"
    assert stratum == "single_step_tool_v1:conversational:function_call"


@pytest.mark.parametrize(
    "tool_format,response",
    [
        ("olmo_pythonic_v1", '<function_calls>lookup(item="apples", count=2)</function_calls>'),
        (
            "olmo_xml_v1",
            "<tool_call><function=lookup><parameter=item>apples</parameter><parameter=count>2</parameter></function></tool_call>",
        ),
    ],
)
def test_native_formats_through_worker(tool_format, response):
    target = tools.target_for_row(raw_row(), {"tool_format": tool_format})
    verifier = tools.SingleStepToolVerifier()

    async def run():
        assert (await verifier.async_call([], "thinking</think>" + response + "<|im_end|>", target)).score == 1
        assert (
            await verifier.async_call([], response.replace("count=2", "count=3").replace(">2<", ">3<"), target)
        ).score == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    [
        '<function_calls>lookup(item="apples", count=2)',
        '<function_calls>lookup(item="apples", count=__import__("os").system("false"))</function_calls>',
        '<function_calls>lookup(item="apples", count=2, count=2)</function_calls>',
        "<function_calls></function_calls>",
        "<tool_call>wrong format</tool_call>",
        "",
    ],
)
def test_invalid_response_zero(response):
    target = tools.target_for_row(raw_row(), {})
    assert tools.SingleStepToolVerifier()([], response, target).score == 0


def test_native_surplus_call_and_message_semantics():
    target = tools.target_for_row(raw_row(), {})
    verifier = tools.SingleStepToolVerifier()
    assert verifier([], '<function_calls>other()\nlookup(item="apples", count=2)</function_calls>', target).score == 1
    target["expected_action"] = {"type": "message", "content": "Reference prose"}
    assert verifier([], "Different prose is still a message.", target).score == 1
    assert verifier([], '<function_calls>lookup(item="apples", count=2)</function_calls>', target).score == 0


def test_target_errors_remain_fatal():
    raw = raw_row()
    raw["expected_action"]["arguments"] = '{"item":"apples","count":2,"count":3}'
    with pytest.raises(ValueError, match="Duplicate"):
        tools.target_for_row(raw, {})
    raw["expected_action"]["arguments"] = "{}"
    raw["expected_action"]["name"] = "undeclared"
    with pytest.raises(ValueError, match="undeclared"):
        tools.target_for_row(raw, {})


def test_native_swe_string_tolerance_is_not_exact_matching():
    raw = raw_row()
    raw["expected_action"]["arguments"] = json.dumps({"item": "apple pear", "count": 2})
    target = tools.target_for_row(raw, {"profile": "swe_pivot"})
    # Native bag-of-words comparison ignores order, even with a zero tolerance.
    assert (
        tools.SingleStepToolVerifier()(
            [], '<function_calls>lookup(item="pear apple", count=2)</function_calls>', target
        ).score
        == 1
    )


def test_unknown_tool_observation_is_not_dropped():
    raw = raw_row()
    raw["responses_create_params"]["input"][-1]["call_id"] = "missing"
    with pytest.raises(ValueError, match="preceding call"):
        source_preparation.adapt(raw, "single_step_tool_v1")


def test_declared_literal_name_with_spaces_and_parentheses():
    raw = raw_row()
    raw["responses_create_params"]["tools"][0]["name"] = "Lookup (Fruit)"
    raw["expected_action"]["name"] = "Lookup (Fruit)"
    target = tools.target_for_row(raw, {})
    response = '<function_calls>Lookup (Fruit)(item="apples", count=2)</function_calls>'
    assert asyncio.run(tools.SingleStepToolVerifier().async_call([], response, target)).score == 1


def test_history_reasoning_is_preserved_in_content_for_older_templates():
    raw = raw_row()
    raw["responses_create_params"]["input"].insert(
        2, {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Look up the first variety."}]}
    )
    messages, _, _, _ = source_preparation.adapt(raw, "single_step_tool_v1")
    rendered = source_preparation.tool_conversations.for_template(messages, "{{ message.content }}")
    assert rendered[2]["content"] == "<think>Look up the first variety.</think>\n"
    assert messages[2]["content"] == ""
    assert rendered[2]["tool_calls"] == messages[2]["tool_calls"]


def test_unknown_tool_plan_option_is_fatal():
    with pytest.raises(source_preparation.PlanError, match="tool rendering"):
        source_preparation.adapt(raw_row(), "single_step_tool_v1", {"tool_format": "misspelled"})


def thinking_target(expected_action):
    target = tools.target_for_row(raw_row(), {})
    target["expected_action"] = expected_action
    return target


CALL = '<function_calls>lookup(item="apples", count=2)</function_calls>'
MESSAGE = {"type": "message", "content": "Reference prose"}
EXPECTED_CALL = {"type": "function_call", "name": "lookup", "arguments": '{"item":"apples","count":2}'}


def opened(response, expected_action):
    """Score a response to a prompt that ends with the opening reasoning tag."""
    return tools.SingleStepToolVerifier()([], response, thinking_target(expected_action), prompt_opens_reasoning=True)


@pytest.mark.parametrize(
    "response,prompt_opens_reasoning",
    [
        ("I should tell the user the stock levels.", True),  # the prompt opened reasoning; it never closed
        ("<think>I should tell the user the stock levels.", False),  # the response opened it itself
        ("Plan.</think>Here are the varieties.<think>More thought", True),  # reopened and left open
    ],
)
def test_unclosed_reasoning_is_not_a_reply(response, prompt_opens_reasoning):
    verifier = tools.SingleStepToolVerifier()
    result = verifier(
        [], response + "<|endoftext|>", thinking_target(MESSAGE), prompt_opens_reasoning=prompt_opens_reasoning
    )
    assert result.score == 0
    assert result.diagnostics["category"] == "unclosed_thinking"


def test_reply_after_closed_reasoning_still_earns_message_credit():
    assert opened("Plan the reply.</think>We have two varieties.<|im_end|>", MESSAGE).score == 1
    # Closing the reasoning is not itself a reply.
    assert opened("Plan the reply.</think>\n<|im_end|>", MESSAGE).score == 0


def test_calls_inside_reasoning_never_count():
    # A call drafted in unfinished reasoning is not an action.
    assert opened("Maybe " + CALL + " would work", EXPECTED_CALL).score == 0
    # A call drafted in closed reasoning is not an action either; the prose after it is.
    assert opened("Maybe " + CALL + "</think>Let me check.", EXPECTED_CALL).score == 0
    assert opened("Maybe " + CALL + "</think>Let me check.", MESSAGE).score == 1
    assert opened("Check stock.</think>" + CALL, EXPECTED_CALL).score == 1


def test_prompts_without_reasoning_keep_grading_the_whole_response():
    verifier = tools.SingleStepToolVerifier()
    assert verifier([], "We have two varieties.", thinking_target(MESSAGE)).score == 1
    assert verifier([], CALL, thinking_target(EXPECTED_CALL), prompt_opens_reasoning=False).score == 1


@pytest.mark.parametrize(
    "prompt,final_answer_only,expected",
    [
        ("<|im_start|>assistant\n<think>", False, True),
        ("<|im_start|>assistant\n<think>\n", False, True),
        ("<|im_start|>assistant\n", False, False),
        ([{"role": "user", "content": "Find apples."}], False, False),
        # The reasoning has already been removed from the response.
        ("<|im_start|>assistant\n<think>", True, False),
    ],
)
def test_reward_step_reads_reasoning_state_from_the_prompt(prompt, final_answer_only, expected):
    assert rewards._prompt_opens_reasoning(prompt, final_answer_only=final_answer_only) is expected


class TemplateTokenizer:
    def __init__(self, prefix):
        self.chat_template = (Path(__file__).parent / "test_data/olmo35_chat_template.jinja").read_text()
        self.prefix = prefix

    def apply_chat_template(self, messages, **kwargs):
        rendered = chat_template_utils._compile_jinja_template(self.chat_template).render(
            messages=messages,
            tools=kwargs.get("tools", []),
            eos_token="<|endoftext|>",
            add_generation_prompt=kwargs.get("add_generation_prompt", False),
        )
        if kwargs.get("add_generation_prompt"):
            rendered = rendered.removesuffix("<think>") + self.prefix
        return rendered

    def encode(self, text, **kwargs):
        return list(text.encode())


def prepared_row(tmp_path, monkeypatch, prefix, expected_action):
    raw = raw_row()
    raw["expected_action"] = expected_action
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(raw) + "\n")
    receipt = tmp_path / "hygiene.json"
    receipt.write_text('{"test_fixture":true}')
    plan = tmp_path / "plan.json"
    inputs = [
        {"path": str(source), "sha256": source_preparation.file_hash(source), "revision": "test", "split": "train"}
    ]
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "nemo_tools": {
                        "adapter": "single_step_tool_v1",
                        "options": {"tool_format": "olmo_xml_v1"},
                        "inputs": inputs,
                        "hygiene_receipt": {"path": str(receipt), "sha256": source_preparation.file_hash(receipt)},
                    }
                },
            }
        )
    )
    tokenizer = TemplateTokenizer(prefix)
    output = tmp_path / "prepared"
    entry = source_preparation.prepare(plan, output, tokenizer)["sources"]["nemo_tools"]
    hf = tmp_path / "hf"
    hf.mkdir()
    monkeypatch.setattr(run_data, "_tokenizer", lambda path: tokenizer)
    data = {
        "sources": [
            {
                "name": "nemo_tools",
                "fraction": 1.0,
                **{key: str(output / entry[key]["path"]) for key in ("prompt_data", "reward_config")},
            }
        ]
    }
    result = run_data.prepare_data(data, hf, tmp_path / "loaded", max_prompt_length=5000, seed=17)
    return json.loads(Path(result["prompt_data"]).read_text().splitlines()[0]), result["reward_config"]


def reward(row, reward_config, response, *, final_answer_only=False):
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            reward_config=reward_config, reward_zero_truncated=True, reward_final_answer_only=final_answer_only
        )
    )
    sample = SimpleNamespace(
        prompt=row["input"],
        response=response,
        metadata=copy.deepcopy(row["metadata"]),
        tokens=[],
        response_length=0,
        status="completed",
    )
    return asyncio.run(rewards.registered_reward(args, sample))


@pytest.mark.parametrize("prefix", ["", "<think>"])
def test_registered_reward_reads_the_rendered_prompt(tmp_path, monkeypatch, prefix):
    row, reward_config = prepared_row(tmp_path, monkeypatch, prefix, MESSAGE)
    assert row["input"].endswith("<think>") is bool(prefix)
    unfinished = "I should tell the user about the two varieties.<|im_end|>"
    finished = "Two varieties.</think>We have two varieties.<|im_end|>"
    # Unfinished reasoning earns no reply credit only when the prompt opened the reasoning.
    assert reward(row, reward_config, unfinished) == (0.0 if prefix else 1.0)
    assert reward(row, reward_config, finished) == 1.0
    # Final-answer-only grading strips the reasoning first and gates a missing close itself.
    assert reward(row, reward_config, finished, final_answer_only=True) == 1.0
    assert reward(row, reward_config, unfinished, final_answer_only=True) == 0.0


@pytest.mark.parametrize("schema", [{"type": "dict"}, {"properties": []}])
@pytest.mark.parametrize("message_reference", [False, True])
def test_xml_admission_checks_unused_tool_schemas(schema, message_reference):
    raw = raw_row()
    raw["responses_create_params"]["tools"].append({"type": "function", "name": "unused", "parameters": schema})
    if message_reference:
        raw["expected_action"] = {"type": "message", "content": "Hello"}
    with pytest.raises(ValueError, match="Invalid declared tool schema"):
        tools.target_for_row(raw, {"tool_format": "olmo_xml_v1"})


@pytest.mark.parametrize("schema", [{"type": "dict"}, {"properties": []}])
def test_existing_xml_target_with_bad_schema_scores_zero(schema):
    target = tools.target_for_row(raw_row(), {})
    target["tool_format"] = "olmo_xml_v1"
    target["tools"][0]["function"]["parameters"] = schema
    response = "<tool_call><function=lookup><parameter=count>2</parameter></function></tool_call>"
    verifier = tools.SingleStepToolVerifier()
    assert verifier([], response, target).score == 0
    assert asyncio.run(verifier.async_call([], response, target)).score == 0


def test_selected_message_uses_shared_gate_instead_of_reparsing_literal_opener(monkeypatch):
    text = "The literal <think> token belongs to this message."
    target = thinking_target({"type": "message", "content": text})
    verifier = tools.SingleStepToolVerifier()
    sample = grading_tests.item("work</think>" + text, target=target)
    assert grading_tests.score(monkeypatch, sample, verifier) == 1
    assert verifier([], text, target).score == 0  # Preserve direct-call normalization.
    sample.response = "work</think><think>unfinished"
    assert grading_tests.score(monkeypatch, sample, verifier) == 0
    assert sample.metadata["reward_gate"] == "unfinished_reasoning"
