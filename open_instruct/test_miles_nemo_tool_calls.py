"""Offline next-action preparation and native NeMo comparison through MILES."""

import asyncio
import copy
import json
from pathlib import Path

import pytest
from transformers.utils import chat_template_utils

from open_instruct.miles.datasets import source_preparation
from open_instruct.miles.rewards import nemo_tool_calls as tools


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


def thinking_target(expected_action, think_prefix_in_prompt=True):
    target = tools.target_for_row(raw_row(), {})
    target["expected_action"] = expected_action
    target["think_prefix_in_prompt"] = think_prefix_in_prompt
    return target


CALL = '<function_calls>lookup(item="apples", count=2)</function_calls>'
MESSAGE = {"type": "message", "content": "Reference prose"}


@pytest.mark.parametrize(
    "response",
    [
        "I should tell the user the stock levels.",  # the prompt opened reasoning; it never closed
        "<think>I should tell the user the stock levels.",  # the response opened it itself
        "Plan.</think>Here are the varieties.<think>More thought",  # reopened and left open
    ],
)
def test_unclosed_reasoning_is_not_a_reply(response):
    verifier = tools.SingleStepToolVerifier()
    result = verifier([], response + "<|endoftext|>", thinking_target(MESSAGE))
    assert result.score == 0
    assert result.diagnostics["category"] == "unclosed_thinking"


def test_reply_after_closed_reasoning_still_earns_message_credit():
    verifier = tools.SingleStepToolVerifier()
    assert verifier([], "Plan the reply.</think>We have two varieties.<|im_end|>", thinking_target(MESSAGE)).score == 1
    # Closing the reasoning is not itself a reply.
    assert verifier([], "Plan the reply.</think>\n<|im_end|>", thinking_target(MESSAGE)).score == 0


def test_calls_inside_reasoning_never_count():
    verifier = tools.SingleStepToolVerifier()
    expected_call = {"type": "function_call", "name": "lookup", "arguments": '{"item":"apples","count":2}'}
    # A call drafted in unfinished reasoning is not an action.
    assert verifier([], "Maybe " + CALL + " would work", thinking_target(expected_call)).score == 0
    # A call drafted in closed reasoning is not an action either; the prose after it is the action.
    assert verifier([], "Maybe " + CALL + "</think>Let me check.", thinking_target(expected_call)).score == 0
    assert verifier([], "Maybe " + CALL + "</think>Let me check.", thinking_target(MESSAGE)).score == 1
    assert verifier([], "Check stock.</think>" + CALL, thinking_target(expected_call)).score == 1


def test_non_thinking_prompts_keep_grading_the_whole_response():
    verifier = tools.SingleStepToolVerifier()
    assert verifier([], "We have two varieties.", thinking_target(MESSAGE, think_prefix_in_prompt=False)).score == 1
    # Labels prepared before the flag existed keep their previous behaviour.
    assert verifier([], CALL, tools.target_for_row(raw_row(), {})).score == 1


def test_think_prefix_flag_must_be_boolean():
    target = thinking_target(MESSAGE)
    target["think_prefix_in_prompt"] = "yes"
    with pytest.raises(ValueError, match="boolean"):
        tools.SingleStepToolVerifier()([], "Reply.", target)


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


@pytest.mark.parametrize("prefix", ["", "<think>"])
def test_preparation_records_whether_the_prompt_opens_reasoning(tmp_path, prefix):
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(raw_row()) + "\n")
    receipt = tmp_path / "hygiene.json"
    receipt.write_text('{"test_fixture":true}')
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "nemo_tools": {
                        "adapter": "single_step_tool_v1",
                        "options": {"tool_format": "olmo_xml_v1"},
                        "inputs": [
                            {
                                "path": str(source),
                                "sha256": source_preparation.file_hash(source),
                                "revision": "test",
                                "split": "train",
                            }
                        ],
                        "hygiene_receipt": {"path": str(receipt), "sha256": source_preparation.file_hash(receipt)},
                    }
                },
            }
        )
    )
    tokenizer = TemplateTokenizer(prefix)
    output = tmp_path / "prepared"
    catalog = source_preparation.prepare(plan, output, tokenizer)
    entry = catalog["sources"]["nemo_tools"]
    rows = [json.loads(line) for line in (output / entry["prompt_data"]["path"]).read_text().splitlines()]
    (target,) = [spec["target"] for spec in rows[0]["metadata"]["verifiers"]]
    assert target["think_prefix_in_prompt"] is bool(prefix)
    tokenizer.prefix = "<think>\n"
    with pytest.raises(source_preparation.PlanError, match="whitespace"):
        source_preparation.prepare(plan, tmp_path / "whitespace-prefix", tokenizer)
