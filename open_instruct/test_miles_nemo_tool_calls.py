"""Offline next-action preparation and native NeMo comparison through MILES."""

import asyncio
import copy
import json

import pytest

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
