"""Offline call grading, split isolation, and MILES adoption without model loading."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.data.rlvr import toolrl
from transformers.utils import chat_template_utils

from open_instruct import toolrl_utils
from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import rewards

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "giveaways",
            "description": "games",
            "parameters": {"type": "object", "properties": {"platform": {"type": "string"}}},
        },
    }
]
TEMPLATE = (Path(__file__).parent / "test_data/olmo35_chat_template.jinja").read_text()


def completion(calls=None, *, reasoning="plan", response="answer", retain_terminal=False):
    message = {"role": "assistant", "content": response if calls is None else "", "reasoning_content": reasoning}
    if calls is not None:
        message["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["parameters"]}} for c in calls]
    rendered = chat_template_utils._compile_jinja_template(TEMPLATE).render(
        messages=[message], tools=[], eos_token="<|endoftext|>", add_generation_prompt=False
    )
    response = rendered.split("<|im_start|>assistant\n")[1]
    return response if retain_terminal else response.removesuffix("<|im_end|>\n").removesuffix("<|endoftext|>")


def target(calls=None, tools=None):
    return {"format": "olmo-toolrl-v1", "calls": calls or [], "tools": tools or TOOLS}


def source_completion(calls):
    return "<think>plan</think>\n<tool_call>\n" + "\n".join(json.dumps(c) for c in calls) + "\n</tool_call>"


def call(platform):
    return {"name": "giveaways", "parameters": {"platform": platform}}


def test_parallel_call_rewards_and_order():
    expected = target([call("xbox"), call("android")])
    correct = toolrl_utils.score_response(completion([call("android"), call("xbox")]), expected)
    missing = toolrl_utils.score_response(completion([call("xbox")]), expected)
    duplicated = toolrl_utils.score_response(completion([call("xbox"), call("xbox")]), expected)
    wrong = toolrl_utils.score_response(completion([{"name": "chess", "parameters": {}}]), expected)
    assert correct.score == 4
    assert wrong.score < missing.score < correct.score
    assert duplicated.correctness_score < correct.correctness_score
    assert missing.correctness_score == pytest.approx(0)


def test_no_call_targets_only_check_format_not_answer_facts():
    expected = target()
    result = toolrl_utils.score_response(completion(response="a different answer"), expected)
    assert result.score == result.format_score == 1
    assert result.correctness_score == 0
    assert toolrl_utils.score_response(completion([call("xbox")]), expected).score == 0


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"name":"giveaways","parameters":{"platform":"xbox","platform":"android"}}',
        '{"name":"giveaways","parameters":{"platform":NaN}}',
        '{"name":"giveaways","parameters":[]}',
        '{"name":"giveaways","parameters":{},"extra":true}',
        "",
    ],
)
def test_malformed_calls_cannot_earn_correctness(payload):
    prediction = "<think>plan</think>\n<tool_call>\n" + payload + "\n</tool_call>"
    result = toolrl_utils.score_response(prediction, target([call("xbox")]))
    assert result.correctness_score == -3


def test_reasoning_only_and_malformed_tail_do_not_supply_actions():
    reference = target([call("xbox")])
    correct = completion([call("xbox")])
    quoted = correct.split("</think>")[1]
    for prediction in (completion(reasoning=quoted), "<think>" + quoted, correct + "\n<tool_call>"):
        assert toolrl_utils.score_response(prediction, reference).correctness_score == -3


def test_forced_think_prefix_is_explicit():
    reference = target([call("xbox")])
    correct = completion([call("xbox")])
    suffix = correct.removeprefix("<think>")
    assert toolrl_utils.score_response(suffix, reference).format_score == 0
    assert toolrl_utils.score_response(suffix, reference, think_prefix_in_prompt=True).score == 4
    assert toolrl_utils.score_response(correct, reference, think_prefix_in_prompt=True).format_score == 0


@pytest.mark.parametrize("forced_prefix", [False, True])
@pytest.mark.parametrize("calls", [None, [call("xbox")], [call("xbox"), call("android")]])
def test_native_template_terminal_tokens_through_grader(calls, forced_prefix):
    prediction = completion(calls, retain_terminal=True)
    if forced_prefix:
        prediction = prediction.removeprefix("<think>")
    result = toolrl_utils.score_response(prediction, target(calls), think_prefix_in_prompt=forced_prefix)
    assert result.score == (4 if calls else 1)


@pytest.mark.parametrize("terminal", ["<|im_end|>", "<|endoftext|>"])
def test_terminal_normalization_preserves_arguments_and_rejects_bad_call_tails(terminal):
    calls = [call("literal " + terminal)]
    prediction = completion(calls)
    reference = target(calls)
    assert toolrl_utils.score_response(prediction + terminal + "\n", reference).score == 4
    for tail in (terminal + terminal, terminal + "garbage", "garbage" + terminal, "<tool_call>" + terminal):
        assert toolrl_utils.score_response(prediction + tail, reference).score == -3
    assert toolrl_utils.score_response("<think></think>" + terminal, target()).score == 0


def test_bad_target_raises_instead_of_punishing_model():
    with pytest.raises(ValueError, match="reference"):
        toolrl_utils.score_response("anything", "malformed reference")


def source_rows():
    instruction = '**Available Tools**\n1. Name: giveaways\nDescription: games\nParameters: {"platform": {"type": "string"}}\n**Steps for Each Turn**'
    rows = [
        {
            "instruction": instruction,
            "input": f"**Dialogue Records History**\n<user>question {index}</user>",
            "output": source_completion([call(str(index))]),
        }
        for index in range(12)
    ]
    duplicate = copy.deepcopy(rows[0])
    duplicate["instruction"] = instruction.replace("Description: games", "Description: other description")
    rows.append(duplicate)
    return rows


def test_preparation_preserves_source_and_keeps_targets_out_of_prompt():
    source = source_rows()
    before = copy.deepcopy(source)
    partitions, summary = toolrl.convert_rows(source, eval_count=3)
    assert source == before
    assert sum(map(len, partitions.values())) == len(source)
    train = {row["messages"][0]["content"] for row in partitions["train"]}
    evaluation = {row["messages"][0]["content"] for row in partitions["eval"]}
    assert not train & evaluation
    assert summary["unique_dialogues"] == 12
    assert toolrl.convert_rows(source, eval_count=3) == (partitions, summary)
    for rows in partitions.values():
        for row in rows:
            index = row["metadata"]["source_row"]
            assert row["messages"] == [{"role": "user", "content": f"question {index if index < 12 else 0}"}]
            assert row["ground_truth"] == row["metadata"]["verifiers"][0]["target"]
            assert row["ground_truth"]["format"] == "olmo-toolrl-v1"


def test_source_digest_checked_before_writing(tmp_path):
    with pytest.raises(ValueError, match="SHA256"):
        toolrl.prepare(b"[]", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_invalid_reference_is_recorded_without_renumbering_source_rows():
    source = source_rows()
    source[2]["output"] = source_completion([{"name": "", "parameters": {}}])
    partitions, summary = toolrl.convert_rows(source, eval_count=3)
    indices = {row["metadata"]["source_row"] for rows in partitions.values() for row in rows}
    assert indices == set(range(len(source))) - {2}
    assert summary["retained_rows"] == len(source) - 1
    assert summary["rejected_rows"][0]["source_row"] == 2


class Tokenizer:
    def __init__(self, prefix):
        self.chat_template = TEMPLATE
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
def test_manifest_to_miles_to_reward_and_resume(tmp_path, monkeypatch, prefix):
    # Use the production writer on an explicitly substituted, local test source.
    source = source_rows() * 308
    source = source[:4000]
    raw = json.dumps(source).encode()
    monkeypatch.setattr(toolrl, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    root = tmp_path / "source"
    toolrl.prepare(raw, root, eval_count=500, think_prefix_in_prompt=bool(prefix))
    hf = tmp_path / "hf"
    hf.mkdir()
    tokenizer = Tokenizer(prefix)
    monkeypatch.setattr(run_data, "_tokenizer", lambda path: tokenizer)
    data = {"rl_manifest": str(root / "rl-manifest.json")}
    result = run_data.prepare_data(data, hf, tmp_path / "prepared", max_prompt_length=5000, seed=17)
    assert run_data.prepare_data(data, hf, tmp_path / "prepared", max_prompt_length=5000, seed=17) == result
    row = json.loads(Path(result["prompt_data"]).read_text().splitlines()[0])
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(
            reward_config=result["reward_config"], reward_zero_truncated=True, reward_final_answer_only=False
        )
    )
    sample = SimpleNamespace(
        prompt=row["input"],
        response=completion(row["label"]["calls"], retain_terminal=True).removeprefix(prefix)
        if prefix
        else completion(row["label"]["calls"], retain_terminal=True),
        metadata=row["metadata"],
        tokens=[],
        response_length=0,
        status="completed",
    )
    assert asyncio.run(rewards.registered_reward(args, sample)) == 4
    tokenizer.prefix = "" if prefix else "<think>"
    with pytest.raises(InputError, match="generation prefix"):
        run_data.prepare_data(data, hf, tmp_path / "wrong-prefix", max_prompt_length=5000, seed=17)
    tokenizer.prefix = "<think>\n"
    with pytest.raises(InputError, match="whitespace"):
        run_data.prepare_data(data, hf, tmp_path / "whitespace-prefix", max_prompt_length=5000, seed=17)
    tokenizer.chat_template = "{{ messages[-1]['content'] }}"
    with pytest.raises(InputError, match="XML tool chat template"):
        run_data.prepare_data(data, hf, tmp_path / "wrong-template", max_prompt_length=5000, seed=17)
    with pytest.raises(FileExistsError):
        toolrl.prepare(raw, root, eval_count=500)


def test_structured_history_has_tool_turn_and_native_rendering():
    history = (
        "**Dialogue Records History**\n<user>games?</user>\n"
        + source_completion([call("xbox")])
        + '\n<obs>[{"results": ["a"]}]</obs>'
    )
    messages = toolrl.source_messages(history, TOOLS)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    rendered = Tokenizer("<think>").apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True)
    assert "<function=giveaways>" in rendered
    assert "<|im_start|>environment\n<tool_response>" in rendered
    assert "<obs>" not in rendered
    assert rendered.endswith("<think>")


def test_typed_parameters_and_duplicate_rejection():
    schemas = {
        "n": {"type": "integer"},
        "values": {"type": "array"},
        "enabled": {"type": "boolean"},
        "text": {"type": "string"},
    }
    tools = [{"function": {"name": "f", "parameters": {"properties": schemas}}}]
    calls = [
        {"name": "f", "parameters": {"n": 3, "values": [1, "x"], "enabled": True, "text": "a & b < c\nnext line"}}
    ]
    prediction = completion(calls)
    assert toolrl_utils.score_response(prediction, target(calls, tools)).score == 4
    duplicate = prediction.replace("</function>", "<parameter=n>4</parameter></function>")
    assert toolrl_utils.score_response(duplicate, target(calls, tools)).correctness_score == -3
    assert (
        toolrl_utils.score_response(
            prediction.replace("<parameter=n>\n3", "<parameter=n>\nNaN"), target(calls, tools)
        ).correctness_score
        == -3
    )


def test_old_target_and_old_response_format_are_rejected():
    with pytest.raises(ValueError, match="regenerate"):
        toolrl_utils.score_response(completion([call("xbox")]), source_completion([call("xbox")]))
    assert toolrl_utils.score_response(source_completion([call("xbox")]), target([call("xbox")])).score == -3
    assert toolrl_utils.score_response("<think>plan</think><response>answer</response>", target()).score == 0


def test_invalid_source_calls_are_audited():
    rows = source_rows()
    rows[0]["output"] = source_completion([{"name": "absent", "parameters": {}}])
    rows[1]["output"] = source_completion([{"name": "giveaways", "parameters": {"absent": 1}}])
    _, summary = toolrl.convert_rows(rows, eval_count=3)
    assert [r["source_row"] for r in summary["rejected_rows"]] == [0, 1]


def test_schema_translation_and_canonical_numeric_target():
    assert toolrl.parameter_schema({"type": "List[List[int]]"}) == {
        "type": "array",
        "items": {"type": "array", "items": {"type": "integer"}},
    }
    assert toolrl.parameter_schema({"type": "str, optional"}) == {"type": "string"}
    tools = [{"function": {"name": "f", "parameters": {"properties": {"n": {"type": "integer"}}}}}]
    assert toolrl.canonical_calls([{"name": "f", "parameters": {"n": "3"}}], tools) == [
        {"name": "f", "parameters": {"n": 3}}
    ]
    with pytest.raises(ValueError):
        toolrl.canonical_calls([{"name": "f", "parameters": {"n": "3 * 4"}}], tools)
