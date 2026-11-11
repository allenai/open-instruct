"""Dataset-derived controls for XML schemas, tool answers and rendering provenance."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.datasets import run_data, source_preparation
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import nemo_tool_calls, structured_outputs, tool_call_parsing


@pytest.mark.parametrize(
    "schema,text,expected",
    [
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, "null", None),
        ({"anyOf": [{"type": "number"}, {"type": "null"}]}, "2.5", 2.5),
        ({"oneOf": [{"type": "object"}, {"type": "null"}]}, '{"x":1}', {"x": 1}),
        ({"type": "string"}, "123", "123"),
        ({"type": ["integer", "null"]}, "null", None),
    ],
)
def test_parameter_union_decoding(schema, text, expected):
    assert tool_call_parsing.parameter_value(text, schema) == expected


def test_parameter_local_reference_keeps_root_scope():
    root = {
        "$defs": {"optional": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
        "properties": {"x": {"$ref": "#/$defs/optional"}},
    }
    assert tool_call_parsing.parameter_value("null", root["properties"]["x"], root) is None
    with pytest.raises(ValueError):
        tool_call_parsing.parameter_value("bad", root["properties"]["x"], root)


def test_xml_structured_payload_retains_schema_reward_and_rejects_wrong_calls():
    target = {
        "format": "json",
        "schema_policy": "as_declared",
        "schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        "response_mode": "tool_call",
        "tool_name": "submit",
        "tool_payload_key": "result",
        "tool_call_format": "olmo_xml_v1",
    }
    response = '<tool_call><function=submit><parameter=result>{"n":3}</parameter></function></tool_call>'
    verifier = structured_outputs.StructuredOutputVerifier()
    assert verifier([], response, target).score == 1
    for invalid in [
        response.replace("submit", "other"),
        response + response,
        response.replace("3}", '"bad"}'),
        response.replace("<tool_call>", ""),
        response.replace('{"n":3}', '{"n":1,"n":3}'),
    ]:
        assert verifier([], invalid, target).score == 0


def test_prepared_template_hash_rejects_different_backbone_template():
    tokenizer = SimpleNamespace(chat_template="native", encode=lambda *a, **k: [1])
    row = {
        "input": "rendered",
        "metadata": {
            "verifiers": [{"name": "v", "target": "x"}],
            "rendering_template_sha256": hashlib.sha256(b"native").hexdigest(),
        },
    }
    run_data._verify_row(row, tokenizer, 10, {"v"})
    tokenizer.chat_template = "different"
    with pytest.raises(InputError, match="template hash"):
        run_data._verify_row(row, tokenizer, 10, {"v"})


def test_new_xml_reference_admission_rejects_schema_conflicts_and_unrepresentable_values():
    raw = {
        "agent_ref": {"name": "single_step_tool_use_with_argument_comparison_agent"},
        "responses_create_params": {
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"anyOf": [{"type": "integer"}, {"type": "null"}]}},
                        "required": ["id"],
                    },
                }
            ]
        },
        "expected_action": {"type": "function_call", "name": "lookup", "arguments": '{"id":null}'},
    }
    target = nemo_tool_calls.target_for_row(raw, {"tool_format": "olmo_xml_v1"})
    response = "<tool_call><function=lookup><parameter=id>null</parameter></function></tool_call>"
    assert nemo_tool_calls.SingleStepToolVerifier()([], response, target).score == 1
    raw["expected_action"]["arguments"] = '{"id":"WD-101"}'
    with pytest.raises(ValueError, match="schema"):
        nemo_tool_calls.target_for_row(raw, {"tool_format": "olmo_xml_v1"})


def test_source_preparation_records_xml_target_and_template_identity(tmp_path):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    raw = {
        "schema_type": "json",
        "schema_str": json.dumps(schema),
        "response_mode": "tool_call",
        "tool_name": "submit",
        "responses_create_params": {
            "input": "Extract n.",
            "tools": [{"type": "function", "name": "submit", "parameters": schema}],
        },
    }
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(raw) + "\n")
    receipt = tmp_path / "hygiene.json"
    receipt.write_text("{}")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "structured": {
                        "adapter": "structured_v1",
                        "options": {"tool_call_format": "olmo_xml_v1"},
                        "inputs": [
                            {
                                "path": str(data),
                                "sha256": source_preparation.file_hash(data),
                                "revision": "pinned",
                                "split": "train",
                            }
                        ],
                        "hygiene_receipt": {"path": str(receipt), "sha256": source_preparation.file_hash(receipt)},
                    }
                },
            }
        )
    )
    tokenizer = SimpleNamespace(
        chat_template="native <tool_call>",
        apply_chat_template=lambda *a, **k: "prompt<think>",
        encode=lambda *a, **k: [1],
    )
    source_preparation.prepare(plan, tmp_path / "output", tokenizer)
    row = json.loads((tmp_path / "output/structured/prompts.jsonl").read_text())
    assert row["label"]["tool_call_format"] == "olmo_xml_v1"
    assert row["metadata"]["rendering_template_sha256"] == hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()


@pytest.mark.parametrize("recorded", [None, "native", "old-pythonic"])
def test_resume_checks_recorded_template_without_retokenizing_or_rewriting(tmp_path, monkeypatch, recorded):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "tokenizer_config.json").write_text("{}")
    tokenizer = SimpleNamespace(chat_template="native", encode=lambda *a, **k: [1])
    monkeypatch.setattr(run_data, "_tokenizer", lambda path: tokenizer)
    metadata = {"verifiers": [{"name": "v", "target": "2"}]}
    if recorded is not None:
        metadata["rendering_template_sha256"] = hashlib.sha256(recorded.encode()).hexdigest()
    inputs = tmp_path / "input.jsonl"
    inputs.write_text(json.dumps({"input": "prompt", "metadata": metadata}) + "\n")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"v": {"factory": "open_instruct.miles.rewards.task_verifiers.MultiplicationVerifier"}})
    )
    data = {"prompt_data": str(inputs), "reward_config": str(registry)}
    output = tmp_path / "prepared"
    # Model a cache created before the recorded hash was enforced.
    with monkeypatch.context() as historical:
        historical.setattr(run_data, "_verify_rendering_template", lambda *a: None)
        result = run_data.prepare_data(data, hf, output, max_prompt_length=10, seed=17)
    before = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    tokenizer.encode = lambda *a, **k: pytest.fail("Resume must not retokenize")
    if recorded == "old-pythonic":
        with pytest.raises(InputError, match="row 1:.*template hash"):
            run_data.prepare_data(data, hf, output, max_prompt_length=10, seed=17)
    else:
        assert run_data.prepare_data(data, hf, output, max_prompt_length=10, seed=17) == result
    assert all(path.read_bytes() == raw for path, raw in before.items())
