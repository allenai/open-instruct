"""Dataset-derived controls for XML schemas, tool answers and rendering provenance."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import nemo_tool_calls, tool_call_parsing


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
    inputs.write_text(
        "".join(json.dumps({"input": f"prompt {index}", "metadata": metadata}) + "\n" for index in range(3))
    )
    warnings = []
    monkeypatch.setattr(run_data.logger, "warning", lambda *args: warnings.append(args))
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
    assert len(warnings) == (1 if recorded is None else 0)
    warnings.clear()
    before = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
    tokenizer.encode = lambda *a, **k: pytest.fail("Resume must not retokenize")
    if recorded == "old-pythonic":
        with pytest.raises(InputError, match="row 1:.*template hash"):
            run_data.prepare_data(data, hf, output, max_prompt_length=10, seed=17)
    else:
        assert run_data.prepare_data(data, hf, output, max_prompt_length=10, seed=17) == result
    assert len(warnings) == (1 if recorded is None else 0)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_legacy_prepared_row_without_fingerprint_does_not_check_template():
    row = {"input": "original Dolci prompt", "metadata": {"verifiers": [{"name": "v", "target": "answer"}]}}
    tokenizer = SimpleNamespace(encode=lambda *a, **k: [1])
    run_data._verify_row(row, tokenizer, 10, {"v"})
    assert row["input"] == "original Dolci prompt"
    assert "rendering_template_sha256" not in row["metadata"]
