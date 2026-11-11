"""Native format coercion, explicit schema policy, and bounded response validation."""

import asyncio
import hashlib
import json
import signal
import time
from types import SimpleNamespace
from unittest import mock

import pytest
from referencing import exceptions

pytest.importorskip("xmltodict")
pytest.importorskip("openapi_schema_validator")

from open_instruct.miles.datasets import source_preparation
from open_instruct.miles.rewards import structured_outputs as structured


def target(kind="json", **extra):
    return {
        "format": kind,
        "schema_policy": "as_declared",
        "coerce_types": True,
        "schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        **extra,
    }


@pytest.mark.parametrize(
    "kind,answer", [("json", '{"n": 3}'), ("yaml", "n: 3"), ("toml", "n = 3"), ("xml", "<n>3</n>")]
)
def test_four_formats_same_schema(kind, answer):
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], answer, target(kind)).score == 1
    assert verifier([], answer.replace("3", "bad"), target(kind)).score == 0


@pytest.mark.parametrize(
    "kind,separator,value",
    [
        ("yaml", ": ", "2024-05-01"),
        ("yaml", ": ", "2024-05-01T12:34:56Z"),
        ("toml", " = ", "2024-05-01"),
        ("toml", " = ", "12:34:56"),
        ("toml", " = ", "2024-05-01T12:34:56Z"),
    ],
)
def test_temporal_scalars_require_quotes(kind, separator, value):
    label = target(kind, schema={"type": "object", "properties": {"n": {"type": "string"}}, "required": ["n"]})
    verifier = structured.StructuredOutputVerifier()
    result = verifier([], f"n{separator}{value}", label)
    assert result.score == 0 and result.diagnostics["status"] == "parse_error"
    assert verifier([], f'n{separator}"{value}"', label).score == 1


@pytest.mark.parametrize("value", ["yes", "no", "on", "off"])
def test_yaml_implicit_booleans_require_quotes_for_strings(value):
    label = target("yaml", schema={"type": "object", "properties": {"n": {"type": "string"}}})
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], f"n: {value}", label).score == 0
    assert verifier([], f'n: "{value}"', label).score == 1
    label["schema"]["properties"]["n"]["type"] = "boolean"
    assert verifier([], f"n: {value}", label).score == 1


def test_xml_root_with_multiple_fields():
    schema = {
        "type": "object",
        "properties": {
            "root": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            }
        },
        "required": ["root"],
    }
    verifier = structured.StructuredOutputVerifier()
    label = target("xml", schema=schema)
    assert verifier([], "<root><a>1</a><b>2</b></root>", label).score == 1
    assert verifier([], "<root><a>1</a></root>", label).score == 0
    assert verifier([], "<root><a>bad</a><b>2</b></root>", label).score == 0


def test_csv_types_and_width():
    label = target("csv", schema={"type": "array", "items": target()["schema"]})
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], "n\n3\n4", label).score == 1
    assert verifier([], "n\nwrong", label).score == 0
    assert verifier([], "n\n3,4", label).score == 0
    assert verifier([], "n,n\n3,4", label).score == 0


@pytest.mark.parametrize("subschema", [True, False])
@pytest.mark.parametrize("location", ["items", "property"])
def test_csv_boolean_subschemas(subschema, location):
    items = subschema if location == "items" else {"type": "object", "properties": {"n": subschema}}
    label = target("csv", schema={"type": "array", "items": items})
    assert structured.StructuredOutputVerifier()([], "n\n3", label).score == float(subschema)


@pytest.mark.parametrize("key", ["1", "true", "null", ".nan"])
@pytest.mark.parametrize("schema", [{"type": "object"}, {"patternProperties": {".*": {"type": "string"}}}])
def test_yaml_nonstring_keys_are_parse_errors(key, schema):
    result = structured.StructuredOutputVerifier()([], f"{key}: value", target("yaml", schema=schema))
    assert result.score == 0 and result.diagnostics["status"] == "parse_error"


def test_schema_strengthening_is_explicit_and_does_not_mutate_target():
    label = target(schema={"type": "object", "properties": {"n": {"type": "integer"}}})
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], "{}", label).score == 1
    assert verifier([], "{}", {**label, "schema_policy": "nemo_strict_v1"}).score == 0
    assert "required" not in label["schema"]


def test_local_references_and_property_names_are_not_confused():
    schema = {
        "type": "object",
        "properties": {"$ref": {"type": "string"}, "n": {"$ref": "#/$defs/num"}},
        "$defs": {"num": {"type": "integer"}},
    }
    assert structured.StructuredOutputVerifier()([], '{"$ref":"literal","n":3}', target(schema=schema)).score == 1
    with pytest.raises(ValueError, match="local schema"):
        structured.validate_target(target(schema={"$ref": "https://example.invalid/schema"}))


def test_registry_refuses_external_retrieval(monkeypatch):
    # Defense in depth if a future schema keyword bypasses the target precheck.
    monkeypatch.setattr(structured, "validate_target", lambda label: {"$ref": "https://example.invalid/schema"})
    with pytest.raises(exceptions.Unresolvable):
        structured.StructuredOutputVerifier()([], "{}", target())


@pytest.mark.parametrize(
    "kind,text",
    [("json", '{"n":1,"n":2}'), ("yaml", "n: 1\nn: 2"), ("yaml", "n: .nan"), ("xml", "<!DOCTYPE n><n>3</n>")],
)
def test_ambiguous_or_unsafe_parse_is_zero(kind, text):
    assert structured.StructuredOutputVerifier()([], text, target(kind)).score == 0


@pytest.mark.parametrize("text", ["n: &n [*n]", "a: &a [1, 2]\nb: &b [*a, *a]\nc: [*b, *b]"])
def test_yaml_aliases_rejected_before_expansion(text):
    result = structured.StructuredOutputVerifier()([], text, target("yaml", schema={}))
    assert result.score == 0 and result.diagnostics["status"] == "parse_error"
    assert "aliases" in result.diagnostics["error"]


def test_outer_worker_timeout_scores_zero(monkeypatch):
    warning = mock.Mock()
    monkeypatch.setattr(structured.logger, "warning", warning)
    monkeypatch.setattr(
        structured.rewards,
        "isolated_verifier_call",
        mock.AsyncMock(side_effect=structured.rewards.MathVerifierTimeout("worker deadline")),
    )
    call = structured.StructuredOutputVerifier().async_call([], "{}", target())
    result = asyncio.run(call)
    assert result.score == 0 and result.diagnostics["status"] == "timeout"
    assert result.diagnostics["failure_policy"] == "zero"
    assert result.diagnostics["elapsed_seconds"] >= 0
    warning.assert_called_once()


def test_unexpected_worker_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        structured.rewards, "isolated_verifier_call", mock.AsyncMock(side_effect=RuntimeError("worker exited"))
    )
    with pytest.raises(RuntimeError, match="worker exited"):
        asyncio.run(structured.StructuredOutputVerifier().async_call([], "{}", target()))


def test_pathological_schema_regex_has_one_response_deadline():
    verifier = structured.StructuredOutputVerifier(structured.StructuredConfig(timeout_seconds=0.05))
    label = target(schema={"type": "string", "pattern": "^(a+)+$"})
    start = time.monotonic()
    result = verifier([], '"' + "a" * 100 + '!"', label)
    assert result.score == 0 and result.diagnostics["status"] == "timeout"
    assert result.diagnostics["failure_policy"] == "zero"
    assert result.diagnostics["elapsed_seconds"] >= 0.05
    assert time.monotonic() - start < 2


def test_alarm_during_cleanup_scores_zero_and_restores_signal(monkeypatch):
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    real_setitimer = signal.setitimer
    cleanup_alarm = mock.Mock()

    def alarm_on_disarm(which, seconds, interval=0):
        if seconds == 0 and not cleanup_alarm.called:
            cleanup_alarm()
            handler = signal.getsignal(signal.SIGALRM)
            assert callable(handler)
            handler(signal.SIGALRM, None)
        return real_setitimer(which, seconds, interval)

    monkeypatch.setattr(signal, "setitimer", alarm_on_disarm)
    try:
        verifier = structured.StructuredOutputVerifier()
        result = verifier([], '{"n": 3}', target())
        assert result.score == 0 and result.diagnostics["status"] == "timeout"
        assert result.diagnostics["failure_policy"] == "zero"
        cleanup_alarm.assert_called_once()
        assert signal.getsignal(signal.SIGALRM) == previous_handler
        assert signal.getitimer(signal.ITIMER_REAL) == previous_timer
    finally:
        real_setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def test_real_worker_path_and_bad_target_error():
    verifier = structured.StructuredOutputVerifier()
    assert asyncio.run(verifier.async_call([], "n = 3", target("toml"))).score == 1
    with pytest.raises(RuntimeError, match="schema_policy"):
        asyncio.run(verifier.async_call([], "{}", {"format": "json", "schema": {}}))


def test_real_worker_timeout_logs_in_parent(monkeypatch):
    warning = mock.Mock()
    monkeypatch.setattr(structured.logger, "warning", warning)
    verifier = structured.StructuredOutputVerifier(structured.StructuredConfig(timeout_seconds=0.05))
    label = target(schema={"type": "string", "pattern": "^(a+)+$"})
    result = asyncio.run(verifier.async_call([], '"' + "a" * 100 + '!"', label))
    assert result.score == 0 and result.diagnostics["status"] == "timeout"
    assert result.diagnostics["failure_policy"] == "zero"
    assert result.diagnostics["elapsed_seconds"] >= 0.05
    warning.assert_called_once()


@pytest.mark.parametrize(
    "schema,prediction,error_type",
    [
        (
            {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"$ref": "#"}}]},
            "[" * 200 + "0" + "]" * 200,
            "RecursionError",
        ),
        ({"type": "number", "multipleOf": 0.1}, "9" * 310, "OverflowError"),
    ],
    ids=["recursive-arrays", "huge-integer-multiple"],
)
@pytest.mark.parametrize("response_mode", ["text", "xml_tool"])
def test_async_instance_validation_limits_score_zero(schema, prediction, error_type, response_mode):
    label = target(schema=schema)
    if response_mode == "xml_tool":
        label.update(
            response_mode="tool_call", tool_name="submit", tool_payload_key="result", tool_call_format="olmo_xml_v1"
        )

    def response(body):
        if response_mode == "xml_tool":
            return f"<tool_call><function=submit><parameter=result>{body}</parameter></function></tool_call>"
        return body

    # Both inputs pass parsing/serialization and target validation. The failure
    # happens only when the schema validator traverses/calculates on the instance.
    structured.parse(prediction, "json", structured.validate_target(label), False)

    async def score_and_recover():
        verifier = structured.StructuredOutputVerifier()
        result = await verifier.async_call([], response(prediction), label)
        assert result.score == 0
        assert result.diagnostics["status"] == "validation_error"
        assert result.diagnostics["error_type"] == error_type
        assert result.diagnostics["failure_policy"] == "zero"
        assert result.diagnostics["elapsed_seconds"] >= 0
        assert result.diagnostics["error"]
        assert (await verifier.async_call([], response("1"), label)).score == 1

    asyncio.run(score_and_recover())


@pytest.mark.parametrize(
    "schema,error_type", [({"type": "invalid"}, "SchemaError"), ({"$ref": "#/$defs/missing"}, "PointerToNowhere")]
)
def test_async_invalid_schema_and_reference_remain_fatal(schema, error_type):
    with pytest.raises(RuntimeError, match=error_type):
        asyncio.run(structured.StructuredOutputVerifier().async_call([], "0", target(schema=schema)))


@pytest.mark.parametrize("error_type", [RecursionError, OverflowError])
@pytest.mark.parametrize("stage", ["validate_target", "OAS31Validator"])
def test_target_and_constructor_resource_errors_remain_fatal(monkeypatch, error_type, stage):
    monkeypatch.setattr(structured, stage, mock.Mock(side_effect=error_type("invalid target")))
    with pytest.raises(error_type, match="invalid target"):
        structured.StructuredOutputVerifier()([], "0", target())


def test_strictification_visits_schemas_not_property_maps_or_defaults():
    schema = {
        "type": "object",
        "properties": {
            "properties": {"type": "string"},
            "default": {"type": "object", "default": {"properties": {"literal": 1}}},
            "variants": {"anyOf": [{"type": "object", "properties": {"x": {"type": "integer"}}}]},
        },
    }
    label = target(schema=schema, schema_policy="nemo_strict_v1")
    validated = structured.validate_target(label)
    assert set(validated["properties"]) == {"properties", "default", "variants"}
    assert validated["properties"]["default"]["default"] == {"properties": {"literal": 1}}
    assert "required" not in validated["properties"]["variants"]["anyOf"][0]
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], '{"properties":"literal","default":{},"variants":{"x":1}}', label).score == 1
    assert verifier([], '{"properties":"literal","default":{},"variants":{}}', label).score == 1
    assert "required" not in schema


def test_nested_resource_identifier_resolves_its_own_local_reference():
    schema = {
        "$id": "https://example.invalid/root",
        "type": "object",
        "properties": {
            "child": {
                "$id": "child",
                "$defs": {"num": {"type": "integer"}},
                "type": "object",
                "properties": {"n": {"$ref": "#/$defs/num"}},
            }
        },
    }
    verifier = structured.StructuredOutputVerifier()
    label = target(schema=schema)
    assert verifier([], '{"child":{"n":1}}', label).score == 1
    assert verifier([], '{"child":{"n":"bad"}}', label).score == 0


@pytest.mark.parametrize("ref_key", ["$ref", "$dynamicRef"])
def test_nested_identifier_cannot_resolve_root_definitions_at_preparation(ref_key):
    label = target(
        schema={
            "$id": "https://example.invalid/root",
            "$defs": {"num": {"type": "integer"}},
            "type": "object",
            "properties": {"child": {"$id": "child", ref_key: "#/$defs/num"}},
        }
    )
    with pytest.raises(ValueError, match="Unresolvable schema reference"):
        structured.validate_target(label)
    # Invalid configuration is fatal even when this instance omits the child;
    # it must not become a response-dependent zero reward or worker crash.
    with pytest.raises(RuntimeError, match="Unresolvable schema reference"):
        asyncio.run(structured.StructuredOutputVerifier().async_call([], "{}", label))


def tool_target(**extra):
    return target(
        response_mode="tool_call",
        tool_name="submit",
        tool_payload_key="result",
        tool_call_format="olmo3_pythonic_v1",
        **extra,
    )


def test_structured_tool_answer_through_worker():
    verifier = structured.StructuredOutputVerifier()
    result = asyncio.run(
        verifier.async_call([], '<function_calls>submit(result={"n":3})</function_calls>', tool_target())
    )
    assert result.score == 1
    assert result.diagnostics["status"] == "completed"


@pytest.mark.parametrize(
    "response",
    [
        '{"n":3}',
        '<function_calls>wrong(result={"n":3})</function_calls>',
        '<function_calls>submit(result={"n":3})\nsubmit(result={"n":3})</function_calls>',
        '<function_calls>submit(result={"n":"wrong"})</function_calls>',
        "<function_calls>submit(n=3)</function_calls>",
        '<function_calls>submit(result={"n":1,"n":3})</function_calls>',
        '<function_calls>submit(result={"n":3}, result={"n":3})</function_calls>',
        '<function_calls>submit(result={"n":__import__("os").getpid()})</function_calls>',
        '<function_calls>submit(result={"n":3}, **{"result":{"n":3}})</function_calls>',
    ],
)
def test_structured_tool_answer_rejects_wrong_ambiguous_or_executable_calls(response):
    assert structured.StructuredOutputVerifier()([], response, tool_target()).score == 0


def test_tool_answers_accept_template_json_literals():
    label = tool_target(
        schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "value": {"type": "null"}}}
    )
    assert (
        structured.StructuredOutputVerifier()(
            [], '<function_calls>submit(result={"ok":true,"value":null})</function_calls>', label
        ).score
        == 1
    )


def test_tool_configuration_errors_remain_fatal():
    label = tool_target()
    label["tool_name"] = "not.a.function"
    with pytest.raises(ValueError, match="tool_name"):
        structured.validate_target(label)


def test_tool_answer_can_represent_a_reserved_field_name():
    label = target(
        response_mode="tool_call",
        tool_name="submit",
        tool_call_format="olmo3_pythonic_v1",
        schema={"type": "object", "properties": {"class": {"type": "string"}}, "required": ["class"]},
    )
    result = asyncio.run(
        structured.StructuredOutputVerifier().async_call(
            [], '<function_calls>submit(**{"class":"Mammalia"})</function_calls>', label
        )
    )
    assert result.score == 1


def test_native_strictification_does_not_close_allof_branches():
    schema = {
        "type": "object",
        "allOf": [
            {"properties": {"a": {"type": "integer"}}, "required": ["a"]},
            {"properties": {"b": {"type": "integer"}}, "required": ["b"]},
        ],
    }
    label = target(schema=schema, schema_policy="nemo_strict_v1")
    assert structured.validate_target(label) == schema
    verifier = structured.StructuredOutputVerifier()
    assert asyncio.run(verifier.async_call([], '{"a":1,"b":2}', label)).score == 1
    assert asyncio.run(verifier.async_call([], '{"a":1}', label)).score == 0
    assert asyncio.run(verifier.async_call([], '{"a":1,"b":"wrong"}', label)).score == 0


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
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], response, target).score == 1
    for invalid in [
        response.replace("submit", "other"),
        response + response,
        response.replace("3}", '"bad"}'),
        response.replace("<tool_call>", ""),
        response.replace('{"n":3}', '{"n":1,"n":3}'),
    ]:
        assert verifier([], invalid, target).score == 0


@pytest.mark.parametrize("root_ref", [False, True])
def test_source_preparation_records_xml_target_and_template_identity(tmp_path, root_ref):
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    if root_ref:
        schema = {"$defs": {"payload": schema}, "$ref": "#/$defs/payload"}
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
    assert row["label"]["schema_policy"] == "nemo_strict_v1"
    response = "<tool_call><function=submit><parameter=n>3</parameter></function></tool_call>"
    assert asyncio.run(structured.StructuredOutputVerifier().async_call([], response, row["label"])).score == 1
    assert row["metadata"]["rendering_template_sha256"] == hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()


@pytest.mark.parametrize(
    "schema",
    [
        {
            "$defs": {"payload": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}},
            "$ref": "#/$defs/payload",
        },
        {
            "$defs": {"payload": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}},
            "$dynamicRef": "#/$defs/payload",
        },
        {"allOf": [{"type": "object", "properties": {"n": {"type": "integer"}}}, {"required": ["n"]}]},
        {
            "$id": "https://example.invalid/root",
            "$defs": {
                "payload": {
                    "$id": "child",
                    "$defs": {"value": {"type": "integer"}},
                    "type": "object",
                    "properties": {"n": {"$ref": "#/$defs/value"}},
                    "required": ["n"],
                }
            },
            "$ref": "#/$defs/payload",
        },
    ],
    ids=["root-ref", "root-dynamic-ref", "root-allof", "nested-resource-scope"],
)
@pytest.mark.parametrize("payload_key", [None, "result"])
@pytest.mark.parametrize("schema_policy", ["as_declared", "nemo_strict_v1"])
def test_xml_answers_resolve_root_property_schemas(schema, payload_key, schema_policy):
    target = {
        "format": "json",
        "schema_policy": schema_policy,
        "schema": schema,
        "response_mode": "tool_call",
        "tool_name": "submit",
        "tool_payload_key": payload_key,
        "tool_call_format": "olmo_xml_v1",
    }
    parameter = "<parameter=n>3</parameter>" if payload_key is None else '<parameter=result>{"n":3}</parameter>'
    response = f"<tool_call><function=submit>{parameter}</function></tool_call>"
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], response, target).score == 1
    assert verifier([], response.replace("3", "3.5"), target).score == 0


def test_xml_allof_properties_decode_independently_of_required_siblings():
    schema = {
        "allOf": [
            {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
            {"properties": {"active": {"type": "boolean"}}, "required": ["active"]},
        ]
    }
    target = {
        "format": "json",
        "schema_policy": "as_declared",
        "schema": schema,
        "response_mode": "tool_call",
        "tool_name": "submit",
        "tool_call_format": "olmo_xml_v1",
    }
    response = "<tool_call><function=submit><parameter=n>3</parameter><parameter=active>true</parameter></function></tool_call>"
    assert structured.StructuredOutputVerifier()([], response, target).score == 1


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_xml_property_projection_leaves_branch_selection_to_final_validation(keyword):
    branches = [
        {
            "type": "object",
            "properties": {"n": {"type": "integer"}, "side": {"enum": [side]}},
            "required": ["n", "side"],
        }
        for side in ["left", "right"]
    ]
    target = {
        "format": "json",
        "schema_policy": "as_declared",
        "schema": {keyword: branches},
        "response_mode": "tool_call",
        "tool_name": "submit",
        "tool_call_format": "olmo_xml_v1",
    }
    response = (
        "<tool_call><function=submit><parameter=n>3</parameter><parameter=side>left</parameter></function></tool_call>"
    )
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], response, target).score == 1
    assert verifier([], response.replace("left", "neither"), target).score == 0


@pytest.mark.parametrize("arguments", ["result=", 'result={"n":3} x', "**"])
def test_malformed_pythonic_answers_score_zero_and_worker_recovers(arguments):
    verifier = structured.StructuredOutputVerifier()
    response = f"<function_calls>submit({arguments})</function_calls>"
    assert verifier([], response, tool_target()).score == 0

    async def run():
        result = await verifier.async_call([], response, tool_target())
        assert result.score == 0
        assert result.diagnostics["status"] == "parse_error"
        assert (
            await verifier.async_call([], '<function_calls>submit(result={"n":3})</function_calls>', tool_target())
        ).score == 1

    asyncio.run(run())
