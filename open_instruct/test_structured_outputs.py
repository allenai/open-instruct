"""Native format coercion, explicit schema policy, and bounded response validation."""

import asyncio
import time

import pytest

pytest.importorskip("xmltodict")
pytest.importorskip("openapi_schema_validator")

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


def test_csv_types_and_width():
    label = target("csv", schema={"type": "array", "items": target()["schema"]})
    verifier = structured.StructuredOutputVerifier()
    assert verifier([], "n\n3\n4", label).score == 1
    assert verifier([], "n\nwrong", label).score == 0
    assert verifier([], "n\n3,4", label).score == 0
    assert verifier([], "n,n\n3,4", label).score == 0


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


@pytest.mark.parametrize(
    "kind,text",
    [("json", '{"n":1,"n":2}'), ("yaml", "n: 1\nn: 2"), ("yaml", "n: .nan"), ("xml", "<!DOCTYPE n><n>3</n>")],
)
def test_ambiguous_or_unsafe_parse_is_zero(kind, text):
    assert structured.StructuredOutputVerifier()([], text, target(kind)).score == 0


def test_pathological_schema_regex_has_one_response_deadline(monkeypatch):
    monkeypatch.setenv("OI_MILES_FORMAT_TIMEOUT_POLICY", "zero")
    verifier = structured.StructuredOutputVerifier(structured.StructuredConfig(timeout_seconds=0.05))
    label = target(schema={"type": "string", "pattern": "^(a+)+$"})
    start = time.monotonic()
    result = verifier([], '"' + "a" * 100 + '!"', label)
    assert result.score == 0 and result.diagnostics["status"] == "timeout"
    assert time.monotonic() - start < 2


def test_real_worker_path_and_bad_target_error():
    verifier = structured.StructuredOutputVerifier()
    assert asyncio.run(verifier.async_call([], "n = 3", target("toml"))).score == 1
    with pytest.raises(RuntimeError, match="schema_policy"):
        asyncio.run(verifier.async_call([], "{}", {"format": "json", "schema": {}}))
