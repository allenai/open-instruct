"""Adversarial target/response and immutable catalog contracts."""

import asyncio
import hashlib
import json

import pytest

from open_instruct.miles.datasets import source_catalog
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import candidate_verifiers as cv
from open_instruct.miles.rewards import json_schema


@pytest.mark.parametrize(
    "target,positive,negative",
    [
        ({"kind": "mcqa", "value": "B", "choices": ["A", "B"]}, "Reasoning A\nAnswer: B", "B is plausible"),
        ({"kind": "numeric", "value": "2"}, "\\boxed{2.00}", "2 molecules"),
        ({"kind": "exact_text", "value": "uncle"}, "<answer>UNCLE</answer>", "not an uncle"),
        (
            {"kind": "knights", "value": {"Ann": True, "Bob": False}},
            "Ann is a knight. Bob is a knave.",
            "Ann is a knight. Ann is a knight.",
        ),
        ({"kind": "countdown", "value": 24, "nums": [3, 5, 3]}, "\\boxed{(3+5)*3}", "24"),
        ({"kind": "json_exact", "value": {"x": True}}, '{"x":true}', '{"x":1}'),
        (
            {"kind": "format", "value": {"type": "regex", "verify_regex": ["^## .{3,40}$"], "verify_min_matches": 2}},
            "## First\n## Second",
            "## Only",
        ),
        (
            {
                "kind": "format",
                "value": {"type": "string_match", "expected_markers": ["[1]"], "patterns": [r"\[\d+\]"]},
            },
            "text [1]",
            "text [1] [2]",
        ),
    ],
)
def test_positive_wrong_empty_and_final_region(target, positive, negative):
    assert cv.score(positive, target) == 1
    assert cv.score(negative, target) == 0
    assert cv.score("", target) == 0
    assert cv.score(negative + "</think>" + positive, target) == 1
    assert cv.score(positive + "</think>" + negative, target) == 0


@pytest.mark.parametrize("value", ['{"a": 1, "a": 2}', "NaN", "Infinity", "-Infinity"])
def test_json_does_not_accept_ambiguous_or_nonfinite(value):
    with pytest.raises(ValueError):
        cv.strict_json(value)


def test_invalid_target_is_not_a_wrong_answer():
    for target in (
        {},
        {"kind": "mcqa", "value": "C", "choices": ["A", "B"]},
        {"kind": "format", "value": {"type": "regex", "verify_regex": []}},
    ):
        with pytest.raises(ValueError):
            cv.score("", target)


def test_countdown_requires_exact_multiset_and_safe_arithmetic():
    assert cv.countdown_value("(3+5)*3", [3, 5, 3]) == 24
    assert cv.countdown_value("3*8", [3, 5, 3]) is None
    assert cv.countdown_value("3**8", [3, 8]) is None
    assert cv.countdown_value("__import__('os').system('true')", [1]) is None
    assert cv.countdown_value("3/(5-5)", [3, 5, 5]) is None


def test_schema_local_reference_and_instance_errors():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    verifier = json_schema.JSONSchemaVerifier()
    assert asyncio.run(verifier.async_call([], '{"x": 1}', schema)).score == 1
    assert asyncio.run(verifier.async_call([], '{"x": "1"}', schema)).score == 0
    with pytest.raises(ValueError, match="local schema"):
        asyncio.run(verifier.async_call([], "{}", {"$ref": "https://example.invalid/schema"}))


def catalog(tmp_path):
    prompts, registry = tmp_path / "p.jsonl", tmp_path / "r.json"
    prompts.write_text('{"input":"question"}\n')
    registry.write_text('{"math":{"factory":"open_instruct.ground_truth_utils.MathVerifier"}}')
    entry = {"status": "qualified"}
    for key, path in (("prompt_data", prompts), ("reward_config", registry)):
        entry[key] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"schema_version": 1, "sources": {"deepmath": entry}}))
    return path, prompts


def test_catalog_uses_existing_mixture_contract_and_pins_bytes(tmp_path):
    path, prompts = catalog(tmp_path)
    data = source_catalog.resolve(path, {"deepmath": 1.0})
    assert data["sources"][0]["prompt_data"] == str(prompts)
    with pytest.raises(InputError, match="sum to 1"):
        source_catalog.resolve(path, {"deepmath": 0.5})
    with pytest.raises(InputError, match="not qualified"):
        source_catalog.resolve(path, {"unknown": 1.0})
    prompts.write_text("changed")
    with pytest.raises(InputError, match="hash mismatch"):
        source_catalog.resolve(path, {"deepmath": 1.0})


def test_catalog_rejects_path_escape(tmp_path):
    path, _ = catalog(tmp_path)
    value = json.loads(path.read_text())
    value["sources"]["deepmath"]["prompt_data"]["path"] = "../outside.jsonl"
    path.write_text(json.dumps(value))
    with pytest.raises(InputError, match="stay within"):
        source_catalog.resolve(path, {"deepmath": 1.0})
