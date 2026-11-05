"""Source preparation must preserve populations, messages and reward contracts."""

import asyncio
import json

import pytest

from open_instruct.miles.datasets import candidate_sources as sources
from open_instruct.miles.rewards import litmus


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)

    def encode(self, text, **kwargs):
        return list(text)


def test_no_prompt_rewrite_and_native_fields():
    raw = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Original system"},
                {"role": "user", "content": "Original question"},
            ]
        },
        "verifier": {"type": "string_match", "expected_markers": ["yes"]},
    }
    ms, target, name, _ = sources.adapt(raw, "nemo_v1")
    assert ms == raw["responses_create_params"]["input"]
    assert name == "candidate" and target["value"] == raw["verifier"]
    assert ms is not raw["responses_create_params"]["input"]
    assert sources.messages({"prompt": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}) == [
        {"role": "user", "content": "hello"}
    ]


def test_explicit_math_and_no_topic_subsetting():
    assert sources.adapt({"problem": "Compute 1+1", "answer": "2"}, "math_v1")[:3] == (
        [{"role": "user", "content": "Compute 1+1"}],
        "2",
        "math",
    )
    with pytest.raises(ValueError, match="proof_requires"):
        sources.adapt({"question": "Prove the theorem", "answer": "a proof"}, "math_v1")
    with pytest.raises(ValueError, match="requires_tool"):
        sources.adapt({"responses_create_params": {"input": "hello", "tools": ["search"]}}, "nemo_v1")


def test_full_population_no_cap_or_screen_reservation(tmp_path):
    records = [
        {"id": "screened", "raw_record": {"prompt": "x" * 3000, "target": {"kind": "numeric", "value": "2"}}},
        {"id": "duplicate-text", "raw_record": {"prompt": "x" * 3000, "target": {"kind": "numeric", "value": "3"}}},
        {"id": "unsupported", "raw_record": {"prompt": "question", "target": {"kind": "unknown"}}},
    ]
    raw = tmp_path / "input.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    receipt = tmp_path / "hygiene.json"
    receipt.write_text('{"completed":true}')
    spec = {
        "schema_version": 1,
        "sources": {
            "full": {
                "adapter": "candidate_v1",
                "inputs": [
                    {"path": str(raw), "sha256": sources.file_hash(raw), "revision": "fixed", "split": "train"}
                ],
                "hygiene_receipt": {"path": str(receipt), "sha256": sources.file_hash(receipt)},
            }
        },
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(spec))
    catalog = sources.prepare(plan, tmp_path / "out", Tokenizer())
    entry = catalog["sources"]["full"]
    assert entry["status"] == "prepared"
    assert entry["counts"]["prepared_rows"] == 2
    assert entry["counts"]["input_rows"] == 3 and entry["counts"]["omitted_rows"] == 1
    assert entry["counts"]["max_prompt_tokens"] > 2048
    assert entry["repeated_prompt_rows"] == 1
    rows = [json.loads(line) for line in (tmp_path / "out/full/prompts.jsonl").read_text().splitlines()]
    assert [row["metadata"]["sample_id"] for row in rows] == ["screened", "duplicate-text"]
    assert rows[0]["metadata"]["verifiers"][0]["target"] == records[0]["raw_record"]["target"]
    with pytest.raises(FileExistsError):
        sources.prepare(plan, tmp_path / "out", Tokenizer())
    raw.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        sources.prepare(plan, tmp_path / "other", Tokenizer())


@pytest.mark.parametrize(
    "kind,expected,gold,wrong",
    [
        ("float", 0.871091, "0.871091", "0.971091"),
        ("bool", True, "true", "2"),
        ("string", "carbon dioxide", "Carbon   dioxide", "carbon monoxide"),
    ],
)
def test_litmus_original_answer_format(kind, expected, gold, wrong):
    raw = {
        "prompt": "Preserve this answer format",
        "answer_type": kind,
        "expected_answer": expected,
        "output_regex": r"Answer: (.+)",
    }
    ms, target, name, _ = sources.adapt(raw, "litmus_v02_v1")
    assert ms[0]["content"] == raw["prompt"] and name == "litmus"
    verifier = litmus.LitmusVerifier()
    assert asyncio.run(verifier.async_call([], "Answer: " + gold, target)).score == 1
    assert asyncio.run(verifier.async_call([], "Answer: " + wrong, target)).score == 0
    assert asyncio.run(verifier.async_call([], "", target)).score == 0
    assert asyncio.run(verifier.async_call([], "Answer: " + gold + " invalid", target)).score == 0


def test_litmus_invalid_contract_is_not_zero():
    with pytest.raises(ValueError, match="exactly one capture"):
        litmus.validate_target({"answer_type": "float", "expected_answer": 1, "output_regex": "none"})
    with pytest.raises(ValueError, match="disagree"):
        litmus.validate_target(
            {"answer_type": "string", "expected_answer": "a", "output_regex": "(.*)", "match": {"rule": "exact"}}
        )
