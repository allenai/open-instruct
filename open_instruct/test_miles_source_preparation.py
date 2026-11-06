"""Source preparation must preserve populations, messages and reward contracts."""

import asyncio
import copy
import json

import pytest
import regex
from tokenizers import Tokenizer as FastTokenizer
from tokenizers import models
from transformers import PreTrainedTokenizerFast

from open_instruct import dataset_transformation
from open_instruct.miles.datasets import source_preparation as sources
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
    assert name == "deterministic" and target["value"] == raw["verifier"]
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
    records: list[dict] = [
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
                "adapter": "deterministic_v1",
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


def prepare_one(tmp_path, adapter, records, tokenizer=None):
    raw = tmp_path / "input.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    receipt = tmp_path / "hygiene.json"
    receipt.write_text('{"completed":true}')
    source = {
        "adapter": adapter,
        "inputs": [{"path": str(raw), "sha256": sources.file_hash(raw), "revision": "fixed", "split": "train"}],
        "hygiene_receipt": {"path": str(receipt), "sha256": sources.file_hash(receipt)},
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "sources": {"source": source}}))
    entry = sources.prepare(plan, tmp_path / "out", tokenizer or Tokenizer())["sources"]["source"]
    lines = (tmp_path / "out/source/dispositions.jsonl").read_text().splitlines()
    return entry, [json.loads(line)["reason"] for line in lines]


def test_malformed_rows_are_omitted_not_fatal(tmp_path):
    question = {"prompt": "question"}
    records = [
        {"prompt": ["not a message object"], "verifier": {"type": "string_match", "expected_markers": ["yes"]}},
        {"prompt": [{"role": "user", "content": ["not a content part"]}], "exp_cal_state": {}},
        {**question, "verifier": {"type": "regex", "verify_regex": ["("]}},
        {**question, "agent_ref": "math_with_judge"},
        {**question, "agent_ref": {"name": "math_with_judge_agent"}},
        {**question, "verifier": {"type": "string_match", "expected_markers": ["yes"]}},
    ]
    entry, reasons = prepare_one(tmp_path, "nemo_v1", records)
    counts = entry["counts"]
    assert (counts["input_rows"], counts["omitted_rows"], counts["prepared_rows"]) == (6, 5, 1)
    assert [reason.split(":")[0] for reason in reasons] == ["ValueError"] * 5
    assert "Invalid formatting regex" in reasons[2]
    assert reasons[3] == reasons[4] == "ValueError: math_or_judge_requires_explicit_adapter"


@pytest.mark.parametrize(
    "adapter,record,reason",
    [
        ("litmus_v02_v1", {"answer_type": "float", "expected_answer": 1, "output_regex": "("}, "Invalid Litmus"),
        ("reasoning_gym_v1", {"entry": "not an entry"}, "malformed_reasoning_gym_entry"),
        ("reasoning_gym_v1", {"metadata": "not metadata", "answer": "1"}, "malformed_reasoning_gym_entry"),
    ],
)
def test_malformed_targets_are_omitted_not_fatal(tmp_path, adapter, record, reason):
    entry, reasons = prepare_one(tmp_path, adapter, [{"prompt": "question", **record}])
    assert entry["counts"] == {"input_rows": 1, "omitted_rows": 1}
    assert reasons[0].startswith("ValueError: " + reason)


@pytest.mark.parametrize("adapter", ["mcqa_v1", "nemo_v1"])
@pytest.mark.parametrize("bad_options", [["A", "B"], [{"A": "one"}, None], {"A": "one", "B": "two"}])
def test_malformed_mcqa_options_do_not_stop_preparation(tmp_path, adapter, bad_options):
    question = {"prompt": "Choose one.\nA. one\nB. two", "expected_answer": "B"}
    records = [
        {"id": "malformed", **question, "options": bad_options},
        {"id": "valid", **question, "options": [{"A": "one"}, {"B": "two"}]},
    ]
    entry, reasons = prepare_one(tmp_path, adapter, records)
    counts = entry["counts"]
    assert (counts["input_rows"], counts["omitted_rows"], counts["prepared_rows"]) == (2, 1, 1)
    assert reasons == ["ValueError: MCQA options must be a list of dictionaries"]
    row = json.loads((tmp_path / "out/source/prompts.jsonl").read_text())
    assert row["metadata"]["sample_id"] == "valid"
    assert row["label"] == {"kind": "mcqa_final_choice", "value": "B", "choices": ["A", "B"]}


def test_invalid_structured_schema_is_omitted_not_fatal(tmp_path):
    pytest.importorskip("xmltodict")
    pytest.importorskip("openapi_schema_validator")
    records = [
        {"prompt": "question", "schema_type": "json", "schema_str": json.dumps({"type": "unknown"})},
        {"prompt": "question", "schema_type": "json", "schema_str": json.dumps({"type": "object"})},
    ]
    entry, reasons = prepare_one(tmp_path, "nemo_v1", records)
    assert entry["counts"]["prepared_rows"] == 1 and entry["counts"]["omitted_rows"] == 1
    assert reasons[0].startswith("SchemaError: ")


def test_responses_instructions_are_preserved():
    params = {"instructions": "Answer in French.", "input": "question"}
    assert sources.messages({"responses_create_params": params}) == [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "question"},
    ]


@pytest.mark.parametrize("instructions", [None, ""])
def test_empty_instructions_do_not_add_a_system_message(instructions):
    assert sources.messages({"responses_create_params": {"instructions": instructions, "input": "question"}}) == [
        {"role": "user", "content": "question"}
    ]


@pytest.mark.parametrize("instructions", [False, 123, ["Answer in French."], {"text": "Answer in French."}])
def test_instructions_must_be_text(instructions):
    with pytest.raises(ValueError, match="requires_text_instructions"):
        sources.messages({"responses_create_params": {"instructions": instructions, "input": "question"}})


@pytest.mark.parametrize("instructions", [None, "Answer in French."])
def test_leading_context_survives_preparation_and_olmo_template(tmp_path, instructions):
    raw = {
        "responses_create_params": {
            "instructions": instructions,
            "input": [
                {"role": "system", "content": "Keep units."},
                {"role": "developer", "content": [{"type": "input_text", "text": "Use two decimals."}]},
                {"role": "system", "content": " Preserve this whitespace.\n"},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Final question"},
            ],
        },
        "target": {"kind": "numeric", "value": "2"},
    }
    original = copy.deepcopy(raw)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=FastTokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        eos_token="<|im_end|>",
        chat_template=dataset_transformation.CHAT_TEMPLATES["olmo_thinker"],
    )
    entry, reasons = prepare_one(tmp_path, "deterministic_v1", [raw], tokenizer)
    assert entry["counts"]["prepared_rows"] == 1 and reasons == []
    row = json.loads((tmp_path / "out/source/prompts.jsonl").read_text())
    context = "Keep units.\n\nUse two decimals.\n\n Preserve this whitespace.\n"
    if instructions:
        context = instructions + "\n\n" + context
    assert row["metadata"]["messages"] == [
        {"role": "system", "content": context},
        *original["responses_create_params"]["input"][3:],
    ]
    assert row["input"].startswith("<|im_start|>system\n" + context)
    assert row["input"].count("<|im_start|>system\n") == 1
    assert row["input"].endswith(
        "<|im_start|>user\nFirst question<|im_end|>\n"
        "<|im_start|>assistant\nPrevious answer<|im_end|>\n"
        "<|im_start|>user\nFinal question<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )
    assert row["label"] == original["target"]
    assert row["metadata"]["raw_record_sha256"] == sources.digest(original)
    assert sources.messages(raw) == row["metadata"]["messages"]
    assert raw == original


def test_developer_message_after_user_is_not_reordered():
    raw = {
        "prompt": [
            {"role": "user", "content": "First question"},
            {"role": "developer", "content": "Later instruction"},
            {"role": "user", "content": "Next question"},
        ]
    }
    with pytest.raises(ValueError, match="requires_leading_developer_messages"):
        sources.messages(raw)


def test_unknown_adapter_is_a_plan_error_not_an_omission(tmp_path):
    with pytest.raises(sources.PlanError, match="unknown_adapter:deterministc_v1"):
        prepare_one(tmp_path, "deterministc_v1", [{"prompt": "question", "target": {"kind": "numeric", "value": "2"}}])


@pytest.mark.parametrize("expected", [None, "", "  ", ["a"], {"a": 1}])
def test_litmus_requires_nonempty_scalar_reference(expected):
    with pytest.raises(ValueError, match="expected_answer"):
        litmus.validate_target({"answer_type": "string", "expected_answer": expected, "output_regex": "Answer: (.*)"})


def test_litmus_invalid_regex_is_a_contract_error():
    with pytest.raises(ValueError, match="Invalid Litmus output_regex"):
        litmus.validate_target({"answer_type": "float", "expected_answer": 1, "output_regex": "("})


def test_litmus_timeout_obeys_policy_and_reports_diagnostics(monkeypatch):
    class SlowPattern:
        def findall(self, text, timeout):
            raise TimeoutError("regex timed out")

    target = {"answer_type": "float", "expected_answer": 1, "output_regex": "Answer: (.+)"}
    monkeypatch.setattr(litmus, "validate_target", lambda label: (SlowPattern(), 1.0, "isclose", {}))
    monkeypatch.delenv("OI_MILES_MATH_TIMEOUT_POLICY", raising=False)
    monkeypatch.delenv("OI_MILES_FORMAT_TIMEOUT_POLICY", raising=False)
    verifier = litmus.LitmusVerifier()
    result = asyncio.run(verifier.async_call([], "Answer: 1", target))
    assert result.score == 0
    assert result.diagnostics["elapsed_seconds"] >= 0
    assert {k: v for k, v in result.diagnostics.items() if k != "elapsed_seconds"} == {
        "kind": "litmus_scalar_v1",
        "status": "timeout",
        "error": "regex timed out",
        "failure_policy": "zero",
    }
    monkeypatch.setenv("OI_MILES_FORMAT_TIMEOUT_POLICY", "raise")
    with pytest.raises(TimeoutError):
        asyncio.run(verifier.async_call([], "Answer: 1", target))
    monkeypatch.setenv("OI_MILES_FORMAT_TIMEOUT_POLICY", "ignore")
    with pytest.raises(ValueError, match="must be zero or raise"):
        asyncio.run(verifier.async_call([], "Answer: 1", target))


def test_real_regex_timeout_type_is_handled():
    # The verifier relies on the regex package raising the builtin TimeoutError.
    with pytest.raises(TimeoutError):
        regex.compile(r"(?:(?:a|aa)+)+(b)").findall("a" * 5000, timeout=1e-9)
