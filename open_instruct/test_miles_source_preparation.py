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
from open_instruct.miles.rewards import litmus, task_verifiers


class Tokenizer:
    chat_template = "fixture template"

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


@pytest.mark.parametrize(
    "instruction,args",
    [
        ("language:response_language", {"language": "en"}),
        ("change_case:english_capital", {}),
        ("change_case:english_lowercase", {}),
    ],
)
def test_native_language_constraints_are_prepared(instruction, args):
    raw = {"prompt": "Original question", "instruction_id_list": [instruction], "kwargs": [args]}
    messages, target, name, _ = sources.adapt(raw, "instruction_v1")
    assert messages == [{"role": "user", "content": "Original question"}]
    assert name == "instruction"
    assert target["target"]["instruction_id_list"] == [instruction]
    assert target["target"]["kwargs"] == [args]


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


def test_mcqa_adapter_carries_the_row_answer_pattern(tmp_path):
    pattern = r"Selected Option\s*->\s*([A-Za-z0-9])\s*"
    record = {
        "prompt": "Choose one.\nA. one\nB. two\nEnd with: Selected Option -> X.",
        "expected_answer": "B",
        "options": [{"A": "one"}, {"B": "two"}],
        "template_metadata": {"output_regex": pattern},
    }
    prepare_one(tmp_path, "mcqa_v1", [record])
    row = json.loads((tmp_path / "out/source/prompts.jsonl").read_text())
    assert row["label"] == {"kind": "mcqa_final_choice", "value": "B", "choices": ["A", "B"], "pattern": pattern}
    assert sources.local.score("Selected Option -> B.", row["label"]) == 1


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


def test_unresolvable_nested_schema_reference_is_quarantined(tmp_path):
    pytest.importorskip("xmltodict")
    pytest.importorskip("openapi_schema_validator")
    schema = {
        "$id": "https://example.invalid/root",
        "$defs": {"num": {"type": "integer"}},
        "properties": {"child": {"$id": "child", "$ref": "#/$defs/num"}},
    }
    records = [
        {"prompt": "broken reference", "schema_type": "json", "schema_str": json.dumps(schema)},
        {"prompt": "valid row", "schema_type": "json", "schema_str": json.dumps({"type": "object"})},
    ]
    entry, reasons = prepare_one(tmp_path, "nemo_v1", records)
    assert entry["counts"]["prepared_rows"] == entry["counts"]["omitted_rows"] == 1
    assert "Unresolvable schema reference" in reasons[0]


@pytest.mark.parametrize(
    "task,answer,metadata,error",
    [
        ("number_sorting", "[1,", {}, "SyntaxError"),
        ("bitwise_arithmetic", "1", {"problem": "1 <<"}, "SyntaxError"),
        ("polynomial_multiplication", "x +", {}, "SyntaxError"),
        ("simple_integration", "x + C", {"integrand": "x +"}, "SyntaxError"),
        ("intermediate_integration", "x + C", {"integrand": "("}, "SyntaxError"),
        ("coin_flip", "1/0", {}, "ZeroDivisionError"),
        ("coin_flip", "1" + "0" * 310, {}, "OverflowError"),
    ],
)
def test_malformed_reasoning_gym_references_do_not_drop_later_rows(tmp_path, task, answer, metadata, error):
    bad = {"question": "Malformed source", "answer": answer, "metadata": {"source_dataset": task, **metadata}}
    valid = {"question": "Convert 10 to binary", "answer": "1010", "metadata": {"source_dataset": "base_conversion"}}
    original = copy.deepcopy(bad)
    records = [{"id": "bad", "raw_record": bad}, {"id": "valid", "raw_record": valid}]
    entry, reasons = prepare_one(tmp_path, "reasoning_gym_v1", records)
    assert bad == original
    assert entry["counts"]["input_rows"] == 2
    assert entry["counts"]["omitted_rows"] == entry["counts"]["prepared_rows"] == 1
    assert len(reasons) == 1 and reasons[0].startswith(
        f"ValueError: Invalid ReasoningGym reference for {task}: {error}:"
    )
    disposition = json.loads((tmp_path / "out/source/dispositions.jsonl").read_text())
    assert disposition["id"] == "bad" and disposition["status"] == "omitted"
    row = json.loads((tmp_path / "out/source/prompts.jsonl").read_text())
    assert row["metadata"]["sample_id"] == "valid" and row["label"] == valid


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


def test_litmus_timeout_scores_zero_with_diagnostics(monkeypatch):
    class SlowPattern:
        def findall(self, text, timeout):
            raise TimeoutError("regex timed out")

    target = {"answer_type": "float", "expected_answer": 1, "output_regex": "Answer: (.+)"}
    monkeypatch.setattr(litmus, "validate_target", lambda label: (SlowPattern(), 1.0, "isclose", {}))
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


def test_real_regex_timeout_type_is_handled():
    # The verifier relies on the regex package raising the builtin TimeoutError.
    with pytest.raises(TimeoutError):
        regex.compile(r"(?:(?:a|aa)+)+(b)").findall("a" * 5000, timeout=1e-9)


def dolci_keyword_row():
    return {
        "dataset": ["ifeval"],
        "ground_truth": [
            repr(
                [
                    {
                        "instruction_id": ["keywords:word_once", "punctuation:no_comma"],
                        "kwargs": [{"keyword": "apple"}, None],
                    }
                ]
            )
        ],
        "prompt": [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Include keyword apple in your response. Do not use commas."},
        ],
    }


def test_dolci_keyword_correction_is_explicit_and_preserves_prompt():
    raw = dolci_keyword_row()
    original = copy.deepcopy(raw)
    ms, unchanged, name, _ = sources.adapt(raw, "dolci_if_v1")
    corrected_ms, corrected, _, _ = sources.adapt(raw, "dolci_if_v1", {"keyword_inclusion": "at_least_once_v1"})
    assert ms == corrected_ms == raw["prompt"]
    assert raw == original
    assert name == "ifeval"
    assert unchanged["instruction_id"] == ["keywords:word_once", "punctuation:no_comma"]
    assert corrected["instruction_id"] == ["keywords:frequency", "punctuation:no_comma"]
    assert corrected["kwargs"] == [{"keyword": "apple", "frequency": 1, "relation": "at least"}, {}]
    assert unchanged["kwargs"] == [{"keyword": "apple"}, {}]
    verifier = task_verifiers.ManifestIFVerifier()
    assert asyncio.run(verifier.async_call([], "apple apple", unchanged)).score == 0.5
    assert asyncio.run(verifier.async_call([], "apple apple", corrected)).score == 1.0
    assert asyncio.run(verifier.async_call([], "pear", corrected)).score == 0.5


@pytest.mark.parametrize("prompt", ["Include keyword apple exactly once in your response.", "Answer a question."])
def test_dolci_correction_rejects_unrecognized_wording(prompt):
    raw = dolci_keyword_row()
    raw["prompt"][-1]["content"] = prompt
    with pytest.raises(ValueError, match="original inclusion instruction"):
        sources.adapt(raw, "dolci_if_v1", {"keyword_inclusion": "at_least_once_v1"})


def test_dolci_keyword_option_cannot_silently_change_nemo():
    raw = {
        "prompt": "Include keyword apple in your response.",
        "instruction_id_list": ["keywords:word_once"],
        "kwargs": [{"keyword": "apple"}],
    }
    _, target, _, _ = sources.adapt(raw, "instruction_v1")
    assert target["target"]["instruction_id_list"] == ["keywords:word_once"]
    for kind in ["instruction_v1", "nemo_v1"]:
        with pytest.raises(sources.PlanError, match="requires the dolci_if_v1"):
            sources.adapt(raw, kind, {"keyword_inclusion": "at_least_once_v1"})
    with pytest.raises(sources.PlanError, match="Unknown keyword_inclusion"):
        sources.adapt(dolci_keyword_row(), "dolci_if_v1", {"keyword_inclusion": "typo"})


def test_dolci_prepared_correction_receipts(tmp_path):
    raw = dolci_keyword_row()
    raw["prompt"] = "user: " + raw["prompt"][-1]["content"]
    unmodified = copy.deepcopy(raw)
    unmodified["ground_truth"] = [repr([{"instruction_id": ["punctuation:no_comma"], "kwargs": [None]}])]
    ambiguous = copy.deepcopy(raw)
    ambiguous["prompt"] += "\nassistant: an ambiguous extra turn"
    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        "\n".join(json.dumps({"id": str(i), "raw_record": r}) for i, r in enumerate([raw, unmodified, ambiguous]))
        + "\n"
    )
    hygiene = tmp_path / "hygiene.json"
    hygiene.write_text('{"completed":true}')
    options = {"keyword_inclusion": "at_least_once_v1"}
    plan = {
        "schema_version": 1,
        "sources": {
            "dolci-if-corrected": {
                "adapter": "dolci_if_v1",
                "options": options,
                "inputs": [
                    {
                        "path": str(input_file),
                        "sha256": sources.file_hash(input_file),
                        "revision": "pinned",
                        "split": "train",
                    }
                ],
                "hygiene_receipt": {"path": str(hygiene), "sha256": sources.file_hash(hygiene)},
            }
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    out = tmp_path / "bundle"
    entry = sources.prepare(plan_path, out, Tokenizer())["sources"]["dolci-if-corrected"]
    assert entry["options"] == options
    assert entry["counts"]["transformed_rows"] == entry["counts"]["transformed_constraints"] == 1
    assert entry["counts"]["prepared_rows"] == 2
    assert entry["counts"]["input_rows"] == 3
    assert entry["counts"]["omitted_rows"] == 1
    dispositions = [
        json.loads(line) for line in (out / "dolci-if-corrected/dispositions.jsonl").read_text().splitlines()
    ]
    assert dispositions == [
        {
            "id": "2",
            "status": "omitted",
            "reason": "ValueError: ambiguous_dolci_conversation: supply structured chat messages",
        }
    ]
    rows = [json.loads(line) for line in (out / "dolci-if-corrected/prompts.jsonl").read_text().splitlines()]
    row = rows[0]
    assert row["metadata"]["messages"] == [{"role": "user", "content": raw["prompt"][len("user: ") :]}]
    assert row["input"] == json.dumps(row["metadata"]["messages"])
    assert row["metadata"]["original_target_sha256"] == sources.digest(sources.dolci_if_target(raw))
    assert row["metadata"]["target_transformations"] == [
        {
            "name": "dolci_keyword_inclusion_v1",
            "constraint_index": 0,
            "from": "keywords:word_once",
            "to": "keywords:frequency",
            "from_kwargs": {"keyword": "apple"},
            "to_kwargs": {"keyword": "apple", "frequency": 1, "relation": "at least"},
        }
    ]
    assert rows[1]["metadata"]["target_transformations"] == []
    assert row["label"] == row["metadata"]["verifiers"][0]["target"]
    assert row["metadata"]["raw_record_sha256"] == sources.digest(raw)


@pytest.mark.parametrize("dataset_field", [{}, {"dataset": ["ifeval"]}])
def test_dolci_published_role_prefix_is_converted(dataset_field):
    raw = dolci_keyword_row()
    raw.pop("dataset")
    raw.update(dataset_field)
    raw["prompt"] = "user: Include keyword apple in your response."
    original = copy.deepcopy(raw)
    ms, label, _, _ = sources.adapt(raw, "dolci_if_v1", {"keyword_inclusion": "at_least_once_v1"})
    assert raw == original
    # Published and explicitly structured forms prepare to the same conversation
    # and target, without retaining a duplicate role prefix inside the message.
    raw["prompt"] = [{"role": "user", "content": "Include keyword apple in your response."}]
    structured_ms, structured_label, _, _ = sources.adapt(
        raw, "dolci_if_v1", {"keyword_inclusion": "at_least_once_v1"}
    )
    assert ms == raw["prompt"]
    assert (ms, label) == (structured_ms, structured_label)
    assert label["instruction_id"][0] == "keywords:frequency"


@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
def test_dolci_structured_role_like_content_is_preserved(role):
    raw = dolci_keyword_row()
    raw["prompt"][-1]["content"] = f"{role}: Include keyword apple in your response.\nassistant: quoted text"
    original = copy.deepcopy(raw)
    ms, _, _, _ = sources.adapt(raw, "dolci_if_v1", {"keyword_inclusion": "at_least_once_v1"})
    assert ms == raw["prompt"]
    assert raw == original


@pytest.mark.parametrize("field", ["prompt", "input", "responses_create_params"])
def test_dolci_conversion_uses_selected_input_and_preserves_whitespace(field):
    raw = dolci_keyword_row()
    content = " user: literal leading label\n  Include keyword apple in your response.  \n"
    raw[field] = {"input": "user: " + content} if field == "responses_create_params" else "user: " + content
    ms, _, _, _ = sources.adapt(raw, "dolci_if_v1")
    assert ms == [{"role": "user", "content": content}]
    # Other adapters do not reinterpret role-like text.
    assert sources.messages(raw) == [{"role": "user", "content": "user: " + content}]


@pytest.mark.parametrize(
    "prompt, reason",
    [
        ("user: Question\nassistant: Answer\nuser: Follow-up", "ambiguous_dolci_conversation"),
        ("user: Quoted dialogue:\nuser: example", "ambiguous_dolci_conversation"),
        ("system: Be helpful.\nuser: Question", "unsupported_dolci_role_serialization"),
        ("User: Question", "unsupported_dolci_role_serialization"),
        ("user: ", "empty_dolci_user_message"),
    ],
)
def test_dolci_ambiguous_serialization_requires_structured_messages(prompt, reason):
    raw = dolci_keyword_row()
    raw["prompt"] = prompt
    with pytest.raises(ValueError, match=reason):
        sources.adapt(raw, "dolci_if_v1")


@pytest.mark.parametrize("keyword", ["a.b", "[", " apple ", "", "(?i)apple"])
def test_dolci_frequency_remap_rejects_unaudited_keyword_shapes(keyword):
    raw = dolci_keyword_row()
    raw["ground_truth"] = [repr([{"instruction_id": ["keywords:word_once"], "kwargs": [{"keyword": keyword}]}])]
    raw["prompt"] = f"Include keyword {keyword} in your response."
    with pytest.raises(ValueError, match="audited alphabetic"):
        sources.adapt(raw, "dolci_if_v1", {"keyword_inclusion": "at_least_once_v1"})


def test_dolci_remap_preserves_existing_frequency_constraint():
    raw = dolci_keyword_row()
    raw["ground_truth"] = [
        repr(
            [
                {
                    "instruction_id": ["keywords:word_once", "keywords:frequency"],
                    "kwargs": [{"keyword": "apple"}, {"keyword": "pear", "frequency": 2, "relation": "at least"}],
                }
            ]
        )
    ]
    _, label, _, _ = sources.adapt(raw, "dolci_if_v1", {"keyword_inclusion": "at_least_once_v1"})
    assert label["instruction_id"] == ["keywords:frequency", "keywords:frequency"]
    assert label["kwargs"][1] == {"keyword": "pear", "frequency": 2, "relation": "at least"}
    verifier = task_verifiers.ManifestIFVerifier()
    assert asyncio.run(verifier.async_call([], "apple apple pear pear", label)).score == 1.0
    assert asyncio.run(verifier.async_call([], "apple apple pear", label)).score == 0.5


def test_structured_tool_declarations_preserved_and_target_declared():
    pytest.importorskip("xmltodict")
    pytest.importorskip("openapi_schema_validator")
    raw = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": "Return the result through the tool."},
                {"role": "user", "content": "Extract the count."},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "submit",
                    "description": "Submit result",
                    "parameters": '{"type":"object","properties":{"result":{"type":"object"}}}',
                }
            ],
        },
        "response_mode": "tool_call",
        "tool_name": "submit",
        "tool_payload_key": "result",
        "schema_type": "json",
        "schema_str": '{"type":"object","properties":{"n":{"type":"integer"}}}',
    }
    original = copy.deepcopy(raw)
    messages, label, name, stratum = sources.adapt(raw, "nemo_v1")
    assert messages == raw["responses_create_params"]["input"]
    assert label["tool_name"] == "submit" and label["tool_payload_key"] == "result"
    assert label["tool_call_format"] == "olmo3_pythonic_v1"
    assert name == "structured" and stratum == "structured_v1:tool_call"
    tools = sources.structured_tool_calls.tools_for_template(raw)
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert raw == original


def test_tool_response_marker_does_not_enable_unrelated_tool_tasks():
    raw = {
        "response_mode": "tool_call",
        "responses_create_params": {"input": "Question", "tools": [{"type": "function", "name": "search"}]},
    }
    with pytest.raises(ValueError, match="requires_tool_environment"):
        sources.adapt(raw, "nemo_v1")


@pytest.mark.parametrize("kind,root_type", [("csv", "object"), ("toml", "array")])
def test_incompatible_format_schema_is_an_omission(kind, root_type):
    pytest.importorskip("xmltodict")
    pytest.importorskip("openapi_schema_validator")
    raw = {
        "prompt": "Return the structured result",
        "schema_type": kind,
        "schema_str": json.dumps({"type": root_type}),
    }
    with pytest.raises(ValueError, match="unrepresentable_"):
        sources.adapt(raw, "structured_v1")


def test_legacy_source_system_prompt_is_preserved():
    raw = {"system_prompt": "detailed thinking off", "input": [{"role": "user", "content": "Question"}]}
    assert sources.messages(raw) == [
        {"role": "system", "content": "detailed thinking off"},
        {"role": "user", "content": "Question"},
    ]


def test_named_judge_preserves_reference_and_requires_explicit_route():
    raw = {"prompt": "Prove the claim", "expected_answer": "Reference proof", "agent_ref": {"name": "math_with_judge"}}
    with pytest.raises(ValueError, match="explicit_adapter"):
        sources.adapt(raw, "nemo_v1")
    route = {"field": "agent_ref.name", "routes": {"math_with_judge": {"adapter": "reference_judge_v1"}}}
    messages, target, name, _ = sources.adapt(raw, "routed_v1", route)
    assert target == "Reference proof" and name == "reference_judge"
    assert messages == [{"role": "user", "content": "Prove the claim"}]
    with pytest.raises(ValueError, match="unmapped"):
        sources.adapt(raw, "routed_v1", {**route, "routes": {}})
