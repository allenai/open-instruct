"""Repeated decoding must not inflate the independent question count."""

import json
import struct

import pytest
from scripts.miles import compare_opd_repeated, evaluate_opd_repeated


def panel():
    return [
        dict(dataset="math_aime_2025", question_id=f"aime:{i}", prompt=f"Q{i}", label="42", input_ids=[i])
        for i in range(2)
    ]


def test_request_seeds_are_stable_and_distinct():
    work = evaluate_opd_repeated.requests(panel(), repeats=3)
    assert work == evaluate_opd_repeated.requests(panel(), repeats=3)
    assert len(work) == 8
    assert len({row["seed"] for row in work}) == 8
    assert len([row for row in work if row["mode"] == "greedy"]) == 2
    assert "--enable-deterministic-inference" in evaluate_opd_repeated.server_command("checkpoint", "tokenizer")


def test_repeated_dapo_is_opt_in_and_preserves_existing_request_seeds():
    rows = panel() + [dict(dataset="dapo", question_id="dapo:0", prompt="D", label="7", input_ids=[3])]
    original = evaluate_opd_repeated.requests(rows, repeats=2)
    expanded = evaluate_opd_repeated.requests(rows, repeats=2, sampled_datasets=["math_aime_2025", "dapo"])
    assert len(original) == 7
    assert len(expanded) == 9
    assert [row for row in expanded if row["dataset"] != "dapo" or row["mode"] == "greedy"] == original
    with pytest.raises(ValueError, match="absent"):
        evaluate_opd_repeated.requests(rows, sampled_datasets=["missing"])


def test_legacy_checkpoint_prefix_conversion_preserves_tensor_payload(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text('{"architectures": ["Qwen3_5ForCausalLM"]}')
    tensor = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
    header = json.dumps({"model.language_model.embed_tokens.weight": tensor}).encode()
    payload = b"\x01\x02\x03\x04"
    original = struct.pack("<Q", len(header)) + header + payload
    (root / "model.safetensors").write_bytes(original)
    converted, audit = evaluate_opd_repeated.compatible_checkpoint(root, tmp_path / "converted")
    with (converted / "model.safetensors").open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        assert json.loads(stream.read(size)) == {"model.embed_tokens.weight": tensor}
        assert stream.read() == payload
    assert (root / "model.safetensors").read_bytes() == original
    assert audit["model.safetensors"]["renamed_keys"] == {
        "model.language_model.embed_tokens.weight": "model.embed_tokens.weight"
    }


def test_uncertainty_unit_is_question_and_accuracy_is_not_pass_at_k():
    left, right = {}, {}
    for row in evaluate_opd_repeated.requests(panel(), repeats=2):
        if row["mode"] == "greedy":
            continue
        key = (row["dataset"], row["question_id"], row["mode"], row["repeat"])
        left[key] = {**row, "reward": 0, "response_length": 10, "status": "completed"}
        right[key] = {**left[key], "reward": int(row["repeat"] == 0)}
    result = compare_opd_repeated.summarize(left, right)["math_aime_2025/sampled"]
    assert result["questions"] == 2
    assert result["responses_per_arm"] == 4
    assert result["right"]["pass_at_1"] == 0.5
    right.pop(next(iter(right)))
    with pytest.raises(ValueError, match="exactly the same"):
        compare_opd_repeated.summarize(left, right)


def test_multimodal_checkpoint_keeps_its_native_prefix(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "config.json").write_text('{"architectures": ["Qwen3_5ForConditionalGeneration"]}')
    header = json.dumps({"model.language_model.embed_tokens.weight": {}}).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    serving, audit = evaluate_opd_repeated.compatible_checkpoint(root, tmp_path / "converted")
    assert serving == root
    assert audit == {}


def test_resume_retains_completed_keys_and_rejects_changed_seed(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    work = evaluate_opd_repeated.requests(panel(), repeats=1)
    provenance = {
        field: "same"
        for field in (
            "arm",
            "checkpoint",
            "panel_sha256",
            "repeats",
            "head",
            "sampling_temperature",
            "response_cap",
            "request_order_sha256",
        )
    }
    (previous / "provenance.json").write_text(json.dumps(provenance))
    completed = {**work[0], "status": "completed", "response": "42", "reward": 1, "response_length": 1}
    (previous / "responses.jsonl").write_text(json.dumps(completed) + '\n{"partial":')
    remaining = evaluate_opd_repeated.resume_responses(previous, tmp_path / "resumed.jsonl", work, provenance)
    assert remaining == work[1:]
    assert provenance["resume"]["retained"] == 1
    assert provenance["resume"]["dropped_partial_last_line"]
    completed["seed"] += 1
    (previous / "responses.jsonl").write_text(json.dumps(completed) + "\n")
    with pytest.raises(ValueError, match="identity changed"):
        evaluate_opd_repeated.resume_responses(previous, tmp_path / "invalid.jsonl", work, provenance)
