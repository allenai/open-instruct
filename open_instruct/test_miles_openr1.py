"""OpenR1 source conversion, reference isolation and reproducible manifest adoption."""

import json

import pytest

from open_instruct.miles.datasets import openr1, run_data
from open_instruct.test_miles_run_data import Tokenizer

REVISION = "a" * 40


def raw(index=0):
    return {
        "uuid": f"uuid-{index}",
        "problem": f"Compute {index}+1.",
        "answer": str(index + 1),
        "question_type": "math-word-problem",
        "source": "test",
        "problem_type": "Algebra",
        "generations": ["incorrect", "<think>Set up the arithmetic.\n\nAdd one.\n\n</think>Answer: \\boxed{1}"],
        "correctness_math_verify": [False, True],
        "is_reasoning_complete": [True, True],
        "finish_reasons": ["stop", "stop"],
    }


def test_select_verified_complete_reference_without_leaking_into_prompt():
    row, reason = openr1.convert(raw(), REVISION, "default")
    assert reason is None
    assert row["metadata"]["bread"]["reference_index"] == 1
    assert "Set up" not in row["messages"][0]["content"]
    assert row["messages"][0]["role"] == "user"
    assert row["ground_truth"] == "1"
    assert "<think>" not in row["metadata"]["bread"]["sections"][0]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"answer": ""}, "missing_fields"),
        ({"answer": "proof"}, "proof"),
        ({"problem": "See the figure."}, "visual"),
        (
            {"correctness_math_verify": [False, False], "correctness_llama": [True, True]},
            "no_usable_verified_reference",
        ),
        ({"finish_reasons": ["stop", "length"]}, "no_usable_verified_reference"),
        ({"is_reasoning_complete": [True, False]}, "no_usable_verified_reference"),
    ],
)
def test_filters_do_not_treat_judge_approval_or_truncation_as_verified_reference(change, reason):
    assert openr1.convert(raw() | change, REVISION, "default") == (None, reason)


def test_manifest_is_deterministic_disjoint_and_adoptable(tmp_path, monkeypatch):
    rows = [raw(i) for i in range(30)] + [raw(0) | {"uuid": "duplicate-statement"}]
    first = openr1.prepare(rows, tmp_path / "first", revision=REVISION, train_count=8, eval_count=4)
    second = openr1.prepare(rows, tmp_path / "second", revision=REVISION, train_count=8, eval_count=4)
    assert first.read_bytes() == second.read_bytes()
    larger_eval = openr1.prepare(rows, tmp_path / "larger", revision=REVISION, train_count=8, eval_count=7)
    assert (first.parent / "train.jsonl").read_bytes() == (larger_eval.parent / "train.jsonl").read_bytes()
    manifest = json.loads(first.read_text())
    assert manifest["statistics"]["duplicate"] == 1
    tokenizer = Tokenizer()
    monkeypatch.setattr(run_data, "_tokenizer", lambda _: tokenizer)
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "tokenizer_config.json").write_text('{"chat_template":"HF template"}')
    partitions, _, _ = run_data._adopt({"rl_manifest": str(first)}, tokenizer, "HF template", {})
    train, evaluation = partitions["train"], partitions["eval"]
    assert len(train) == 8 and len(evaluation) == 4
    assert all("bread" not in row["metadata"] for row in evaluation)
    assert all("Set up" not in row["input"] for row in train)
    assert all("bread" in row["metadata"] for row in train)
    assert {row["metadata"]["source_uuid"] for row in train}.isdisjoint(
        row["metadata"]["source_uuid"] for row in evaluation
    )


def test_preparation_rejects_mutable_revision_and_insufficient_data(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        openr1.prepare([], tmp_path / "data", revision="main")
    with pytest.raises(ValueError, match="eligible"):
        openr1.prepare([raw()], tmp_path / "data", revision=REVISION)
    assert not (tmp_path / "data").exists()
