"""Preserve the exact evaluation prompt tokens without leaking prior answers."""

import json

import pytest
from scripts.miles import evaluate_opd_head_pair


def test_replay_uses_only_original_prompt_tokens_and_stable_order(tmp_path):
    path = tmp_path / "capture.jsonl"
    rows = [
        dict(dataset="dapo", sample_index=2, prompt="B", tokens=[10, 11, 12, 13], response_length=2),
        dict(dataset="dapo", sample_index=1, prompt="A", tokens=[20, 21, 22], response_length=1),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    replay = evaluate_opd_head_pair.load_prompts(path)
    assert [row["prompt"] for row in replay] == ["A", "B"]
    assert [row["input_ids"] for row in replay] == [[20, 21], [10, 11]]
    rows[0]["response_length"] = 0
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="boundary"):
        evaluate_opd_head_pair.load_prompts(path)


def test_replay_refuses_duplicate_questions(tmp_path):
    path = tmp_path / "capture.jsonl"
    row = dict(dataset="dapo", sample_index=1, prompt="A", tokens=[20, 21], response_length=1)
    path.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_opd_head_pair.load_prompts(path)
