"""Hero data rendering must use canonical messages and verified source bytes."""

import json

import pytest
from scripts.miles import prepare_hero_basket

from open_instruct.miles import run_data


class Tokenizer:
    chat_template = "hero"

    def apply_chat_template(self, messages, *, chat_template, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return chat_template + json.dumps(messages)


@pytest.fixture
def manifest(tmp_path, monkeypatch):
    template = b"original"
    rows = [{"messages": [{"role": "system", "content": "instruction"}, {"role": "user", "content": "question"}]}]
    raw = (json.dumps(rows[0]) + "\n").encode()
    (tmp_path / "template").write_bytes(template)
    (tmp_path / "train.jsonl").write_bytes(raw)
    payload = {
        "miles": {"input_key": "messages", "chat_template": {"path": "template", "sha256": run_data._sha(template)}},
        "artifacts": {"train": {"path": "train.jsonl", "sha256": run_data._sha(raw), "records": 1}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(prepare_hero_basket.prepare_baseline_basket, "MANIFEST", path)
    return path, rows[0]["messages"]


def test_hero_prompt_map_preserves_canonical_messages(manifest):
    path, messages = manifest
    inputs = {}
    result = prepare_hero_basket.prompt_map(Tokenizer(), inputs)
    assert result == {"original" + json.dumps(messages): "hero" + json.dumps(messages)}
    assert set(inputs) == {str(path), str(path.parent / "template"), str(path.parent / "train.jsonl")}


@pytest.mark.parametrize("name", ("template", "train.jsonl"))
def test_hero_prompt_map_rejects_changed_sources(manifest, name):
    path, _ = manifest
    (path.parent / name).write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_hero_basket.prompt_map(Tokenizer(), {})
