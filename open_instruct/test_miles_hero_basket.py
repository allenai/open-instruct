"""Hero data rendering must use canonical messages and verified source bytes."""

import json

import pytest
from scripts.miles import prepare_hero_basket

from open_instruct.miles.datasets import run_data


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


@pytest.mark.parametrize("omitted", (False, True))
@pytest.mark.parametrize("changed", (False, True))
def test_missing_manifest_template_uses_verified_original_policy(manifest, monkeypatch, omitted, changed):
    path, messages = manifest
    payload = json.loads(path.read_text())
    if omitted:
        payload["miles"].pop("chat_template")
    else:
        payload["miles"]["chat_template"] = None
    path.write_text(json.dumps(payload))
    original = Tokenizer()
    original.chat_template = "changed" if changed else "original"
    source = path.parent / "original-policy"
    provenance = path.parent / "preparation.json"
    provenance.write_text(
        json.dumps({"model": {"source": str(source)}, "template_sha256": run_data._sha(b"original")})
    )
    monkeypatch.setattr(prepare_hero_basket, "BASELINE_PROVENANCE", provenance)

    def load_tokenizer(actual):
        assert actual == source
        return original

    monkeypatch.setattr(run_data, "_tokenizer", load_tokenizer)
    if changed:
        with pytest.raises(ValueError, match="template hash mismatch"):
            prepare_hero_basket.prompt_map(Tokenizer(), {})
    else:
        result = prepare_hero_basket.prompt_map(Tokenizer(), {})
        assert result == {"original" + json.dumps(messages): "hero" + json.dumps(messages)}
