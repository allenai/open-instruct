import asyncio
import json
from types import SimpleNamespace

import pytest
from scripts.miles import prepare_colleague_exercises as preparation


class Tokenizer:
    chat_template = "model-specific-template"

    def encode(self, text, **kwargs):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["chat_template"] == self.chat_template
        return "user: " + messages[-1]["content"] + " assistant:"


def row(text, identity):
    return {
        "messages": [{"role": "user", "content": text}],
        "target": "42",
        "metadata": {
            "prepared_sample_id": identity,
            "prompt_token_ids": [999],
            "verifiers": [{"name": "math", "target": "42"}],
        },
    }


def test_selection_rejects_normalized_overlap_and_rerenders(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation.run_data, "_tokenizer", lambda _: Tokenizer())
    spec = SimpleNamespace(
        model={"source": "/model"}, data={"prompt_data": str(tmp_path / "data/train.jsonl")}, judges={}
    )
    manifest = {"miles": {"input_key": "messages", "metadata_key": "metadata", "label_key": "target"}}
    partitions = {
        "eval": [row("held out", "eval-1"), row("another held out", "eval-2")],
        "train": [row("held   out", "different-id"), row("different content", "eval-2"), row("training", "train-1")],
    }
    report = preparation.select(spec, manifest, partitions, {"math": 1}, 1)
    assert report["dropped"] == {"train:duplicate": 2}
    prepared = json.loads((tmp_path / "data/train.jsonl").read_text())
    assert prepared["input"] == "user: training assistant:"
    assert "prompt_token_ids" not in prepared["metadata"]
    assert "source_prompt_token_ids_sha256" in prepared["metadata"]
    assert partitions["train"][-1]["metadata"]["prompt_token_ids"] == [999]


def test_rejected_service_zero_does_not_pass_wrong_answer_canary(monkeypatch):
    async def execute(args, program, target, **kwargs):
        if "return a+b" in program:
            return 1, {"status": "ok"}
        return 0, {"status": "rejected", "http_status": 500}

    monkeypatch.setattr(preparation.code_rewards, "execute", execute)
    with pytest.raises(RuntimeError, match="did not establish execution semantics"):
        asyncio.run(preparation.canaries())


def test_checkpoint_inventory_requires_all_shards(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"olmo3"}')
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        preparation.inventory(tmp_path)
