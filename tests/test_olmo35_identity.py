"""Identity data generation stays blocked until a concrete tokenizer is supplied."""

import hashlib
import json
from collections import Counter
from functools import partial
from importlib import util
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative_path):
    spec = util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


identity = load_script("build_olmo35_identity", "scripts/data/sft/build_olmo35_identity.py")
release = load_script("prepare_olmo35_release", "scripts/tokenizers/prepare_olmo35_release.py")
build_identity = partial(identity.build_identity, context_length=65536, context_source="synthetic-test config")


@pytest.fixture
def prepared(tmp_path):
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel()
    backend.train_from_iterator(
        ["Hello identity knowledge cutoff Olmo Ai2"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<eos>", "<pad>", "<|im_start|>", "<|im_end|>"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
        chat_template=(ROOT / "open_instruct/test_data/olmo35_chat_template.jinja").read_text(),
    )
    source = tmp_path / "source"
    tokenizer.save_pretrained(source)
    output = tmp_path / "tokenizer"
    # Synthetic test value; this does not choose a release cutoff.
    release.prepare_tokenizer(str(source), "synthetic-test", "January 2025", output)
    return output


def test_builds_balanced_variants_with_actual_default_system(prepared, tmp_path):
    output = tmp_path / "identity"
    manifest = build_identity(prepared, output)
    rows = [json.loads(line) for line in (output / "identity.jsonl").read_text().splitlines()]
    assert len(rows) == len({row["id"] for row in rows}) == 116
    assert set(Counter(row["parent_id"] for row in rows).values()) == {2}
    seeds = {row["id"]: row for row in json.loads(identity.SEEDS.read_text())}
    tokenizer = identity.tokenizer_utils.load_tokenizer(prepared)
    for implicit, explicit in zip(rows[::2], rows[1::2]):
        assert implicit["parent_id"] == explicit["parent_id"]
        assert implicit["messages"][0] == {"role": "user", "content": seeds[implicit["parent_id"]]["user"]}
        assert explicit["messages"][1:] == implicit["messages"]
        assert [message["role"] for message in implicit["messages"]] == ["user", "assistant"]
        assert implicit["source"] == explicit["source"] == identity.SOURCE
        prompt = tokenizer.apply_chat_template(implicit["messages"][:1], tokenize=False)
        system = explicit["messages"][0]["content"]
        assert prompt.startswith(f"<|im_start|>system\n{system}<|im_end|>\n")
        assert tokenizer.apply_chat_template(implicit["messages"], tokenize=False) == tokenizer.apply_chat_template(
            explicit["messages"], tokenize=False
        )
        answer = implicit["messages"][-1]["content"]
        assert "{cutoff_date}" not in answer and "[CUTOFF_DATE]" not in answer
        assert "<think>" not in answer
        assert "<think></think>" in tokenizer.apply_chat_template(implicit["messages"], tokenize=False)
    cutoff_answer = next(row for row in rows if row["parent_id"] == "hard_coded_44")["messages"][-1]["content"]
    assert "January 2025" in cutoff_answer
    assert manifest["data_sha256"] == hashlib.sha256((output / "identity.jsonl").read_bytes()).hexdigest()
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_missing_receipt_keeps_output_pending(tmp_path):
    output = tmp_path / "identity"
    with pytest.raises(ValueError, match="cutoff remains pending"):
        build_identity(tmp_path / "unprepared", output)
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["template", "cutoff", "backend", "missing_field", "eos_config"])
def test_invalid_receipt_does_not_publish(prepared, tmp_path, tamper):
    receipt_file = prepared / "release_tokenizer.json"
    receipt = json.loads(receipt_file.read_text())
    if tamper == "template":
        (prepared / "chat_template.jinja").write_text("Changed template")
    elif tamper == "cutoff":
        receipt["cutoff_date"] = "[CUTOFF_DATE]"
    elif tamper == "backend":
        receipt["backend_sha256"] = "0" * 64
    elif tamper == "missing_field":
        receipt.pop("source_template_sha256")
    else:
        config_file = prepared / "tokenizer_config.json"
        config = json.loads(config_file.read_text())
        config["eos_token"] = "<|im_end|>"
        config_file.write_text(json.dumps(config))
    receipt_file.write_text(json.dumps(receipt))
    output = tmp_path / "identity"
    with pytest.raises(ValueError):
        build_identity(prepared, output)
    assert not output.exists()


def test_receipt_cutoff_must_match_actual_default(prepared, tmp_path):
    receipt_file = prepared / "release_tokenizer.json"
    receipt = json.loads(receipt_file.read_text())
    receipt["cutoff_date"] = "February 2025"
    receipt_file.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="prepared cutoff"):
        build_identity(prepared, tmp_path / "identity")


def test_existing_output_is_untouched(prepared, tmp_path):
    output = tmp_path / "identity"
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(FileExistsError, match="fresh directory"):
        build_identity(prepared, output)
    assert (output / "keep").read_text() == "existing"


def test_context_claims_come_from_verified_build_input(prepared, tmp_path):
    output = tmp_path / "identity"
    manifest = build_identity(prepared, output, context_length=262144, context_source="synthetic-256k config")
    rows = [json.loads(line) for line in (output / "identity.jsonl").read_text().splitlines()]
    answers = [row["messages"][-1]["content"] for row in rows]
    assert all("65,536" not in answer and "{context_length}" not in answer for answer in answers)
    assert "262,144" in next(row for row in rows if row["parent_id"] == "hard_coded_55")["messages"][-1]["content"]
    assert manifest["context_length"] == 262144
    assert manifest["context_source"] == "synthetic-256k config"


def test_changed_audited_prompt_is_rejected(prepared, tmp_path, monkeypatch):
    seeds = json.loads(identity.SEEDS.read_text())
    seeds[0]["user"] += " changed"
    seed_path = tmp_path / "seeds.json"
    seed_path.write_text(json.dumps(seeds))
    monkeypatch.setattr(identity, "SEEDS", seed_path)
    with pytest.raises(ValueError, match="differ from the 58 audited rows"):
        build_identity(prepared, tmp_path / "identity")
