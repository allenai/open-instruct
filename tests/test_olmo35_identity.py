"""Identity data generation stays blocked until the tokenizer's cutoff is filled."""

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
build_identity = partial(identity.build_identity, context_length=65536, context_source="synthetic-test config")
REVISION = "synthetic-test"


def save_tokenizer(directory, cutoff):
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
        chat_template=(ROOT / "open_instruct/test_data/olmo35_chat_template.jinja")
        .read_text()
        .replace("[CUTOFF_DATE]", cutoff),
    )
    tokenizer.save_pretrained(directory)
    return str(directory)


@pytest.fixture
def prepared(tmp_path):
    # Synthetic test value; this does not choose a release cutoff.
    return save_tokenizer(tmp_path / "tokenizer", "January 2025")


def test_builds_balanced_variants_with_actual_default_system(prepared, tmp_path):
    output = tmp_path / "identity"
    manifest = build_identity(prepared, REVISION, output)
    rows = [json.loads(line) for line in (output / "identity.jsonl").read_text().splitlines()]
    assert len(rows) == len({row["id"] for row in rows}) == 116
    assert set(Counter(row["parent_id"] for row in rows).values()) == {2}
    seeds = {row["id"]: row for row in json.loads(identity.SEEDS.read_text())}
    tokenizer = identity.release_tokenizer.tokenizer_utils.load_tokenizer(prepared)
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
    assert manifest["tokenizer_receipt"]["cutoff_date"] == "January 2025"
    assert manifest["tokenizer_receipt"]["revision"] == REVISION
    assert manifest["data_sha256"] == hashlib.sha256((output / "identity.jsonl").read_bytes()).hexdigest()
    assert json.loads((output / "manifest.json").read_text()) == manifest


@pytest.mark.parametrize("cutoff", ["[CUTOFF_DATE]", "2025-01"])
def test_unfilled_tokenizer_keeps_output_pending(tmp_path, cutoff):
    output = tmp_path / "identity"
    with pytest.raises(ValueError, match="CUTOFF_DATE|Month YYYY"):
        build_identity(save_tokenizer(tmp_path / "tokenizer", cutoff), REVISION, output)
    assert not output.exists()


def test_existing_output_is_untouched(prepared, tmp_path):
    output = tmp_path / "identity"
    output.mkdir()
    (output / "keep").write_text("existing")
    with pytest.raises(FileExistsError, match="fresh directory"):
        build_identity(prepared, REVISION, output)
    assert (output / "keep").read_text() == "existing"


def test_context_claims_come_from_verified_build_input(prepared, tmp_path):
    output = tmp_path / "identity"
    manifest = build_identity(
        prepared, REVISION, output, context_length=262144, context_source="synthetic-256k config"
    )
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
        build_identity(prepared, REVISION, tmp_path / "identity")
