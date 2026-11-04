"""Offline checks for pinned Olmo 3.5 release tokenizers."""

import hashlib
import json
from importlib import util
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
SPEC = util.spec_from_file_location("check_olmo35_release", ROOT / "scripts/tokenizers/check_olmo35_release.py")
assert SPEC is not None and SPEC.loader is not None
release = util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
CUTOFF = "January 2025"  # Synthetic test date, not a proposed model cutoff.
UNFILLED = (ROOT / "open_instruct/test_data/olmo35_chat_template.jinja").read_text()


def save_tokenizer(directory: Path, chat_template) -> Path:
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\p{L}+|\p{N}|[^\s\p{L}\p{N}]+|\s+"), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Hello Reply briefly. You are Olmo January 2025 <think></think>"],
        trainers.BpeTrainer(
            vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<eos>", "<pad>", "<|im_start|>", "<|im_end|>"],
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>", chat_template=chat_template
    )
    tokenizer.save_pretrained(directory)
    # Exercise the transformers GPT2 fallback that must retain Dolma's Split regex.
    config_file = directory / "tokenizer_config.json"
    config = json.loads(config_file.read_text())
    config["tokenizer_class"] = "GPT2Tokenizer"
    config_file.write_text(json.dumps(config))
    return directory


@pytest.fixture
def filled(tmp_path):
    return save_tokenizer(tmp_path / "filled", UNFILLED.replace(release.PLACEHOLDER, CUTOFF))


def test_filled_tokenizer_passes_without_changing_files(filled, tmp_path):
    original_files = {path.name: path.read_bytes() for path in filled.iterdir()}
    output = tmp_path / "receipt.json"
    receipt = release.check_tokenizer(str(filled), "local-test", output)
    loaded = release.tokenizer_utils.load_tokenizer(filled)
    assert receipt["cutoff_date"] == CUTOFF
    assert receipt["source"] == str(filled.resolve())
    assert receipt["revision"] == "local-test"
    assert receipt["default_system"].startswith("You are Olmo")
    assert f"knowledge cutoff is {CUTOFF}," in receipt["default_system"]
    assert receipt["template_sha256"] == release.template_hash(loaded.chat_template)
    assert receipt["backend_sha256"] == hashlib.sha256(loaded.backend_tokenizer.to_str().encode()).hexdigest()
    assert json.loads(output.read_text()) == receipt
    assert {path.name: path.read_bytes() for path in filled.iterdir()} == original_files


def test_unfilled_template_is_refused(tmp_path):
    source = save_tokenizer(tmp_path / "unfilled", UNFILLED)
    output = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match=r"still contains \[CUTOFF_DATE\]"):
        release.check_tokenizer(str(source), "local", output)
    assert not output.exists()


def test_any_unfilled_named_template_is_refused(tmp_path):
    filled = UNFILLED.replace(release.PLACEHOLDER, CUTOFF)
    source = save_tokenizer(tmp_path / "named", {"default": filled, "tool_use": UNFILLED})
    with pytest.raises(ValueError, match="tool_use still contains"):
        release.check_tokenizer(str(source), "local")


@pytest.mark.parametrize("cutoff", ["2025-01", "Jan 2025", "January 0000", "January 25", ""])
def test_cutoff_must_be_month_and_year(tmp_path, cutoff):
    source = save_tokenizer(tmp_path / "bad", UNFILLED.replace(release.PLACEHOLDER, cutoff))
    with pytest.raises(ValueError, match="Month YYYY"):
        release.check_tokenizer(str(source), "local")


@pytest.mark.parametrize("revision", ["", "main", "v1", "8b971706", "A" * 40])
def test_remote_requires_immutable_revision(monkeypatch, revision):
    monkeypatch.setattr(release.tokenizer_utils, "load_tokenizer", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="commit SHA"):
        release.check_tokenizer("remote/tokenizer", revision)


def test_existing_receipt_is_untouched(tmp_path, monkeypatch):
    output = tmp_path / "receipt.json"
    output.write_text("existing")
    monkeypatch.setattr(release.tokenizer_utils, "load_tokenizer", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(FileExistsError, match="new file"):
        release.check_tokenizer("remote/tokenizer", "a" * 40, output)
    assert output.read_text() == "existing"
