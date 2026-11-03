"""Offline checks for concrete Olmo 3.5 release tokenizers."""

import hashlib
import json
from importlib import util
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
SPEC = util.spec_from_file_location("prepare_olmo35_release", ROOT / "scripts/tokenizers/prepare_olmo35_release.py")
assert SPEC is not None and SPEC.loader is not None
release = util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
CUTOFF = "January 2025"  # Synthetic test date, not a proposed model cutoff.


@pytest.fixture
def source(tmp_path):
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
    template = (ROOT / "open_instruct/test_data/olmo35_chat_template.jinja").read_text()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>", chat_template=template
    )
    directory = tmp_path / "source"
    tokenizer.save_pretrained(directory)
    # Exercise the transformers GPT2 fallback that must retain Dolma's Split regex.
    config_file = directory / "tokenizer_config.json"
    config = json.loads(config_file.read_text())
    config["tokenizer_class"] = "GPT2Tokenizer"
    config_file.write_text(json.dumps(config))
    return directory


def test_prepare_roundtrips_backend_template_and_explicit_system(source, tmp_path):
    original_files = {path.name: path.read_bytes() for path in source.iterdir()}
    original = release.tokenizer_utils.load_tokenizer(source)
    output = tmp_path / "prepared"
    receipt = release.prepare_tokenizer(str(source), "local-test", CUTOFF, output)
    restored = AutoTokenizer.from_pretrained(output)
    assert restored.chat_template == original.chat_template.replace(release.PLACEHOLDER, CUTOFF)
    assert restored.backend_tokenizer.to_str() == original.backend_tokenizer.to_str()
    assert (output / "chat_template.jinja").read_text() == restored.chat_template
    assert json.loads((output / "tokenizer_config.json").read_text())["chat_template"] == restored.chat_template
    assert json.loads((output / "release_tokenizer.json").read_text()) == receipt
    assert receipt["source_template_sha256"] == release.template_hash(original.chat_template)
    assert receipt["template_sha256"] == release.template_hash(restored.chat_template)
    assert receipt["backend_sha256"] == hashlib.sha256(original.backend_tokenizer.to_str().encode()).hexdigest()
    assert receipt["source"] == str(source.resolve())
    assert receipt["revision"] == "local-test"
    assert receipt["cutoff_date"] == CUTOFF
    messages = [{"role": "user", "content": "Hello"}]
    rendered = restored.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert f"Your knowledge cutoff is {CUTOFF}" in rendered
    assert release.PLACEHOLDER not in rendered
    messages.insert(0, {"role": "system", "content": "Reply briefly."})
    assert restored.apply_chat_template(messages, tokenize=False) == original.apply_chat_template(
        messages, tokenize=False
    )
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original_files


def test_named_templates_are_synchronized(source, tmp_path):
    tokenizer = release.tokenizer_utils.load_tokenizer(source)
    template = tokenizer.chat_template
    tokenizer.chat_template = {"default": template, "tool_use": template}
    tokenizer.save_pretrained(source)
    output = tmp_path / "prepared"
    release.prepare_tokenizer(str(source), "local", CUTOFF, output)
    restored = AutoTokenizer.from_pretrained(output)
    expected = template.replace(release.PLACEHOLDER, CUTOFF)
    assert restored.chat_template == {"default": expected, "tool_use": expected}
    assert (output / "chat_template.jinja").read_text() == expected
    assert (output / "additional_chat_templates/tool_use.jinja").read_text() == expected
    assert json.loads((output / "tokenizer_config.json").read_text())["chat_template"] == [
        {"name": "default", "template": expected},
        {"name": "tool_use", "template": expected},
    ]


@pytest.mark.parametrize("date", ["", "[CUTOFF_DATE]", "2025-01-01", "Januar 2025", "January 0000", "January 25"])
def test_invalid_date_fails_before_loading(tmp_path, monkeypatch, date):
    monkeypatch.setattr(release.tokenizer_utils, "load_tokenizer", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="Month YYYY"):
        release.prepare_tokenizer("remote/tokenizer", "a" * 40, date, tmp_path / "out")


@pytest.mark.parametrize("revision", ["", "main", "v1", "8b971706"])
def test_remote_requires_immutable_revision(tmp_path, monkeypatch, revision):
    monkeypatch.setattr(release.tokenizer_utils, "load_tokenizer", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="commit SHA"):
        release.prepare_tokenizer("remote/tokenizer", revision, CUTOFF, tmp_path / "out")


def test_unexpected_template_does_not_publish_output(source, tmp_path):
    tokenizer = release.tokenizer_utils.load_tokenizer(source)
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(source)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="does not contain"):
        release.prepare_tokenizer(str(source), "local", CUTOFF, output)
    assert not output.exists()


@pytest.mark.parametrize("populated", [False, True])
def test_existing_output_is_untouched(tmp_path, monkeypatch, populated):
    output = tmp_path / "out"
    output.mkdir()
    if populated:
        (output / "keep").write_text("existing")
    monkeypatch.setattr(release.tokenizer_utils, "load_tokenizer", lambda *a, **kw: pytest.fail("must not load"))
    with pytest.raises(FileExistsError, match="fresh directory"):
        release.prepare_tokenizer("remote/tokenizer", "a" * 40, CUTOFF, output)
    assert sorted(path.name for path in output.iterdir()) == (["keep"] if populated else [])
    if populated:
        assert (output / "keep").read_text() == "existing"


def test_failed_reload_does_not_publish_partial_output(source, tmp_path, monkeypatch):
    load = release.tokenizer_utils.load_tokenizer
    calls = 0

    def fail_reload(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("reload failed")
        return load(*args, **kwargs)

    monkeypatch.setattr(release.tokenizer_utils, "load_tokenizer", fail_reload)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="reload failed"):
        release.prepare_tokenizer(str(source), "local", CUTOFF, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".out-*"))
