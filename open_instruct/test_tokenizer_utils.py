"""Regression tests for serialized GPT-2 pre-tokenizers."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from open_instruct import tokenizer_utils


class TestSerializedTokenizer(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.backend = Tokenizer(models.BPE())
        self.backend.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(r"\p{L}+|\p{N}|[^\s\p{L}\p{N}]+|\s+"), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        self.backend.decoder = decoders.ByteLevel()
        self.backend.train_from_iterator(
            ["<think>\nOkay 12345 can't stop. café 中文\n\n"] * 30,
            trainers.BpeTrainer(
                vocab_size=320,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<eos>", "<pad>"],
                show_progress=False,
            ),
        )
        self.template = "{% for m in messages %}{{ m['content'] }}{% endfor %}{{ eos_token }}"
        original = PreTrainedTokenizerFast(
            tokenizer_object=self.backend, eos_token="<eos>", pad_token="<pad>", chat_template=self.template
        )
        original.save_pretrained(self.path)
        config_file = self.path / "tokenizer_config.json"
        config = json.loads(config_file.read_text())
        config["tokenizer_class"] = "GPT2Tokenizer"
        config_file.write_text(json.dumps(config))

    def test_restores_file_and_roundtrips_special_tokens_and_template(self):
        broken = AutoTokenizer.from_pretrained(self.path)
        saved_pre = json.loads(self.backend.to_str())["pre_tokenizer"]
        self.assertNotEqual(json.loads(broken.backend_tokenizer.to_str())["pre_tokenizer"], saved_pre)
        fixed = tokenizer_utils.load_tokenizer(str(self.path))
        for text in ["<think>\nOkay", "12345 can't stop.", "café 中文\n\n"]:
            self.assertEqual(fixed.encode(text, add_special_tokens=False), self.backend.encode(text).ids)
        self.assertEqual(fixed.eos_token_id, self.backend.token_to_id("<eos>"))
        self.assertEqual(fixed.pad_token_id, self.backend.token_to_id("<pad>"))
        self.assertEqual(fixed.chat_template, self.template)
        export = self.path / "export"
        fixed.save_pretrained(export)
        restored = AutoTokenizer.from_pretrained(export)
        self.assertEqual(restored.encode("<think>\nOkay 12345"), fixed.encode("<think>\nOkay 12345"))
        self.assertEqual(restored.chat_template, self.template)

    def test_bytelevel_export_is_not_reinterpreted(self):
        legacy = AutoTokenizer.from_pretrained(self.path)
        legacy.save_pretrained(self.path)
        actual = tokenizer_utils.load_tokenizer(str(self.path))
        self.assertEqual(
            json.loads(actual.backend_tokenizer.to_str())["pre_tokenizer"],
            json.loads(legacy.backend_tokenizer.to_str())["pre_tokenizer"],
        )
        self.assertEqual(actual.encode("<think>\nOkay 12345"), legacy.encode("<think>\nOkay 12345"))

    def test_missing_json_keeps_legacy_loader(self):
        expected = AutoTokenizer.from_pretrained(self.path)
        with (
            mock.patch.object(tokenizer_utils.AutoTokenizer, "from_pretrained", return_value=expected),
            mock.patch.object(tokenizer_utils.hub, "cached_file", return_value=None),
        ):
            self.assertIs(tokenizer_utils.load_tokenizer(str(self.path)), expected)

    def test_slow_loader_is_unchanged(self):
        expected = object()
        with mock.patch.object(tokenizer_utils.AutoTokenizer, "from_pretrained", return_value=expected) as load:
            self.assertIs(tokenizer_utils.load_tokenizer("repo", revision="pinned", use_fast=False), expected)
        load.assert_called_once_with("repo", revision="pinned", trust_remote_code=False, use_fast=False)

    def test_revision_is_used_for_backend_file(self):
        original = AutoTokenizer.from_pretrained(self.path)
        fixed = PreTrainedTokenizerFast.from_pretrained(self.path)
        with (
            mock.patch.object(tokenizer_utils.AutoTokenizer, "from_pretrained", return_value=original),
            mock.patch.object(tokenizer_utils.PreTrainedTokenizerFast, "from_pretrained", return_value=fixed) as load,
            mock.patch.object(
                tokenizer_utils.hub, "cached_file", return_value=str(self.path / "tokenizer.json")
            ) as source,
        ):
            self.assertIs(tokenizer_utils.load_tokenizer("repo", revision="pinned"), fixed)
        source.assert_called_once_with(
            "repo", "tokenizer.json", revision="pinned", _raise_exceptions_for_missing_entries=False
        )
        load.assert_called_once_with("repo", revision="pinned", trust_remote_code=False, use_fast=True)
