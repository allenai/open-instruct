import gc
import gzip
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import datasets
import torch
from parameterized import parameterized
from transformers import AutoTokenizer

import open_instruct.dataset_transformation

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


def _get_tokenizer_path():
    src_dir = os.path.join(os.path.dirname(__file__), "test_data", "tokenizer")
    dst_dir = tempfile.mkdtemp(prefix="test_tokenizer_")
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if name.endswith(".gz"):
            dst = os.path.join(dst_dir, name[:-3])
            with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src, dst_dir)
    return dst_dir


TOKENIZER_PATH = _get_tokenizer_path()

GOLD_SFT = {"count": 100, "hash": "3e745ff9615c9b0e3d8efe74f3f96cde01ac6a720535f0b4ef7175ebb2d1d6cf"}
GOLD_PREFERENCE = {"count": 97, "hash": "415d8c34ac25cf04d798f27a88c90df38826f71a404e5345563635778bdf9bb3"}
GOLD_RLVR = {"count": 100, "hash": "9ebada598693087c4cd4804d474fbbe07f41a7dffb38104ddee4e93ba0bfd3b1"}


class TestEnvConfigNormalization(unittest.TestCase):
    def test_normalize_single_dict_env_config(self):
        row = {"env_config": {"env_name": "guess_number", "number": "7"}}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(row["env_config"], {"env_configs": [{"env_name": "guess_number", "number": "7"}]})

    def test_normalize_list_env_config(self):
        row = {"env_config": [{"env_name": "counter", "target": "3"}]}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(row["env_config"], {"env_configs": [{"env_name": "counter", "target": "3"}]})

    def test_normalize_canonical_env_config(self):
        row = {"env_config": {"max_steps": 10, "env_configs": [{"env_name": "guess_number", "number": "5"}]}}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(
            row["env_config"], {"max_steps": 10, "env_configs": [{"env_name": "guess_number", "number": "5"}]}
        )


class TestConfigHash(unittest.TestCase):
    def test_config_hash_different(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH, tokenizer_revision="main", chat_template_name="tulu"
        )

        sft_data = os.path.join(TEST_DATA_DIR, "sft_sample.jsonl")
        dcs1 = [
            open_instruct.dataset_transformation.DatasetConfig(
                dataset_name=sft_data,
                dataset_split="train",
                dataset_revision="main",
                transform_fn=["sft_tokenize_v1"],
                transform_fn_args=[{}],
            )
        ]

        dcs2 = [
            open_instruct.dataset_transformation.DatasetConfig(
                dataset_name=sft_data,
                dataset_split="train",
                dataset_revision="main",
                transform_fn=["sft_tokenize_mask_out_prompt_v1"],
                transform_fn_args=[{}],
            )
        ]
        hash1 = open_instruct.dataset_transformation.compute_config_hash(dcs1, tc)
        hash2 = open_instruct.dataset_transformation.compute_config_hash(dcs2, tc)
        self.assertNotEqual(hash1, hash2, "Different configs should have different hashes")

    def test_config_hash_stable_across_tokenizer_access(self):
        # Regression test for the numpy SFT cache mismatch: loading the tokenizer
        # populates tc.tokenizer_files_hash, so the hash must not depend on whether
        # tc.tokenizer was accessed before compute_config_hash was called.
        def make_tc():
            return open_instruct.dataset_transformation.TokenizerConfig(
                tokenizer_name_or_path=TOKENIZER_PATH, tokenizer_revision="main", chat_template_name="tulu"
            )

        dcs = [
            open_instruct.dataset_transformation.DatasetConfig(
                dataset_name=os.path.join(TEST_DATA_DIR, "sft_sample.jsonl"),
                dataset_split="train",
                dataset_revision="main",
                transform_fn=["sft_tokenize_v1"],
                transform_fn_args=[{}],
            )
        ]

        tc = make_tc()
        hash_before_access = open_instruct.dataset_transformation.compute_config_hash(dcs, tc)
        hash_after_access = open_instruct.dataset_transformation.compute_config_hash(dcs, tc)
        self.assertEqual(hash_before_access, hash_after_access)

        tc_preloaded = make_tc()
        _ = tc_preloaded.tokenizer
        hash_preloaded = open_instruct.dataset_transformation.compute_config_hash(dcs, tc_preloaded)
        self.assertEqual(hash_preloaded, hash_before_access)

    def test_dataset_commit_hash_resolved_after_download(self):
        # get_commit_hash only looks in the local HF hub cache, so it must run
        # after load_dataset has downloaded the dataset (which populates the
        # cache); otherwise a fresh machine hashes dataset_commit_hash=None while
        # a warm one hashes the real commit, producing different cache hashes.
        calls = []
        fake_dataset = datasets.Dataset.from_dict({"messages": [[{"role": "user", "content": "hi"}]]})

        def fake_load_dataset(*args, **kwargs):
            calls.append("load_dataset")
            return fake_dataset

        def fake_get_commit_hash(*args, **kwargs):
            calls.append("get_commit_hash")
            return "abc123"

        with (
            mock.patch.object(open_instruct.dataset_transformation, "load_dataset", side_effect=fake_load_dataset),
            mock.patch.object(
                open_instruct.dataset_transformation, "get_commit_hash", side_effect=fake_get_commit_hash
            ),
        ):
            dc = open_instruct.dataset_transformation.DatasetConfig(
                dataset_name="fake-org/fake-hub-dataset", dataset_split="train", dataset_revision="main"
            )

        self.assertEqual(calls, ["load_dataset", "get_commit_hash"])
        self.assertEqual(dc.dataset_commit_hash, "abc123")


class TestCachedDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.original_hf_home = os.environ.get("HF_HOME")
        self.original_hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        self.original_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")

        os.environ["HF_HOME"] = self.temp_dir.name
        os.environ["HF_DATASETS_CACHE"] = os.path.join(self.temp_dir.name, "datasets")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.temp_dir.name, "transformers")

    def tearDown(self):
        if self.original_hf_home is not None:
            os.environ["HF_HOME"] = self.original_hf_home
        else:
            os.environ.pop("HF_HOME", None)

        if self.original_hf_datasets_cache is not None:
            os.environ["HF_DATASETS_CACHE"] = self.original_hf_datasets_cache
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)

        if self.original_transformers_cache is not None:
            os.environ["TRANSFORMERS_CACHE"] = self.original_transformers_cache
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)

        self.temp_dir.cleanup()
        if os.path.exists(self.temp_dir.name):
            shutil.rmtree(self.temp_dir.name, ignore_errors=True)
        gc.collect()

    def test_get_cached_dataset_tulu_sft(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=True,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = [os.path.join(TEST_DATA_DIR, "sft_sample.jsonl"), "1.0"]
        dataset_mixer_list_splits = ["train"]
        dataset_transform_fn = ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]

        transform_fn_args = [{"max_seq_length": 4096}, {}]
        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list,
            dataset_mixer_list_splits,
            tc,
            dataset_transform_fn,
            transform_fn_args,
            open_instruct.dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        self.assertEqual(len(dataset), GOLD_SFT["count"])
        dataset_hash = hashlib.sha256()
        for row in dataset:
            dataset_hash.update(str(row["input_ids"]).encode())
        self.assertEqual(dataset_hash.hexdigest(), GOLD_SFT["hash"])

    def test_sft_tokenization_drops_tools_column_when_target_columns_none(self):
        # With target_columns=None it defaults to all dataset columns (including "tools");
        # SFT tokenization must still drop the consumed tools column.
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=True,
            chat_template_name="tulu",
            add_bos=False,
        )
        jsonl_path = os.path.join(self.temp_dir.name, "sft_with_tools.jsonl")
        with open(jsonl_path, "w") as f:
            for _ in range(3):
                f.write(
                    json.dumps(
                        {
                            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                            "tools": None,
                        }
                    )
                    + "\n"
                )

        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            [jsonl_path, "1.0"],
            ["train"],
            tc,
            ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
            [{"max_seq_length": 4096}, {}],
            target_columns=None,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        self.assertNotIn(open_instruct.dataset_transformation.TOOLS_COLUMN_KEY, dataset.column_names)

    def test_tools_column_preserved_for_transforms_that_do_not_consume_it(self):
        # sft_tokenize_v1 does not forward `tools` to the chat template, so the column must
        # survive tokenization. Dropping it there would silently discard the tool schemas
        # without ever rendering them into the prompt.
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=True,
            chat_template_name="tulu",
            add_bos=False,
        )
        jsonl_path = os.path.join(self.temp_dir.name, "sft_with_tools_preserved.jsonl")
        tools = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        with open(jsonl_path, "w") as f:
            for _ in range(3):
                f.write(
                    json.dumps(
                        {
                            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
                            "tools": tools,
                        }
                    )
                    + "\n"
                )

        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            [jsonl_path, "1.0"],
            ["train"],
            tc,
            ["sft_tokenize_v1"],
            [{}],
            target_columns=None,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        self.assertIn(open_instruct.dataset_transformation.TOOLS_COLUMN_KEY, dataset.column_names)

    def test_get_cached_dataset_tulu_preference(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=False,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = [os.path.join(TEST_DATA_DIR, "preference_sample.jsonl"), "1.0"]
        dataset_mixer_list_splits = ["train"]
        dataset_transform_fn = ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]
        transform_fn_args = [{"max_seq_length": 2048}, {}]
        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list,
            dataset_mixer_list_splits,
            tc,
            dataset_transform_fn,
            transform_fn_args,
            open_instruct.dataset_transformation.TOKENIZED_PREFERENCE_DATASET_KEYS,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        self.assertEqual(len(dataset), GOLD_PREFERENCE["count"])
        dataset_hash = hashlib.sha256()
        for row in dataset:
            dataset_hash.update(str(row["chosen_input_ids"]).encode())
        self.assertEqual(dataset_hash.hexdigest(), GOLD_PREFERENCE["hash"])

    def test_get_cached_dataset_tulu_rlvr(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=False,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = [os.path.join(TEST_DATA_DIR, "rlvr_sample.jsonl"), "1.0"]
        dataset_mixer_list_splits = ["train"]
        dataset_transform_fn = ["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"]
        transform_fn_args = [{}, {"max_prompt_token_length": 2048}]
        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list,
            dataset_mixer_list_splits,
            tc,
            dataset_transform_fn,
            transform_fn_args,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        self.assertEqual(len(dataset), GOLD_RLVR["count"])
        dataset_hash = hashlib.sha256()
        for row in dataset:
            dataset_hash.update(str(row[open_instruct.dataset_transformation.INPUT_IDS_PROMPT_KEY]).encode())
        self.assertEqual(dataset_hash.hexdigest(), GOLD_RLVR["hash"])


def _mask_non_assistant(idx, msg, _msgs):
    return msg["role"] != "assistant"


def _mask_all_but_last(idx, _msg, msgs):
    return idx < len(msgs) - 1


class TestMaskLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    def _tokenize(self, messages):
        ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=False,
            padding=False,
            truncation=False,
            add_generation_prompt=False,
        )
        assert isinstance(ids, torch.Tensor)
        return ids

    def _prefix_len(self, messages, add_generation_prompt=False):
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=False,
            add_generation_prompt=add_generation_prompt,
        ).shape[1]

    @parameterized.expand(
        [
            (
                "system_user_assistant",
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ],
                _mask_non_assistant,
            ),
            (
                "user_assistant_single_turn",
                [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}],
                _mask_non_assistant,
            ),
            (
                "multiturn",
                [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Second question"},
                    {"role": "assistant", "content": "Second answer"},
                ],
                _mask_non_assistant,
            ),
        ]
    )
    def test_has_both_masked_and_kept_tokens(self, _name, messages, should_mask):
        input_ids = self._tokenize(messages)
        labels = input_ids.clone()
        open_instruct.dataset_transformation.mask_labels(labels, messages, self.tokenizer, 4096, should_mask)
        flat = labels.flatten().tolist()
        self.assertTrue(any(x == -100 for x in flat), "Should have masked tokens")
        self.assertTrue(any(x != -100 for x in flat), "Should have kept tokens")

    @parameterized.expand(
        [
            (
                "last_turn_only",
                [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                    {"role": "user", "content": "And 3+3?"},
                    {"role": "assistant", "content": "6"},
                ],
                _mask_all_but_last,
                3,
            ),
            (
                "deferred_system_prefix",
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                _mask_non_assistant,
                2,
            ),
        ]
    )
    def test_boundary_masking(self, _name, messages, should_mask, prefix_msg_count):
        """Everything before prefix_msg_count messages is masked, and there are
        kept tokens after (the assistant response)."""
        input_ids = self._tokenize(messages)
        labels = input_ids.clone()
        open_instruct.dataset_transformation.mask_labels(labels, messages, self.tokenizer, 4096, should_mask)
        flat = labels.flatten().tolist()
        add_gen = messages[prefix_msg_count]["role"] == "assistant" if prefix_msg_count < len(messages) else False
        boundary = self._prefix_len(messages[:prefix_msg_count], add_generation_prompt=add_gen)
        self.assertTrue(all(x == -100 for x in flat[:boundary]), f"Tokens before {boundary} should be masked")
        self.assertTrue(
            any(x != -100 for x in flat[boundary:]), f"Tokens from {boundary} onward should have kept tokens"
        )


class TestToolNormalization(unittest.TestCase):
    def test_normalize_tools_accepts_json_encoded_schema_list(self):
        tool_schema = {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute bash",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        tools = open_instruct.dataset_transformation._normalize_tools_for_chat_template(json.dumps([tool_schema]))

        self.assertEqual(tools, [tool_schema])

    def test_normalize_tools_wraps_single_dict(self):
        tool_schema = {"type": "function", "function": {"name": "bash"}}
        self.assertEqual(
            open_instruct.dataset_transformation._normalize_tools_for_chat_template(tool_schema), [tool_schema]
        )

    def test_normalize_tools_treats_empty_as_none(self):
        self.assertIsNone(open_instruct.dataset_transformation._normalize_tools_for_chat_template(None))
        self.assertIsNone(open_instruct.dataset_transformation._normalize_tools_for_chat_template(""))
        self.assertIsNone(open_instruct.dataset_transformation._normalize_tools_for_chat_template([]))
        self.assertIsNone(open_instruct.dataset_transformation._normalize_tools_for_chat_template(float("nan")))

    def test_normalize_tools_treats_json_null_and_empty_string_as_none(self):
        self.assertIsNone(open_instruct.dataset_transformation._normalize_tools_for_chat_template("null"))
        self.assertIsNone(open_instruct.dataset_transformation._normalize_tools_for_chat_template('""'))
        self.assertIsNone(open_instruct.dataset_transformation._normalize_tools_for_chat_template("[]"))

    def test_normalize_tools_rejects_tool_name_lists(self):
        with self.assertRaises(TypeError):
            open_instruct.dataset_transformation._normalize_tools_for_chat_template(["bash"])

    def test_normalize_tools_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            open_instruct.dataset_transformation._normalize_tools_for_chat_template("{not json")


class TestSFTTuluTokenizeLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    def test_only_assistant_tokens_are_trainable(self):
        row = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
            ]
        }
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), self.tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()

        self.assertEqual(len(input_ids), len(labels))
        # At least one assistant token is trainable, and every trainable label matches its input id.
        self.assertTrue(any(label != -100 for label in labels))
        for input_id, label in zip(input_ids, labels):
            if label != -100:
                self.assertEqual(label, input_id)
        # The prompt prefix (first token, part of the user turn) must be masked.
        self.assertEqual(labels[0], -100)

    def test_tools_column_is_consumed_and_accepts_json_string(self):
        # The test chat template ignores tools, but tokenization must still succeed
        # when a JSON-encoded tools column is present (it is parsed, not crashed on).
        row = {
            "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            "tools": json.dumps([{"type": "function", "function": {"name": "bash"}}]),
        }
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), self.tokenizer, max_seq_length=4096
        )
        self.assertIn(open_instruct.dataset_transformation.LABELS_KEY, out)

    def test_content_span_handles_header_without_newline_and_multiline_content(self):
        # A "simple_chat"-style template: the assistant header ("Assistant: ") has no
        # trailing newline, and the content itself spans a newline. The newline heuristic
        # would mis-mask here; matching the actual content keeps the full content trainable
        # and the header masked.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}User: {{ m['content'] }}\n"
            "{% elif m['role'] == 'assistant' %}Assistant: {{ m['content'] }}{{ eos_token }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok\ndone"}]}
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        trained_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
        trained_text = tokenizer.decode(trained_ids)
        self.assertIn("ok", trained_text)
        self.assertIn("done", trained_text)
        self.assertNotIn("Assistant", trained_text)
        self.assertNotIn("User", trained_text)

    def test_template_rejecting_prefix_without_user_turn_is_masked_out(self):
        # Some templates (e.g. Qwen3.5) raise when handed a prefix containing only
        # system/tool turns. That happens here for the assistant at index 1, whose prefix
        # is [system]. The span is underivable for that conversation, so the row trains on
        # nothing (and is dropped) rather than aborting the whole dataset map. The underlying
        # derivation still names the situation rather than surfacing the template's error.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.chat_template = (
            "{% if messages | selectattr('role', 'equalto', 'user') | list | length == 0 %}"
            "{{ raise_exception('conversation must contain a user turn') }}{% endif %}"
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )
        row = {
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "bye"},
            ]
        }
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), tokenizer, max_seq_length=4096
        )
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        self.assertTrue(all(label == -100 for label in labels))
        with self.assertRaisesRegex(open_instruct.dataset_transformation.AssistantSpanDerivationError, "no user turn"):
            open_instruct.dataset_transformation._tokenize_tulu_sft_with_assistant_labels(
                row["messages"], tokenizer, None, 4096
            )

    def test_template_appending_eos_only_on_last_turn_is_masked_out(self):
        # A template that appends eos_token only on the final turn (loop.last) is not
        # prefix-stable. Derivation falls back to prefix token counts, which for this template
        # puts the boundary inside the role header, so the span is rejected and the row trains
        # on nothing rather than being silently mis-masked.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}"
            "{% if loop.last %}{{ eos_token }}{% endif %}\n{% endfor %}"
        )
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "HELLOWORLD"}]}
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), tokenizer, max_seq_length=4096
        )
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        self.assertTrue(all(label == -100 for label in labels))
        self.assertFalse(open_instruct.dataset_transformation.sft_tulu_filter_v1(out, tokenizer))

    def test_conversation_starting_with_assistant(self):
        # message_idx == 0 -> messages[:0] is empty; must not call apply_chat_template([]).
        row = {"messages": [{"role": "assistant", "content": "OPENINGLINE"}]}
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), self.tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        trained_text = self.tokenizer.decode([tid for tid, lab in zip(input_ids, labels) if lab != -100])
        self.assertIn("OPENINGLINE", trained_text)

    def test_content_offset_falls_back_to_generation_prompt(self):
        # Template uppercases assistant content so rfind(content) fails, forcing the
        # add_generation_prompt header-length fallback (no newline heuristic).
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.chat_template = (
            "{% for m in messages %}"
            "{% if m['role'] == 'user' %}User: {{ m['content'] }}\n"
            "{% elif m['role'] == 'assistant' %}Assistant: {{ m['content'] | upper }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant: {% endif %}"
        )
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        trained_text = tokenizer.decode([tid for tid, lab in zip(input_ids, labels) if lab != -100])
        self.assertIn("HELLO", trained_text)
        self.assertNotIn("Assistant", trained_text)
        self.assertNotIn("hi", trained_text)

    def test_slow_tokenizer_raises_clear_error(self):
        slow_tokenizer = mock.MagicMock()
        slow_tokenizer.is_fast = False
        type(slow_tokenizer).__name__ = "FakeSlowTokenizer"
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
                dict(row), slow_tokenizer, max_seq_length=4096
            )

    def test_without_truncation_variant_runs(self):
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello there"}]}
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_without_truncation_v1(dict(row), self.tokenizer)
        self.assertTrue(any(label != -100 for label in out[open_instruct.dataset_transformation.LABELS_KEY].tolist()))

    def test_last_turn_only_trains_final_assistant_turn(self):
        row = {
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "FIRSTANSWER"},
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "SECONDANSWER"},
            ]
        }
        out = open_instruct.dataset_transformation.last_turn_tulu_tokenize_and_truncate_v1(
            dict(row), self.tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        trained_text = self.tokenizer.decode([tid for tid, lab in zip(input_ids, labels) if lab != -100])
        # Only the final assistant turn is trained; the earlier assistant turn is masked.
        self.assertIn("SECONDANSWER", trained_text)
        self.assertNotIn("FIRSTANSWER", trained_text)

    def test_last_turn_only_when_conversation_does_not_end_with_assistant(self):
        # Trailing non-assistant message must not cause the real last assistant turn to be skipped.
        row = {
            "messages": [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "FIRSTANSWER"},
                {"role": "user", "content": "second question"},
                {"role": "assistant", "content": "SECONDANSWER"},
                {"role": "user", "content": "trailing user message"},
            ]
        }
        out = open_instruct.dataset_transformation.last_turn_tulu_tokenize_and_truncate_v1(
            dict(row), self.tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        trained_text = self.tokenizer.decode([tid for tid, lab in zip(input_ids, labels) if lab != -100])
        self.assertIn("SECONDANSWER", trained_text)
        self.assertNotIn("FIRSTANSWER", trained_text)
        self.assertNotIn("trailing user message", trained_text)


# Templates from CHAT_TEMPLATES that are used for SFT (as opposed to the RL/inference-only
# ones, which never reach the assistant-label code path).
SFT_CHAT_TEMPLATE_NAMES = [
    "tulu",
    "tulu_thinker",
    "tulu_thinker_r1_style",
    "olmo",
    "olmo_old",
    "olmo_thinker",
    "olmo_thinker_no_think_7b",
    "olmo_thinker_no_think_sft_tokenization",
    "olmo_thinker_remove_intermediate_thinking",
    "zephyr",
    "simple_chat",
]

# Whether a template renders prefix-stably depends on the tokenizer's eos_token, not just on
# the template: the olmo family emits <|im_end|> on non-final assistant turns and eos_token on
# the final one, so it happens to round-trip when eos_token *is* <|im_end|> (Qwen-style
# tokenizers) and breaks when it is not (OLMo-2, whose eos is <|endoftext|>). Both are swept.
EOS_VARIANTS = [("native_eos", None), ("im_end_eos", "<|im_end|>")]

CONVERSATION_SHAPES = {
    # Plain alternating multi-turn conversation.
    "alternating": [
        {"role": "user", "content": "USERONE"},
        {"role": "assistant", "content": "ASSISTONE"},
        {"role": "user", "content": "USERTWO"},
        {"role": "assistant", "content": "ASSISTTWO"},
    ],
    # Two assistant turns back to back. Rare (~0.005% of tulu-3-sft-olmo-2-mixture) but present,
    # and one such row aborts the whole `dataset.map`.
    "consecutive_assistant": [
        {"role": "system", "content": "SYSTEMZERO"},
        {"role": "user", "content": "USERONE"},
        {"role": "assistant", "content": "ASSISTONE"},
        {"role": "user", "content": "USERTWO"},
        {"role": "assistant", "content": "ASSISTTWO"},
        {"role": "assistant", "content": "ASSISTTHREE"},
    ],
}

# (template, eos_variant, shape) combinations whose assistant spans can be derived, and which
# must therefore produce correct labels. Everything not listed here is expected to be detected
# as underivable and masked out (the row is then dropped by `sft_tulu_filter_v1`).
#
# The two categories together are the specification: a combination either trains exactly the
# assistant content, or trains nothing at all. Producing *wrong* labels is the failure this
# sweep exists to catch, and no combination is allowed to do it.
#
# Growing this set is the goal of the P1 work in
# https://github.com/allenai/open-instruct/issues/1800: templates that rewrite assistant
# content as they render (`*_thinker*` inject or split on <think>) cannot be located by either
# the char-offset or the token-count derivation, and need `{% generation %}` markers instead.
# Consecutive assistant turns are underivable for the same underlying reason -- every
# derivation must render a prefix ending in an assistant turn, which templates that
# special-case the final turn render differently from the full conversation.
DERIVABLE_COMBINATIONS = {
    ("tulu", "native_eos", "alternating"),
    ("tulu", "im_end_eos", "alternating"),
    ("olmo", "native_eos", "alternating"),
    ("olmo", "im_end_eos", "alternating"),
    ("olmo", "im_end_eos", "consecutive_assistant"),
    ("olmo_old", "native_eos", "alternating"),
    ("olmo_old", "im_end_eos", "alternating"),
    ("olmo_old", "im_end_eos", "consecutive_assistant"),
    ("olmo_thinker_no_think_7b", "native_eos", "alternating"),
    ("olmo_thinker_no_think_7b", "im_end_eos", "alternating"),
    ("olmo_thinker_no_think_7b", "im_end_eos", "consecutive_assistant"),
    ("olmo_thinker_no_think_sft_tokenization", "native_eos", "alternating"),
    ("olmo_thinker_no_think_sft_tokenization", "im_end_eos", "alternating"),
    ("olmo_thinker_no_think_sft_tokenization", "im_end_eos", "consecutive_assistant"),
    ("zephyr", "native_eos", "alternating"),
    ("zephyr", "native_eos", "consecutive_assistant"),
    ("zephyr", "im_end_eos", "alternating"),
    ("zephyr", "im_end_eos", "consecutive_assistant"),
}


def _sweep_cases():
    for template_name in SFT_CHAT_TEMPLATE_NAMES:
        for eos_name, eos_token in EOS_VARIANTS:
            for shape_name in CONVERSATION_SHAPES:
                combo = (template_name, eos_name, shape_name)
                yield (f"{template_name}__{eos_name}__{shape_name}", template_name, eos_token, shape_name, combo)


WORKING_SWEEP_CASES = [c for c in _sweep_cases() if c[4] in DERIVABLE_COMBINATIONS]
BROKEN_SWEEP_CASES = [c for c in _sweep_cases() if c[4] not in DERIVABLE_COMBINATIONS]


class TestChatTemplateAssistantLabelSweep(unittest.TestCase):
    """Every SFT chat template must mask exactly the non-assistant text.

    The specification is template-independent: after tokenization, decoding the unmasked
    tokens must yield all of the assistant content and none of the user/system content. This
    sweep is what makes template regressions visible -- the individual tests above each pin a
    single template, so a template that silently trains on nothing (or on the prompt) passes
    them.
    """

    def _tokenizer_for(self, template_name, eos_token):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        if eos_token is not None:
            tokenizer.eos_token = eos_token
        tokenizer.chat_template = open_instruct.dataset_transformation.CHAT_TEMPLATES[template_name]
        return tokenizer

    @parameterized.expand([(c[0], c[1], c[2], c[3]) for c in WORKING_SWEEP_CASES])
    def test_assistant_content_is_exactly_what_is_trained(self, _name, template_name, eos_token, shape_name):
        messages = CONVERSATION_SHAPES[shape_name]
        tokenizer = self._tokenizer_for(template_name, eos_token)
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            {"messages": [dict(m) for m in messages]}, tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        trained_text = tokenizer.decode([tid for tid, lab in zip(input_ids, labels) if lab != -100])

        for message in messages:
            if message["role"] == "assistant":
                self.assertIn(
                    message["content"], trained_text, f"assistant content dropped from loss ({trained_text!r})"
                )
            elif message["content"]:
                self.assertNotIn(message["content"], trained_text, f"{message['role']} content leaked into loss")

    @parameterized.expand([(c[0], c[1], c[2], c[3]) for c in BROKEN_SWEEP_CASES])
    def test_underivable_spans_are_masked_out_not_mislabelled(self, _name, template_name, eos_token, shape_name):
        # Detection, not silence: the span can't be located, so the row trains on nothing and
        # `sft_tulu_filter_v1` drops it. Training on a misaligned span is the failure mode this
        # whole area guards against, so "no labels" is the correct outcome, not "some labels".
        messages = CONVERSATION_SHAPES[shape_name]
        tokenizer = self._tokenizer_for(template_name, eos_token)
        row = {"messages": [dict(m) for m in messages]}
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), tokenizer, max_seq_length=4096
        )
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        self.assertTrue(all(label == -100 for label in labels), "underivable row must train on nothing")
        self.assertFalse(
            open_instruct.dataset_transformation.sft_tulu_filter_v1(out, tokenizer),
            "an all-masked row must be dropped by the filter",
        )
        # The underlying derivation must still report *why*, so the drop is diagnosable.
        tools = None
        with self.assertRaises(open_instruct.dataset_transformation.AssistantSpanDerivationError):
            open_instruct.dataset_transformation._tokenize_tulu_sft_with_assistant_labels(
                messages, tokenizer, tools, 4096
            )

    def test_one_bad_row_does_not_abort_a_dataset_map(self):
        # The rung-5 failure mode: a single underivable conversation used to raise inside
        # `dataset.map` and kill the whole tokenization job. Good rows must survive alongside it.
        tokenizer = self._tokenizer_for("tulu", None)
        good = {"messages": [dict(m) for m in CONVERSATION_SHAPES["alternating"]]}
        bad = {"messages": [dict(m) for m in CONVERSATION_SHAPES["consecutive_assistant"]]}
        rows = [
            open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(dict(r), tokenizer, 4096)
            for r in (good, bad, good)
        ]
        kept = [r for r in rows if open_instruct.dataset_transformation.sft_tulu_filter_v1(r, tokenizer)]
        self.assertEqual(len(kept), 2, "the two derivable rows must survive the undecidable one")

    def test_prefix_unstable_template_falls_back_instead_of_raising(self):
        # `olmo` swaps <|im_end|> for eos_token on the final assistant turn, so the rendered
        # prefixes are not literal prefixes of the full render. That used to raise; it must now
        # fall back to token-count derivation and produce correct labels.
        messages = CONVERSATION_SHAPES["alternating"]
        tokenizer = self._tokenizer_for("olmo", None)
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            {"messages": [dict(m) for m in messages]}, tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        trained_text = tokenizer.decode([tid for tid, lab in zip(input_ids, labels) if lab != -100])
        self.assertIn("ASSISTONE", trained_text)
        self.assertIn("ASSISTTWO", trained_text)
        self.assertNotIn("USERONE", trained_text)
        self.assertNotIn("USERTWO", trained_text)

    def test_last_turn_only_still_trains_only_the_final_turn_after_fallback(self):
        # The fallback derives a span per assistant turn; `last_turn_only` must still restrict
        # to the final one rather than training every turn.
        messages = CONVERSATION_SHAPES["alternating"]
        tokenizer = self._tokenizer_for("olmo", None)
        out = open_instruct.dataset_transformation.last_turn_tulu_tokenize_and_truncate_v1(
            {"messages": [dict(m) for m in messages]}, tokenizer, max_seq_length=4096
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        trained_text = tokenizer.decode([tid for tid, lab in zip(input_ids, labels) if lab != -100])
        self.assertIn("ASSISTTWO", trained_text)
        self.assertNotIn("ASSISTONE", trained_text)

    def test_span_running_past_the_turn_is_rejected(self):
        # Guards the over-wide direction directly: a span that starts correctly but runs on
        # into the following turns must be rejected, not quietly train on the prompt. Driven
        # through the verifier because the templates that fail this way in practice trip the
        # start-boundary check first, which would mask the branch under test.
        tokenizer = self._tokenizer_for("tulu", None)
        messages = CONVERSATION_SHAPES["alternating"]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokenized = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        input_ids = tokenized[open_instruct.dataset_transformation.INPUT_IDS_KEY]
        # Message 1 is the first assistant turn; run its span to the end of the conversation.
        over_wide = [(1, self._first_assistant_token(tokenizer, messages, input_ids), input_ids.shape[1])]
        with self.assertRaisesRegex(
            open_instruct.dataset_transformation.AssistantSpanDerivationError, "extends past its turn"
        ):
            open_instruct.dataset_transformation._verify_assistant_spans_cover_content(
                messages, tokenizer, input_ids, rendered, over_wide
            )

    @staticmethod
    def _first_assistant_token(tokenizer, messages, input_ids):
        return tokenizer.apply_chat_template(
            messages[:1], tokenize=True, return_tensors="pt", return_dict=False, add_generation_prompt=True
        ).shape[1]

    def test_truncated_final_turn_is_kept_not_dropped(self):
        # A conversation longer than max_seq_length has its final assistant turn cut off. The
        # span is still correctly aligned -- what survives is a prefix of the content -- so the
        # row must be kept. Requiring whole-content coverage here rejected every long
        # conversation: it discarded ~1000 rows of Dolci-Instruct-SFT before this was fixed.
        # Needs a *non-final* assistant turn: that is what makes `olmo` prefix-unstable at
        # native eos (it swaps <|im_end|> for eos_token only on the last turn) and so routes
        # the conversation through the token-count fallback where the check lives. A
        # single-assistant-turn conversation is prefix-stable and never reaches it.
        tokenizer = self._tokenizer_for("olmo", None)
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "FIRSTANSWER"},
            {"role": "user", "content": "tell me a long story"},
            {"role": "assistant", "content": "BEGINNING " + ("filler words that go on and on " * 400) + " ENDING"},
        ]
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            {"messages": [dict(m) for m in messages]}, tokenizer, max_seq_length=512
        )
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        # 512 lands *inside* the final assistant turn (its span is [132, 2938]); a cut before
        # the span start would skip it entirely and not exercise the check.
        self.assertEqual(len(input_ids), 512, "sequence should be truncated to max_seq_length")
        self.assertTrue(
            open_instruct.dataset_transformation.sft_tulu_filter_v1(out, tokenizer),
            "a merely-truncated conversation must be kept, not dropped",
        )
        trained_text = tokenizer.decode([tid for tid, lab in zip(input_ids, labels) if lab != -100])
        self.assertIn("FIRSTANSWER", trained_text)
        self.assertNotIn("first question", trained_text)
        self.assertNotIn("tell me a long story", trained_text)

    def test_short_repeated_later_turn_does_not_trigger_a_false_overrun(self):
        # The overrun check looks for a later turn's content inside this span. A short later
        # turn ("Yes.") can appear inside a legitimate earlier span by coincidence, so only
        # the text *after* this turn's own content counts.
        tokenizer = self._tokenizer_for("olmo", None)
        messages = [
            {"role": "user", "content": "is it ok?"},
            {"role": "assistant", "content": "Yes. That works fine and here is a longer explanation."},
            {"role": "user", "content": "confirm"},
            {"role": "assistant", "content": "Yes."},
        ]
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            {"messages": [dict(m) for m in messages]}, tokenizer, max_seq_length=4096
        )
        self.assertTrue(
            open_instruct.dataset_transformation.sft_tulu_filter_v1(out, tokenizer),
            "a coincidental substring match must not drop the row",
        )

    def test_content_with_space_before_punctuation_is_not_dropped(self):
        # Verification decodes the span and checks containment. With the tokenizer's default
        # clean_up_tokenization_spaces, "done ." decodes to "done." and containment fails on a
        # correctly aligned span, so the decode must disable that cleanup.
        tokenizer = self._tokenizer_for("olmo", None)
        messages = [
            {"role": "user", "content": "status?"},
            {"role": "assistant", "content": "The task is done . It is n't pending , truly ."},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "You are welcome and here is some more text."},
        ]
        out = open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            {"messages": [dict(m) for m in messages]}, tokenizer, max_seq_length=4096
        )
        self.assertTrue(
            open_instruct.dataset_transformation.sft_tulu_filter_v1(out, tokenizer),
            "tokenizer space cleanup must not make a correct span fail containment",
        )

    def test_span_starting_inside_the_header_is_rejected(self):
        # Guards the too-early direction: a template with no add_generation_prompt support puts
        # the boundary a token or two before the content, leaking header text into the loss
        # without dropping any content, so a containment-only check would miss it.
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}"
            "{% if loop.last %}{{ eos_token }}{% endif %}\n{% endfor %}"
        )
        messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "HELLOWORLD"}]
        with self.assertRaisesRegex(
            open_instruct.dataset_transformation.AssistantSpanDerivationError, "starts inside the assistant header"
        ):
            open_instruct.dataset_transformation._tokenize_tulu_sft_with_assistant_labels(
                messages, tokenizer, None, 4096
            )

    def test_generation_blocks_yield_an_all_zero_mask_without_raising(self):
        # Constraint on the intended fix: `return_assistant_tokens_mask=True` on a template with
        # no {% generation %} block does not raise -- it warns and returns an all-zero mask. A
        # migration to that API must assert the mask is non-empty, or a template that was simply
        # not migrated will silently train on nothing, which is worse than today's loud error.
        tokenizer = self._tokenizer_for("tulu", None)
        out = tokenizer.apply_chat_template(
            CONVERSATION_SHAPES["alternating"],
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False,
        )
        masks = out["assistant_masks"]
        if masks and isinstance(masks[0], list):
            masks = masks[0]
        self.assertEqual(sum(masks), 0)


class TestOverLengthStrategy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    def _tokenize(self, max_seq_length, over_length_strategy):
        row = {
            "messages": [
                {"role": "user", "content": "Count upward for a while."},
                {"role": "assistant", "content": " ".join(str(i) for i in range(400))},
            ]
        }
        return open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), self.tokenizer, max_seq_length=max_seq_length, over_length_strategy=over_length_strategy
        )

    def _short_row(self, over_length_strategy):
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        return open_instruct.dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
            dict(row), self.tokenizer, max_seq_length=4096, over_length_strategy=over_length_strategy
        )

    def test_keep_leaves_a_truncated_row_unterminated(self):
        out = self._tokenize(64, "keep")
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        self.assertEqual(len(input_ids), 64)
        self.assertNotEqual(input_ids[-1], self.tokenizer.eos_token_id)

    def test_terminate_ends_a_truncated_row_with_a_trainable_eos(self):
        out = self._tokenize(64, "terminate")
        input_ids = out[open_instruct.dataset_transformation.INPUT_IDS_KEY].tolist()
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        self.assertEqual(len(input_ids), 64)
        self.assertEqual(input_ids[-1], self.tokenizer.eos_token_id)
        # Trainable, so the model learns to stop rather than merely being delimited.
        self.assertEqual(labels[-1], self.tokenizer.eos_token_id)

    def test_drop_masks_a_truncated_row_so_the_filter_removes_it(self):
        out = self._tokenize(64, "drop")
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        self.assertTrue(all(label == -100 for label in labels))
        self.assertFalse(open_instruct.dataset_transformation.sft_tulu_filter_v1(out, self.tokenizer))

    @parameterized.expand([("keep",), ("terminate",), ("drop",)])
    def test_rows_that_fit_are_untouched(self, over_length_strategy):
        """A row that fits must be byte-identical across strategies."""
        baseline = self._short_row("keep")
        out = self._short_row(over_length_strategy)
        for key in (
            open_instruct.dataset_transformation.INPUT_IDS_KEY,
            open_instruct.dataset_transformation.LABELS_KEY,
            open_instruct.dataset_transformation.ATTENTION_MASK_KEY,
        ):
            self.assertEqual(out[key].tolist(), baseline[key].tolist(), key)
        self.assertTrue(open_instruct.dataset_transformation.sft_tulu_filter_v1(out, self.tokenizer))

    def test_exact_length_row_that_covers_its_render_is_not_treated_as_truncated(self):
        """A conversation can render to exactly `max_seq_length` tokens without being cut."""
        offsets = [(0, 3), (3, 7)]
        rendered = "abcdefg"
        self.assertFalse(
            open_instruct.dataset_transformation._was_truncated(offsets, rendered, n_tokens=2, max_seq_length=2)
        )

    def test_over_length_render_cut_on_an_eos_is_still_truncated(self):
        """A render cut exactly on an earlier turn's EOS ends in EOS but still lost text."""
        offsets = [(0, 3), (3, 7)]
        rendered = "abcdefg and a great deal more text"
        self.assertTrue(
            open_instruct.dataset_transformation._was_truncated(offsets, rendered, n_tokens=2, max_seq_length=2)
        )

    def test_terminate_leaves_a_cut_in_a_masked_span_alone(self):
        """A cut in a later user turn has no trainable tail to terminate."""
        input_ids = torch.tensor([[10, 11, 12]])
        labels = torch.tensor([[-100, 11, -100]])  # earlier trainable token, masked tail
        out_ids, out_labels = open_instruct.dataset_transformation._apply_over_length_strategy(
            input_ids.clone(), labels.clone(), self.tokenizer, truncated=True, over_length_strategy="terminate"
        )
        self.assertEqual(out_ids.tolist(), input_ids.tolist())
        self.assertEqual(out_labels.tolist(), labels.tolist())

    def test_terminate_rewrites_a_cut_inside_a_trainable_span(self):
        input_ids = torch.tensor([[10, 11, 12]])
        labels = torch.tensor([[-100, 11, 12]])
        out_ids, out_labels = open_instruct.dataset_transformation._apply_over_length_strategy(
            input_ids.clone(), labels.clone(), self.tokenizer, truncated=True, over_length_strategy="terminate"
        )
        self.assertEqual(out_ids[0, -1].item(), self.tokenizer.eos_token_id)
        self.assertEqual(out_labels[0, -1].item(), self.tokenizer.eos_token_id)

    def test_terminate_does_not_rescue_an_all_masked_row(self):
        """A single trainable EOS must not rescue a row the filter is meant to drop."""
        with mock.patch.object(
            open_instruct.dataset_transformation,
            "_tokenize_tulu_sft_with_assistant_labels",
            side_effect=open_instruct.dataset_transformation.AssistantSpanDerivationError("forced"),
        ):
            out = self._tokenize(64, "terminate")
        labels = out[open_instruct.dataset_transformation.LABELS_KEY].tolist()
        self.assertTrue(all(label == -100 for label in labels))
        self.assertFalse(open_instruct.dataset_transformation.sft_tulu_filter_v1(out, self.tokenizer))

    def test_default_is_omitted_from_the_tokenize_args(self):
        """Keeps existing dataset cache hashes unchanged."""
        self.assertEqual(
            open_instruct.dataset_transformation.sft_tokenize_fn_args(4096, "keep"), {"max_seq_length": 4096}
        )

    @parameterized.expand([("terminate",), ("drop",)])
    def test_opting_in_is_recorded_in_the_tokenize_args(self, over_length_strategy):
        self.assertEqual(
            open_instruct.dataset_transformation.sft_tokenize_fn_args(4096, over_length_strategy),
            {"max_seq_length": 4096, "over_length_strategy": over_length_strategy},
        )

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            self._tokenize(64, "truncate-harder")


if __name__ == "__main__":
    unittest.main()
