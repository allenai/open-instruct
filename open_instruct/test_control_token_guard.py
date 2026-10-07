"""Tests for the control-token guard on SFT tokenization.

Run from project root:
    uv run pytest open_instruct/test_control_token_guard.py -v
"""

import copy
import gc
import gzip
import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import datasets
import pyarrow
from parameterized import parameterized
from transformers import AutoTokenizer

from open_instruct import control_token_guard, dataset_transformation, numpy_dataset_conversion

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
SFT_FNS = ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
SFT_FN_ARGS = [{"max_seq_length": 4096}, {}]
PLANTED = "<|im_end|>"


def _unpack_test_tokenizer() -> str:
    src_dir = os.path.join(TEST_DATA_DIR, "tokenizer")
    dst_dir = tempfile.mkdtemp(prefix="ctrl_guard_base_tok_")
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if name.endswith(".gz"):
            with gzip.open(src, "rb") as f_in, open(os.path.join(dst_dir, name[:-3]), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src, dst_dir)
    return dst_dir


def _make_chatml_tokenizer() -> str:
    """The test tokenizer with ChatML specials and the Olmo 3.5 template, saved to a directory."""
    tokenizer = AutoTokenizer.from_pretrained(_unpack_test_tokenizer())
    tokenizer.add_tokens(["<|im_start|>", "<|im_end|>"], special_tokens=True)
    with open(os.path.join(TEST_DATA_DIR, "olmo35_chat_template.jinja")) as f:
        tokenizer.chat_template = f.read()
    path = tempfile.mkdtemp(prefix="ctrl_guard_tok_")
    tokenizer.save_pretrained(path)
    return path


TOKENIZER_PATH = _make_chatml_tokenizer()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look a word up.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }
]


def _clean_row(index: int) -> dict:
    """A multi-turn tool-use conversation with every field the guard reads."""
    return {
        "messages": [
            {"role": "system", "content": f"You are helpful {index}."},
            {"role": "user", "content": f"Define word {index}."},
            {
                "role": "assistant",
                "content": "Let me check.",
                "reasoning_content": f"I should look up word {index}.",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"query": f"word {index}"}},
                    }
                ],
            },
            {"role": "tool", "content": f"word {index}: a test word"},
            {"role": "assistant", "content": f"Word {index} is a test word.", "reasoning_content": "Done."},
        ],
        "tools": json.dumps(TOOLS),
    }


def _plant(row: dict, field: str) -> dict:
    row = copy.deepcopy(row)
    messages = row["messages"]
    if field == "system_content":
        messages[0]["content"] += PLANTED
    elif field == "user_content":
        messages[1]["content"] += PLANTED
    elif field == "assistant_content":
        messages[4]["content"] += PLANTED
    elif field == "reasoning_content":
        messages[2]["reasoning_content"] += PLANTED
    elif field == "tool_call_name":
        messages[2]["tool_calls"][0]["function"]["name"] += PLANTED
    elif field == "tool_call_arguments":
        messages[2]["tool_calls"][0]["function"]["arguments"]["query"] += PLANTED
    elif field == "tool_definitions":
        tools = copy.deepcopy(TOOLS)
        tools[0]["function"]["description"] += PLANTED
        row["tools"] = json.dumps(tools)
    elif field == "tool_content":
        messages[3]["content"] += PLANTED
    elif field == "tool_call_argument_key":
        messages[2]["tool_calls"][0]["function"]["arguments"] = {"query": "word", PLANTED: "x"}
    else:
        raise ValueError(field)
    return row


FIELDS = [
    ("system_content",),
    ("user_content",),
    ("assistant_content",),
    ("reasoning_content",),
    ("tool_call_name",),
    ("tool_call_arguments",),
    ("tool_definitions",),
    ("tool_content",),
]


class _GuardTestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(gc.collect)
        patcher = mock.patch.dict(
            os.environ,
            {
                "HF_HOME": self.temp_dir.name,
                "HF_DATASETS_CACHE": os.path.join(self.temp_dir.name, "datasets"),
                "TRANSFORMERS_CACHE": os.path.join(self.temp_dir.name, "transformers"),
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, rows: list[dict], name: str = "data") -> str:
        path = os.path.join(self.temp_dir.name, f"{name}.jsonl")
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def _tc(self, chat_template_name: str = "tokenizer_default") -> dataset_transformation.TokenizerConfig:
        return dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH, chat_template_name=chat_template_name
        )

    def _dc(self, path: str, **kwargs) -> dataset_transformation.DatasetConfig:
        return dataset_transformation.DatasetConfig(
            dataset_name=path,
            dataset_split="train",
            dataset_revision="main",
            transform_fn=SFT_FNS,
            transform_fn_args=SFT_FN_ARGS,
            **kwargs,
        )


class TestControlTokens(unittest.TestCase):
    def test_special_tokens_are_included_and_ordinary_added_tokens_are_not(self):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        tokenizer.add_tokens(["<think>"], special_tokens=False)
        tokens = control_token_guard.control_tokens(tokenizer)
        for token in ("<|im_start|>", "<|im_end|>", tokenizer.eos_token, tokenizer.bos_token):
            self.assertIn(token, tokens)
        self.assertNotIn("<think>", tokens)


class TestRowMask(unittest.TestCase):
    TOKENS = ["<|im_end|>", "<|endoftext|>"]

    @parameterized.expand(FIELDS)
    def test_planted_literal_is_found(self, field):
        rows = [_clean_row(0), _plant(_clean_row(1), field), _clean_row(2)]
        dataset = datasets.Dataset.from_list(rows)
        mask = control_token_guard.control_token_row_mask(dataset, ["messages", "tools"], self.TOKENS)
        self.assertEqual(mask.tolist(), [False, True, False])

    def test_struct_tool_definitions_and_null_lists(self):
        rows = [_clean_row(0), _clean_row(1), _clean_row(2)]
        for row in rows:
            row["tools"] = copy.deepcopy(TOOLS)
        rows[2]["tools"][0]["function"]["parameters"]["properties"]["query"]["type"] = "string<|endoftext|>"
        rows[1]["messages"][2]["tool_calls"] = None
        dataset = datasets.Dataset.from_list(rows)
        self.assertTrue(pyarrow.types.is_list(dataset.data.schema.field("tools").type))
        mask = control_token_guard.control_token_row_mask(dataset, ["messages", "tools"], self.TOKENS)
        self.assertEqual(mask.tolist(), [False, False, True])

    def test_struct_field_names_flag_every_row_that_renders_them(self):
        # Decoding fills a field missing from a row with None, and templates render None values,
        # so a special token in a field name reaches every row whose struct is set.
        rows = [
            {"messages": [{"role": "user", "content": "a"}], "tools": [{"properties": {PLANTED: "x"}}]},
            {"messages": [{"role": "user", "content": "b"}], "tools": [{"properties": {"query": "y"}}]},
            {"messages": [{"role": "user", "content": "c"}], "tools": None},
        ]
        # Built from Arrow directly so `tools` is a struct whatever `datasets` would infer.
        dataset = datasets.Dataset(pyarrow.Table.from_pylist(rows))
        mask = control_token_guard.control_token_row_mask(dataset, ["messages", "tools"], self.TOKENS)
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_respects_selected_and_upsampled_rows(self):
        dataset = datasets.Dataset.from_list([_clean_row(0), _plant(_clean_row(1), "user_content")])
        selected = dataset.select([1, 0, 1])
        mask = control_token_guard.control_token_row_mask(selected, ["messages", "tools"], self.TOKENS)
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_escaped_json_spellings_are_found(self):
        # The pipeline decodes JSON-string tool schemas and `Json` features before rendering, so an
        # escaped spelling still renders, and tokenizes, as the control token.
        escaped = json.dumps(TOOLS).replace("Look", "\\u003c|im_end|\\u003eLook")
        self.assertIn(PLANTED, json.dumps(json.loads(escaped), ensure_ascii=False))
        dataset = datasets.Dataset.from_list([{**_clean_row(0), "tools": escaped}, _clean_row(1)])
        mask = control_token_guard.control_token_row_mask(dataset, ["tools"], self.TOKENS, json_columns=["tools"])
        self.assertEqual(mask.tolist(), [True, False])
        storage = pyarrow.array([r'{"q": "\u003C|im_end|>"}', '{"q": "fine"}'])
        extension = pyarrow.ExtensionArray.from_storage(pyarrow.json_(), storage)
        patterns = control_token_guard._patterns(self.TOKENS)
        self.assertEqual(
            control_token_guard._rows_with_match(extension, patterns, json_text=False).tolist(), [True, False]
        )

    def test_json_matches_are_confirmed_by_decoding(self):
        patterns = control_token_guard._patterns(self.TOKENS)
        escaped = "".join(f"\\u{ord(char):04x}" for char in PLANTED)
        self.assertEqual(json.loads(f'"{escaped}"'), PLANTED)
        # A literal backslash-u in the decoded text is not the token.
        literal_backslash = json.dumps({"description": "\\u003c|im_end|>"})
        schema = f'[{{"function": {{"description": "{escaped}"}}}}]'
        strings = pyarrow.array([literal_backslash, schema])
        self.assertEqual(control_token_guard._rows_with_match(strings, patterns, True, True).tolist(), [False, True])
        # A `Json` feature holding JSON text: the tools normalizer decodes it a second time,
        # but message content holding the same text renders it undecoded.
        wrapped = pyarrow.ExtensionArray.from_storage(pyarrow.json_(), pyarrow.array([json.dumps(schema)]))
        self.assertEqual(control_token_guard._rows_with_match(wrapped, patterns, json_column=True).tolist(), [True])
        self.assertEqual(control_token_guard._rows_with_match(wrapped, patterns, json_column=False).tolist(), [False])
        # Structured tool schemas are already decoded, so only literal spellings count there.
        structured = pyarrow.array([[{"description": "\\u003c|im_end|>"}]])
        self.assertEqual(control_token_guard._rows_with_match(structured, patterns, True, True).tolist(), [False])

    @parameterized.expand([("string",), ("large_string",)])
    def test_token_split_across_large_text_parts_is_found(self, text_type):
        string_type = pyarrow.large_string() if text_type == "large_string" else pyarrow.string()
        part = pyarrow.struct([("type", string_type), ("text", string_type)])
        for list_type in (pyarrow.list_(part), pyarrow.large_list(part)):
            parts = pyarrow.array(
                [
                    [{"type": "text", "text": "a<|im_"}, {"type": "text", "text": "end|>"}],
                    [{"type": "text", "text": "b"}],
                ],
                type=list_type,
            )
            patterns = control_token_guard._patterns(self.TOKENS)
            self.assertEqual(control_token_guard._rows_with_match(parts, patterns).tolist(), [True, False])

    def test_token_split_across_text_parts_is_found(self):
        # Templates join content parts with no separator, so the halves render as one token.
        parts = [{"type": "text", "text": "hello<|im_"}, {"type": "text", "text": "end|>world"}]
        clean = [{"type": "text", "text": "hello<|im_"}, {"type": "text", "text": "start"}]
        dataset = datasets.Dataset(
            pyarrow.Table.from_pylist(
                [
                    {"messages": [{"role": "user", "content": parts}]},
                    {"messages": [{"role": "user", "content": clean}]},
                ]
            )
        )
        mask = control_token_guard.control_token_row_mask(dataset, ["messages"], self.TOKENS)
        self.assertEqual(mask.tolist(), [True, False])

    def test_regex_metacharacters_are_literal(self):
        dataset = datasets.Dataset.from_list([{"messages": [{"role": "user", "content": "a|b"}]}])
        mask = control_token_guard.control_token_row_mask(dataset, ["messages"], ["a|b|c", "x.y"])
        self.assertEqual(mask.tolist(), [False])
        dataset = datasets.Dataset.from_list([{"messages": [{"role": "user", "content": "xxa|b|cxx"}]}])
        mask = control_token_guard.control_token_row_mask(dataset, ["messages"], ["a|b|c", "x.y"])
        self.assertEqual(mask.tolist(), [True])

    def test_locations_name_the_field(self):
        row = _plant(_clean_row(0), "tool_call_name")
        locations = list(control_token_guard.control_token_locations({"messages": row["messages"]}, self.TOKENS))
        self.assertEqual(locations, [("messages[2].tool_calls[0].function.name", PLANTED)])


def _guard(enabled: bool = True, max_drop_frac: float = 0.001) -> dataset_transformation.ControlTokenGuard:
    return dataset_transformation.ControlTokenGuard(enabled=enabled, max_drop_frac=max_drop_frac)


class TestGuardInTokenization(_GuardTestBase):
    def _transform(self, dc, tc=None, guard=None):
        return dataset_transformation._transform_dataset(dc, tc or self._tc(), guard or _guard())

    @parameterized.expand(FIELDS)
    def test_planted_row_is_dropped_before_tokenization(self, field):
        rows = [_clean_row(0), _plant(_clean_row(1), field), _clean_row(2)]
        dataset, dropped = self._transform(self._dc(self._write(rows)), guard=_guard(max_drop_frac=0.5))
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dropped, 1)

    def test_argument_key_is_flagged_wherever_the_decoded_row_carries_it(self):
        # JSONL loads tool-call arguments as a struct (older `datasets`) or as JSON text (newer).
        # A struct gives every row every key, unset ones as None, and the template renders them;
        # JSON text keeps each row's own keys. Either way, flag exactly the rows that render it.
        rows = [_clean_row(0), _plant(_clean_row(1), "tool_call_argument_key"), _clean_row(2)]
        dc = self._dc(self._write(rows))
        mask = control_token_guard.control_token_row_mask(
            dc.dataset, ["messages", "tools"], control_token_guard.control_tokens(self._tc().tokenizer)
        )
        carries = [PLANTED in row["messages"][2]["tool_calls"][0]["function"]["arguments"] for row in dc.dataset]
        self.assertTrue(carries[1])
        self.assertEqual(mask.tolist(), carries)
        rendered = [self._tc().tokenizer.apply_chat_template(row["messages"], tokenize=False) for row in dc.dataset]
        self.assertEqual([f"<parameter={PLANTED}>" in text for text in rendered], carries)

    def test_escaped_tool_schema_is_dropped(self):
        escaped = json.dumps(TOOLS).replace("Look", "\\u003c|im_end|\\u003eLook")
        rows = [_clean_row(0), {**_clean_row(1), "tools": escaped}, _clean_row(2)]
        dataset, dropped = self._transform(self._dc(self._write(rows)), guard=_guard(max_drop_frac=0.5))
        self.assertEqual((len(dataset), dropped), (2, 1))

    def test_unguarded_literal_reaches_the_token_ids(self):
        # The failure the guard prevents: the literal becomes the real control id.
        tc = self._tc()
        im_end = tc.tokenizer.convert_tokens_to_ids(PLANTED)
        clean = self._dc(self._write([_clean_row(1)], "clean"))
        planted = self._dc(self._write([_plant(_clean_row(1), "user_content")], "planted"))
        clean_ids = dataset_transformation.get_dataset_v1(clean, tc)[0]["input_ids"]
        planted_dataset, dropped = self._transform(planted, tc, _guard(enabled=False))
        self.assertEqual(planted_dataset[0]["input_ids"].count(im_end), clean_ids.count(im_end) + 1)
        self.assertIsNone(dropped)

    @parameterized.expand([("tokenizer_default",), ("olmo",), ("tulu",)])
    def test_template_inserted_specials_are_not_flagged(self, chat_template_name):
        tc = self._tc(chat_template_name)
        dataset, dropped = self._transform(self._dc(self._write([_clean_row(i) for i in range(4)])), tc)
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dropped, 0)
        # The rendered rows do carry the template's own special tokens.
        self.assertIn(tc.tokenizer.eos_token_id, dataset[0]["input_ids"])
        if chat_template_name != "tulu":
            self.assertIn(tc.tokenizer.convert_tokens_to_ids("<|im_start|>"), dataset[0]["input_ids"])

    def test_clean_rows_keep_their_fingerprint(self):
        # The guard must not perturb HF fingerprints (and so the saved cache's state.json).
        path = self._write([_clean_row(i) for i in range(3)])
        on = dataset_transformation.get_dataset_v1(self._dc(path), self._tc())
        off = dataset_transformation.get_dataset_v1(self._dc(path), self._tc(), _guard(enabled=False))
        self.assertEqual(on._fingerprint, off._fingerprint)

    def test_threshold_error(self):
        rows = [_clean_row(i) for i in range(9)] + [_plant(_clean_row(9), "assistant_content")]
        with self.assertRaisesRegex(ValueError, "control_token_max_drop_frac"):
            dataset_transformation.get_dataset_v1(self._dc(self._write(rows)), self._tc())
        dataset, dropped = self._transform(self._dc(self._write(rows)), guard=_guard(max_drop_frac=0.1))
        self.assertEqual((len(dataset), dropped), (9, 1))

    def test_non_sft_transforms_are_not_guarded(self):
        dc = dataset_transformation.DatasetConfig(
            dataset_name=os.path.join(TEST_DATA_DIR, "sft_sample.jsonl"),
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["rlvr_tokenize_v1"],
            transform_fn_args=[{}],
        )
        self.assertEqual(dataset_transformation._control_token_guard_columns(dc), [])

    def _cached_statistics(self, rows, cache_mode="local", **kwargs):
        return dataset_transformation.get_cached_dataset_tulu_with_statistics(
            [self._write(rows), "1.0"],
            ["train"],
            self._tc(),
            SFT_FNS,
            SFT_FN_ARGS,
            dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            dataset_cache_mode=cache_mode,
            hf_entity="test-entity",
            dataset_local_cache_dir=os.path.join(self.temp_dir.name, "cache"),
            **kwargs,
        )

    def test_local_cache_statistics_record_drops(self):
        rows = [_clean_row(0), _plant(_clean_row(1), "tool_content"), _clean_row(2)]
        _, statistics = self._cached_statistics(rows, control_token_max_drop_frac=0.5)
        self.assertEqual(statistics["per_dataset_stats"][0]["control_token_rows_dropped"], 1)
        self.assertEqual(statistics["control_token_guard"]["version"], "v1")
        self.assertIn(PLANTED, statistics["control_token_guard"]["tokens"])

    def test_hf_cache_statistics_record_drops_when_transforming(self):
        rows = [_clean_row(0), _plant(_clean_row(1), "tool_content"), _clean_row(2)]
        with mock.patch.object(dataset_transformation, "revision_exists", return_value=False):
            dataset, statistics = self._cached_statistics(
                rows, cache_mode="hf", dataset_skip_cache=True, control_token_max_drop_frac=0.5
            )
        self.assertEqual(len(dataset), 2)
        self.assertEqual(statistics["per_dataset_stats"][0]["control_token_rows_dropped"], 1)


class TestCacheKey(_GuardTestBase):
    def _hash(self, path: str, guard=None) -> str:
        return dataset_transformation.compute_config_hash([self._dc(path)], self._tc(), guard or _guard())

    def test_clean_mix_keeps_its_key(self):
        path = self._write([_clean_row(i) for i in range(3)])
        self.assertEqual(self._hash(path), self._hash(path, _guard(enabled=False)))
        self.assertEqual(self._hash(path), self._hash(path, _guard(max_drop_frac=0.5)))

    def test_mix_that_loses_rows_gets_a_new_key(self):
        path = self._write([_clean_row(0), _plant(_clean_row(1), "user_content")])
        self.assertNotEqual(self._hash(path), self._hash(path, _guard(enabled=False)))
        # A cache built under a permissive threshold must not satisfy a stricter run.
        self.assertNotEqual(self._hash(path), self._hash(path, _guard(max_drop_frac=0.5)))

    def test_hashed_config_matches_the_pre_guard_layout(self):
        # Pins the hashed dict to what it was before the guard existed, so a clean mix's
        # key cannot drift from the caches already on disk.
        path = self._write([_clean_row(0)])
        combined = dataset_transformation._config_hash_payload([self._dc(path)], self._tc())
        self.assertEqual(set(combined), {"cache_version", "dataset_configs", "tokenizer_config", "chat_template_hash"})
        self.assertEqual(
            set(combined["dataset_configs"][0]),
            {
                "dataset_name",
                "dataset_split",
                "dataset_revision",
                "dataset_range",
                "transform_fn",
                "transform_fn_args",
                "dataset_config_seed",
                "original_dataset_size",
                "is_upsampled",
            },
        )


class TestNumpyCaches(_GuardTestBase):
    def _convert(self, output_dir: pathlib.Path, rows_path: str, **kwargs) -> dict:
        numpy_dataset_conversion.convert_hf_to_numpy_sft(
            output_dir=output_dir,
            dataset_mixer_list=[rows_path, "1.0"],
            dataset_mixer_list_splits=["train"],
            tc=self._tc(),
            dataset_transform_fn=SFT_FNS,
            transform_fn_args=SFT_FN_ARGS,
            dataset_target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
            **kwargs,
        )
        with open(output_dir / "dataset_statistics.json") as f:
            return json.load(f)

    def test_clean_mix_gives_byte_identical_files(self):
        path = self._write([_clean_row(i) for i in range(5)])
        out = pathlib.Path(self.temp_dir.name)
        on = self._convert(out / "on", path)
        off = self._convert(out / "off", path, drop_control_token_rows=False)
        for pattern in ("token_ids_part_*.npy", "labels_mask_part_*.npy", "token_ids_part_*.csv.gz"):
            on_files = sorted((out / "on").glob(pattern))
            off_files = sorted((out / "off").glob(pattern))
            self.assertTrue(on_files)
            self.assertEqual([f.name for f in on_files], [f.name for f in off_files])
            for on_file, off_file in zip(on_files, off_files):
                if pattern.endswith(".gz"):
                    # gzip headers carry a write time, so compare the payloads.
                    with gzip.open(on_file) as on_fh, gzip.open(off_file) as off_fh:
                        self.assertEqual(on_fh.read(), off_fh.read())
                else:
                    self.assertEqual(on_file.read_bytes(), off_file.read_bytes())
        self.assertEqual(on["per_dataset_statistics"][0]["control_token_rows_dropped"], 0)
        self.assertEqual(on["overall_statistics"]["control_token_rows_dropped"], 0)
        self.assertIsNotNone(on["configuration"]["control_token_guard"])
        self.assertIsNone(off["overall_statistics"]["control_token_rows_dropped"])
        self.assertIsNone(off["configuration"]["control_token_guard"])

    def test_dropped_rows_are_recorded(self):
        rows = [_clean_row(0), _plant(_clean_row(1), "reasoning_content"), _clean_row(2)]
        stats = self._convert(
            pathlib.Path(self.temp_dir.name) / "out", self._write(rows), control_token_max_drop_frac=0.5
        )
        self.assertEqual(stats["overall_statistics"]["total_instances"], 2)
        self.assertEqual(stats["per_dataset_statistics"][0]["control_token_rows_dropped"], 1)
        self.assertEqual(stats["overall_statistics"]["control_token_rows_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
