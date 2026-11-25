"""Tests for open_instruct.token_row_cache: the row cache must not change one byte of the tokenize map's output.

Run from project root:
    uv run pytest open_instruct/test_token_row_cache.py -v
"""

import gc
import gzip
import json
import logging
import os
import pathlib
import shutil
import tempfile
import unittest
import unittest.mock

import datasets
import numpy as np
import pyarrow as pa
from parameterized import parameterized
from pyarrow import ipc, parquet

from open_instruct import dataset_transformation, token_row_cache

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TOKENIZE_FN = "sft_tulu_tokenize_and_truncate_v1"


def _get_tokenizer_path():
    src_dir = os.path.join(TEST_DATA_DIR, "tokenizer")
    dst_dir = tempfile.mkdtemp(prefix="test_tokenizer_")
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if name.endswith(".gz"):
            with gzip.open(src, "rb") as f_in, open(os.path.join(dst_dir, name[:-3]), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src, dst_dir)
    return dst_dir


TOKENIZER_PATH = _get_tokenizer_path()

_TOOLS = json.dumps(
    [{"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {}}}}]
)


def _rows(num_rows, seed):
    """Chats with one to three turns, a tools column that is sometimes null, and rows the filter drops: one with no
    assistant turn, and one whose prompt alone fills max_seq_length. Includes exact duplicates and empty content."""
    rng = np.random.default_rng(seed)
    words = [
        "the",
        "of",
        "and",
        "to",
        "model",
        "token",
        "data",
        "answer",
        "think",
        "step",
        "proof",
        "value",
        "0",
        "1",
        "2",
        "=",
        "+",
        "naïve",
        "東京",
        "🙂",
    ]

    def text(n):
        return " ".join(words[i] for i in rng.integers(0, len(words), size=n))

    rows = []
    for i in range(num_rows):
        messages = []
        for _ in range(int(rng.integers(1, 4))):
            messages.append({"role": "user", "content": text(int(rng.integers(1, 30)))})
            messages.append({"role": "assistant", "content": text(int(rng.integers(1, 60)))})
        rows.append({"id": f"row-{i}", "messages": messages, "tools": _TOOLS if i % 3 == 0 else None})
    rows.append({"id": "no-assistant", "messages": [{"role": "user", "content": text(10)}], "tools": None})
    rows.append(
        {
            "id": "long-prompt",
            "messages": [{"role": "user", "content": text(3000)}, {"role": "assistant", "content": "ok"}],
            "tools": None,
        }
    )
    rows[3]["messages"][-1] = {"role": "assistant", "content": ""}
    rows.extend(dict(rows[i]) for i in (1, 2, 5))
    return rows


def _swap_final_turns(rows, every):
    """Replace the final assistant turn of every `every`-th row and drop every 31st; return (rows, rows changed)."""
    out, changed = [], 0
    for i, row in enumerate(rows):
        if i % 31 == 30:
            continue
        if i % every == 0 and row["messages"][-1]["role"] == "assistant":
            row = {**row, "messages": [*row["messages"][:-1], {"role": "assistant", "content": f"new answer {i}"}]}
            changed += 1
        out.append(row)
    return out, changed


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class TestRowKey(unittest.TestCase):
    def test_distinguishes_types_order_and_absent_tools(self):
        messages = [{"role": "user", "content": "1"}]
        keys = {
            token_row_cache.row_key(messages, None, has_tools=False),
            token_row_cache.row_key(messages, None, has_tools=True),
            token_row_cache.row_key(messages, "", has_tools=True),
            token_row_cache.row_key([{"role": "user", "content": 1}], None, has_tools=False),
            token_row_cache.row_key([{"role": "user", "content": 1.0}], None, has_tools=False),
            token_row_cache.row_key([{"role": "user", "content": True}], None, has_tools=False),
            token_row_cache.row_key([{"content": "1", "role": "user"}], None, has_tools=False),
            token_row_cache.row_key([{"role": "user", "content": "1", "extra": None}], None, has_tools=False),
            token_row_cache.row_key([{"role": "user", "content": "1"}, {}], None, has_tools=False),
        }
        self.assertEqual(len(keys), 9)
        self.assertEqual(
            token_row_cache.row_key(messages, None, has_tools=False),
            token_row_cache.row_key([{"role": "user", "content": "1"}], None, has_tools=False),
        )

    def test_unknown_types_are_uncacheable(self):
        with self.assertRaises(token_row_cache.Uncacheable):
            token_row_cache.row_key([{"role": "user", "content": np.int64(1)}], None, has_tools=False)
        self.assertEqual(token_row_cache._row_key_or_empty({"messages": [{"content": (1, 2)}]}), b"")


class TestTokenRowCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(gc.collect)
        self.root = pathlib.Path(self.temp_dir.name)
        self.store = self.root / "store"
        self.runs = 0
        self._set_env(BEAKER_ASSIGNED_CPU_COUNT="1")
        self._set_env(**{token_row_cache.ROOT_ENV: None})
        self.handler = _ListHandler()
        token_row_cache.logger.addHandler(self.handler)
        self.addCleanup(token_row_cache.logger.removeHandler, self.handler)
        # Under pytest the root logger stays at WARNING, which would drop the cache's INFO summaries.
        self.addCleanup(token_row_cache.logger.setLevel, token_row_cache.logger.level)
        token_row_cache.logger.setLevel(logging.INFO)

    def _set_env(self, **values):
        """Set (or, for None, remove) variables until the test ends. unittest.mock.patch.dict cannot be used: the
        first get_dataset_v1 in a process replaces the os.environ object, so patch.dict would restore a stale one."""
        for name, value in values.items():
            old = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
            self.addCleanup(
                lambda name=name, old=old: os.environ.pop(name, None)
                if old is None
                else os.environ.update({name: old})
            )

    def _write(self, name, rows):
        path = self.root / f"{name}.parquet"
        parquet.write_table(pa.Table.from_pylist(rows), path)
        return str(path)

    def _tokenize(
        self,
        data_path,
        store=None,
        hf_cache=None,
        max_seq_length=512,
        template="tulu",
        with_filter=False,
        fn=TOKENIZE_FN,
    ):
        """get_dataset_v1 with the SFT tokenize transform, in a fresh Hugging Face datasets cache unless `hf_cache`
        is given. Returns the dataset, the tokenize map's output file bytes, the row cache's summaries and its log.

        A fresh cache is always made at the same path and moved away afterwards: datasets fingerprints a loaded file
        by its path, and the fingerprint is part of the map output's metadata."""
        self.runs += 1
        slot = self.root / "hf-slot"
        fresh = hf_cache is None
        hf_cache = slot if fresh else hf_cache
        fns = [fn] + (["sft_tulu_filter_v1"] if with_filter else [])
        fn_args = {} if fn == "sft_tulu_tokenize_without_truncation_v1" else {"max_seq_length": max_seq_length}
        args = [fn_args] + ([{}] if with_filter else [])
        start = len(self.handler.messages)
        if store is not None:
            os.environ[token_row_cache.ROOT_ENV] = str(store)
        try:
            with unittest.mock.patch.object(datasets.config, "HF_DATASETS_CACHE", pathlib.Path(hf_cache)):
                tc = dataset_transformation.TokenizerConfig(
                    tokenizer_name_or_path=TOKENIZER_PATH,
                    tokenizer_revision="main",
                    use_fast=True,
                    chat_template_name=template,
                    add_bos=False,
                )
                dc = dataset_transformation.DatasetConfig(
                    dataset_name=data_path,
                    dataset_split="train",
                    dataset_revision="main",
                    transform_fn=fns,
                    transform_fn_args=args,
                    target_columns=dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
                )
                dataset = dataset_transformation.get_dataset_v1(dc, tc)
        finally:
            os.environ.pop(token_row_cache.ROOT_ENV, None)
        files = [pathlib.Path(f["filename"]).read_bytes() for f in dataset.cache_files]
        if fresh:
            shutil.move(slot, self.root / f"hf-{self.runs}")
        logs = self.handler.messages[start:]
        summaries = [json.loads(m.split("SUMMARY ", 1)[1]) for m in logs if m.startswith("TOK-ROW-CACHE SUMMARY ")]
        return dataset, files, summaries, logs

    def _assert_same(self, ref, got):
        self.assertTrue(got[0].data.table.equals(ref[0].data.table, check_metadata=True))
        self.assertEqual(got[1], ref[1], "tokenize map output files differ")

    def _shards(self, store):
        return sorted(store.rglob("shards/*.arrow"))

    def test_off_unless_the_variable_is_set(self):
        self.assertIsNone(token_row_cache.store_root())
        path = self._write("a", _rows(40, seed=0))
        with unittest.mock.patch.object(token_row_cache, "cached_map") as cached_map:
            self._tokenize(path)
        cached_map.assert_not_called()

    def test_cold_warm_and_partial_builds_match_the_plain_map(self):
        rows = _rows(120, seed=1)
        mix_a = self._write("a", rows)
        swapped, changed = _swap_final_turns(rows, every=4)
        mix_b = self._write("b", swapped)
        ref_a, ref_b = self._tokenize(mix_a), self._tokenize(mix_b)
        distinct = len({json.dumps([r["messages"], r["tools"]]) for r in rows})

        cold = self._tokenize(mix_a, store=self.store)
        self._assert_same(ref_a, cold)
        (summary,) = cold[2]
        self.assertEqual((summary["hits"], summary["tokenized"], summary["fallbacks"]), (0, distinct, 0))
        self.assertEqual(summary["served"], len(rows))
        self.assertTrue(self._shards(self.store))

        warm = self._tokenize(mix_a, store=self.store)
        self._assert_same(ref_a, warm)
        (summary,) = warm[2]
        self.assertEqual((summary["hits"], summary["tokenized"], summary["fallbacks"]), (len(rows), 0, 0))

        partial = self._tokenize(mix_b, store=self.store)
        self._assert_same(ref_b, partial)
        (summary,) = partial[2]
        self.assertEqual((summary["tokenized"], summary["fallbacks"]), (changed, 0))

        leftovers = [
            p for p in self.store.rglob("*") if p.is_file() and p.name.startswith(("tmp", ".probe", "incoming"))
        ]
        self.assertEqual(leftovers, [])

    def test_filter_drops_the_same_rows(self):
        path = self._write("a", _rows(60, seed=2))
        ref = self._tokenize(path, with_filter=True)
        self.assertEqual(len(ref[0]), 60 + 3)  # the no-assistant and long-prompt rows are dropped
        self._tokenize(path, store=self.store, with_filter=True)
        warm = self._tokenize(path, store=self.store, with_filter=True)
        self.assertTrue(warm[0].data.table.equals(ref[0].data.table, check_metadata=True))
        self.assertEqual(warm[2][0]["tokenized"], 0)

    @parameterized.expand([("max_seq_length", {"max_seq_length": 64}), ("template", {"template": "olmo"})])
    def test_a_changed_argument_is_a_new_identity(self, _name, change):
        path = self._write("a", _rows(40, seed=3))
        self._tokenize(path, store=self.store)
        ref = self._tokenize(path, **change)
        got = self._tokenize(path, store=self.store, **change)
        self._assert_same(ref, got)
        self.assertEqual(got[2][0]["hits"], 0)
        self.assertEqual(len([p for p in self.store.iterdir() if p.is_dir()]), 2)

    def test_identity_is_stable_and_covers_the_arguments(self):
        tc = dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH, tokenizer_revision="main", chat_template_name="tulu"
        )
        fn = dataset_transformation.TRANSFORM_FNS[TOKENIZE_FN][0]

        def digest(fn_args, version="v10"):
            kwargs = {"tokenizer": tc.tokenizer, **fn_args}
            return token_row_cache.identity_digest(
                token_row_cache.compute_identity(fn, TOKENIZE_FN, kwargs, tc, version)
            )

        base = dataset_transformation.sft_tokenize_fn_args(512, "keep")
        self.assertEqual(digest(base), digest(dict(base)))
        others = {
            digest(dataset_transformation.sft_tokenize_fn_args(513, "keep")),
            digest(dataset_transformation.sft_tokenize_fn_args(512, "terminate")),
            digest(base, version="v9"),
        }
        self.assertEqual(len(others), 3)
        self.assertNotIn(digest(base), others)

    @parameterized.expand([("last_turn_tulu_tokenize_and_truncate_v1",), ("sft_tulu_tokenize_without_truncation_v1",)])
    def test_other_tokenize_transforms(self, fn):
        path = self._write("a", _rows(60, seed=8))
        ref = self._tokenize(path, fn=fn)
        cold = self._tokenize(path, store=self.store, fn=fn)
        warm = self._tokenize(path, store=self.store, fn=fn)
        self._assert_same(ref, cold)
        self._assert_same(ref, warm)
        self.assertEqual((warm[2][0]["tokenized"], warm[2][0]["fallbacks"]), (0, 0))

    def test_identity_ignores_per_call_truncation_state(self):
        tc = dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH, tokenizer_revision="main", chat_template_name="tulu"
        )
        fn = dataset_transformation.TRANSFORM_FNS[TOKENIZE_FN][0]
        kwargs = {"tokenizer": tc.tokenizer, **dataset_transformation.sft_tokenize_fn_args(512, "keep")}
        before = token_row_cache.identity_digest(token_row_cache.compute_identity(fn, TOKENIZE_FN, kwargs, tc, "v10"))
        tc.tokenizer("some text", truncation=True, max_length=7)
        after = token_row_cache.identity_digest(token_row_cache.compute_identity(fn, TOKENIZE_FN, kwargs, tc, "v10"))
        self.assertEqual(before, after)

    def test_a_shard_gone_after_indexing_is_tokenized(self):
        # As when another job seals (renames) an incoming shard between this job's index and its final map.
        path = self._write("a", _rows(60, seed=9))
        ref = self._tokenize(path)
        self._tokenize(path, store=self.store)
        real_prepare = token_row_cache._prepare

        def prepare_then_remove(*args, **kwargs):
            prep = real_prepare(*args, **kwargs)
            for shard in prep["paths"]:
                os.remove(shard)
            return prep

        with unittest.mock.patch.object(token_row_cache, "_prepare", side_effect=prepare_then_remove):
            got = self._tokenize(path, store=self.store)
        self._assert_same(ref, got)
        (summary,) = got[2]
        self.assertEqual(summary["read_failures"], summary["rows"])
        self.assertEqual(summary["fallbacks"], summary["rows"])

    def test_damaged_store_is_detected_and_repaired(self):
        rows = _rows(120, seed=4)
        path = self._write("a", rows)
        ref = self._tokenize(path)
        self._tokenize(path, store=self.store)
        shards = self._shards(self.store)
        self.assertGreaterEqual(len(shards), 1)
        flipped = _flip_token_bytes(shards[0], rows=3)
        self.assertEqual(flipped, 3)
        (shards[0].parent / ("0" * 64 + ".arrow")).write_bytes(b"not an arrow file")

        damaged = self._tokenize(path, store=self.store)
        self._assert_same(ref, damaged)
        (summary,) = damaged[2]
        # Every row of the mix that maps to a damaged stored row, duplicates included, fails the digest check.
        self.assertGreaterEqual(summary["fallbacks"], flipped)
        self.assertEqual(summary["store_unreadable_shards"], 1)

        # A shard cut short is unreadable; its rows are tokenized again and stored under the same name.
        shards[0].write_bytes(shards[0].read_bytes()[: shards[0].stat().st_size // 2])
        cut = self._tokenize(path, store=self.store)
        self._assert_same(ref, cut)
        self.assertGreater(cut[2][0]["tokenized"], 0)
        again = self._tokenize(path, store=self.store)
        self._assert_same(ref, again)
        self.assertEqual((again[2][0]["tokenized"], again[2][0]["fallbacks"]), (0, 0))

    def test_unreachable_root_runs_the_plain_map(self):
        path = self._write("a", _rows(40, seed=5))
        ref = self._tokenize(path)
        missing = self.root / "not-mounted" / "store"
        got = self._tokenize(path, store=missing)
        self._assert_same(ref, got)
        self.assertEqual(got[2], [])
        self.assertTrue(any("SETUP FAILED" in m for m in got[3]))
        self.assertFalse(missing.parent.exists())

    def test_a_complete_hugging_face_map_cache_is_loaded(self):
        path = self._write("a", _rows(40, seed=6))
        hf_cache = self.root / "shared-hf"
        first = self._tokenize(path, store=self.store, hf_cache=hf_cache)
        second = self._tokenize(path, store=self.store, hf_cache=hf_cache)
        self._assert_same(first, second)
        self.assertEqual(second[2], [])
        self.assertTrue(any("already holds" in m for m in second[3]))

    def test_several_processes(self):
        # get_num_proc gives two processes for at least 800 rows when two CPUs are available.
        rows = _rows(820, seed=7)
        path = self._write("a", rows)
        self._set_env(BEAKER_ASSIGNED_CPU_COUNT="2")
        ref = self._tokenize(path, max_seq_length=128)
        cold = self._tokenize(path, store=self.store, max_seq_length=128)
        warm = self._tokenize(path, store=self.store, max_seq_length=128)
        self._assert_same(ref, cold)
        self._assert_same(ref, warm)
        self.assertEqual(len(ref[1]), 2)
        self.assertEqual(cold[2][0]["miss_procs"], 2)
        self.assertEqual((warm[2][0]["hits"], warm[2][0]["fallbacks"]), (len(rows), 0))


def _flip_token_bytes(path, rows):
    """Change the first 16 token ids of `rows` compact rows inside a shard file; return how many were changed."""
    table = ipc.open_stream(pa.memory_map(str(path))).read_all()
    ids = table.column(token_row_cache.IDS)
    blob = bytearray(path.read_bytes())
    done = 0
    for r in range(table.num_rows):
        values = np.asarray(ids[r].values, dtype=np.uint32)
        if table.column(token_row_cache.MODE)[r].as_py() != 0 or len(values) < 32:
            continue
        pattern = values[:16].tobytes()
        at = blob.find(pattern)
        if at < 0 or blob.find(pattern, at + 1) >= 0:
            continue
        blob[at : at + 64] = (values[:16] ^ np.uint32(1)).tobytes()
        done += 1
        if done == rows:
            break
    path.write_bytes(bytes(blob))
    return done


if __name__ == "__main__":
    unittest.main()
