"""Row-level cache for the SFT tokenize map of open-instruct 37a119ffd, applied in the container by the launcher.

    python tok_row_cache_patch.py <open-instruct root>

copies this file to open_instruct/token_row_cache.py and routes get_dataset_v1's transform map through
cached_map below. Applied after the sed fix, tok_stats_patch.py, tok_count_patch.py and tok_collect_patch.py; it
touches none of their lines.

Why: a mix that changes the final assistant turn of some rows gives its source parquet a new fingerprint, so the
Hugging Face map cache tokenizes every row of it again (about an hour for the 1.96M-row Dolci parquet at num_proc 90),
although most rows were tokenized in an earlier mix. This cache keys each row by a hash of exactly what the tokenize
function reads (messages and tools), scoped under an identity digest of everything else that decides the output
(tokenizer state and files, chat template, transform name, kwargs and source, library versions). Only rows whose key is
not in the store are tokenized, by the original function, and the result is written to the store.

What stays the same: get_dataset_v1 calls cached_map with the arguments of its original dataset.map call. cached_map
runs one dataset.map with those arguments (same fingerprint, num_proc, remove_columns, so the same cache file names),
whose function sets each row's input_ids, labels and attention_mask from the store, in the order the original sets
them, and calls the original function for any row it cannot serve. The filter, the transformation cache, the
statistics, the numpy cache name and resume are untouched. A served row must pass three checks: its key, recomputed
from the exact row object the original function would receive, equals the planned key; the stored row carries the same
key; and the sha256 of the rebuilt int64 arrays equals the digest taken from the original function's output when the
row was stored. scripts/c12/test_tok_row_cache.py checks that the output is byte-identical to the unpatched pipeline.

Store: $TOK_ROW_CACHE (default below; "off" disables), one directory per identity digest with identity.json and
shards/*.arrow. Shards are written by the miss map itself (Hugging Face writes each to a temporary file in the same
directory and renames it when complete), then renamed to the sha256 of their keys and digests. Nothing is ever
rewritten or deleted, so concurrent jobs and preemption cannot leave a half-written shard under a shard name. Any
error before the final map falls back to the original map; an unreadable shard is skipped; a row that fails a check is
tokenized.
"""

import concurrent.futures
import dataclasses
import glob
import hashlib
import importlib.metadata
import inspect
import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import socket
import struct
import sys
import tempfile
import time
import traceback
import uuid

import numpy as np

FORMAT = "tok-row-cache-v1"
DEFAULT_ROOT = "/weka/olmo-3p5-post-checkpoints/chrisg/token-row-cache"
OFF_VALUES = ("off", "0", "false", "no", "none", "disable", "disabled")
# Transforms that read only row["messages"] and row.get("tools") (through _tokenize_row_or_mask_out) and set only
# input_ids, labels and attention_mask. Any other transform runs the original map.
CACHEABLE_FNS = frozenset(
    {
        "sft_tulu_tokenize_and_truncate_v1",
        "sft_tulu_tokenize_without_truncation_v1",
        "last_turn_tulu_tokenize_and_truncate_v1",
    }
)
MESSAGES_KEY = "messages"
TOOLS_KEY = "tools"
OUT_KEYS = ("input_ids", "labels", "attention_mask")  # the order the original function sets them in
MASKED = -100
KEY = "__trc_key"
DIGEST = "__trc_digest"
MODE = "__trc_mode"  # 0: ids as uint32 plus a label mask (labels are ids or -100, attention all ones); 1: raw int64
IDS = "__trc_ids"
MASK = "__trc_mask"
RAW_IDS = "__trc_raw_ids"
RAW_LABELS = "__trc_raw_labels"
RAW_ATTN = "__trc_raw_attn"
SHARD_COLUMNS = (KEY, DIGEST, MODE, IDS, MASK, RAW_IDS, RAW_LABELS, RAW_ATTN)
_NAMED_SHARD = re.compile(r"[0-9a-f]{64}\.arrow")
# A miss-map output that a preempted job completed but did not rename; complete because Hugging Face renames it
# from a temporary file only when its rank finishes.
_INCOMING_SHARD = re.compile(r"incoming-[0-9a-f]{32}(_\d{5}_of_\d{5})?\.arrow")
_INDEX_THREADS = 32

try:
    from open_instruct import logger_utils

    logger = logger_utils.setup_logger(__name__)
except Exception:  # noqa: BLE001
    logger = logging.getLogger(__name__)


def _say(msg: str, level: int = logging.INFO) -> None:
    logger.log(level, f"TOK-ROW-CACHE {msg}")


# ---------------------------------------------------------------- row keys and output digests


class Uncacheable(Exception):
    """The row holds a value the canonical encoding does not cover; it is tokenized and never stored."""


def _canon(obj, out: list) -> None:
    """Append an injective, type-tagged encoding of obj. Dict order is kept: templates iterate mappings in order."""
    t = type(obj)
    if obj is None:
        out.append(b"N")
    elif t is bool:
        out.append(b"T" if obj else b"F")
    elif t is str:
        b = obj.encode("utf-8", "surrogatepass")
        out.append(b"s%d:" % len(b))
        out.append(b)
    elif t is int:
        b = str(obj).encode()
        out.append(b"i%d:" % len(b))
        out.append(b)
    elif t is float:
        out.append(b"f")
        out.append(struct.pack("<d", obj))
    elif t is list:
        out.append(b"l%d:" % len(obj))
        for x in obj:
            _canon(x, out)
    elif t is dict:
        out.append(b"d%d:" % len(obj))
        for k, v in obj.items():
            if type(k) is not str:
                raise Uncacheable(f"dict key of type {type(k).__name__}")
            kb = k.encode("utf-8", "surrogatepass")
            out.append(b"%d:" % len(kb))
            out.append(kb)
            _canon(v, out)
    elif t is bytes:
        out.append(b"b%d:" % len(obj))
        out.append(obj)
    else:
        raise Uncacheable(f"value of type {t.__module__}.{t.__qualname__}")


def row_key(messages, tools, has_tools: bool) -> bytes:
    """sha256 over everything the cacheable transforms read from a row: messages and, if the column exists, tools."""
    out = [b"trc-row-v1\x00m"]
    _canon(messages, out)
    if has_tools:
        out.append(b"t")
        _canon(tools, out)
    else:
        out.append(b"-")
    return hashlib.sha256(b"".join(out)).digest()


def _row_key_or_empty(row) -> bytes:
    try:
        return row_key(row[MESSAGES_KEY], row[TOOLS_KEY] if TOOLS_KEY in row else None, TOOLS_KEY in row)
    except Uncacheable:
        return b""


def out_digest(ids: np.ndarray, labels: np.ndarray, attn: np.ndarray) -> bytes:
    """sha256 of the three int64 output arrays, as the original function returns them."""
    h = hashlib.sha256(b"trc-out-v1")
    for a in (ids, labels, attn):
        h.update(struct.pack("<q", a.shape[0]))
        h.update(np.ascontiguousarray(a, dtype="<i8").tobytes())
    return h.digest()


def _as_int64_1d(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    if not isinstance(x, np.ndarray) or x.dtype != np.int64 or x.ndim != 1:
        raise TypeError(f"expected a 1-D int64 tensor, got {type(x).__name__} {getattr(x, 'dtype', None)}")
    return x


def _key_batch(messages, tools=None):
    """Batched key pass. With a tools column Hugging Face passes it as the second argument."""
    has_tools = tools is not None
    keys = []
    for i, m in enumerate(messages):
        try:
            keys.append(row_key(m, tools[i] if has_tools else None, has_tools))
        except Uncacheable:
            keys.append(b"")
    return {KEY: keys}


# ---------------------------------------------------------------- identity


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except Exception:  # noqa: BLE001
        return "absent"


def _tokenizer_file_hashes(name_or_path: str, revision) -> dict:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "chat_template.json",
    )
    out = {}
    for name in names:
        path = None
        if os.path.isdir(str(name_or_path)):
            p = os.path.join(name_or_path, name)
            path = p if os.path.isfile(p) else None
        else:
            try:
                from huggingface_hub import try_to_load_from_cache

                p = try_to_load_from_cache(name_or_path, name, revision=revision)
                path = p if isinstance(p, str) and os.path.isfile(p) else None
            except Exception:  # noqa: BLE001
                path = None
        out[name] = _file_sha256(path) if path else "absent"
    return out


def compute_identity(fn, fn_name: str, fn_kwargs: dict, tc, cache_version: str) -> dict:
    """Everything besides the row's messages and tools that can change the tokenize function's output."""
    tok = fn_kwargs["tokenizer"]
    kwargs = {k: v for k, v in fn_kwargs.items() if k != "tokenizer"}
    sources = {}
    for mod_name in (fn.__module__, __name__, "open_instruct.tokenizer_utils"):
        mod = sys.modules.get(mod_name)
        path = inspect.getsourcefile(mod) if mod is not None else None
        sources[mod_name] = _file_sha256(path) if path else "absent"
    tc_dict = {k: v for k, v in dataclasses.asdict(tc).items() if v is not None}
    init_kwargs = {
        k: v
        for k, v in getattr(tok, "init_kwargs", {}).items()
        if k not in ("name_or_path", "tokenizer_file", "vocab_file", "merges_file")
    }
    backend = tok.backend_tokenizer.to_str() if getattr(tok, "is_fast", False) else ""
    return {
        "format": FORMAT,
        "fn_name": fn_name,
        "fn_qualname": f"{fn.__module__}.{fn.__qualname__}",
        "fn_source_sha256": _sha256_bytes(inspect.getsource(fn).encode()),
        "fn_kwargs": json.dumps(kwargs, sort_keys=True, default=repr),
        "module_sources_sha256": sources,
        "dataset_cache_version": cache_version,
        "tokenizer_config": json.dumps(tc_dict, sort_keys=True, default=repr),
        "tokenizer": {
            "class": f"{type(tok).__module__}.{type(tok).__qualname__}",
            "is_fast": bool(getattr(tok, "is_fast", False)),
            "backend_sha256": _sha256_bytes(backend.encode()),
            "chat_template": tok.chat_template if isinstance(tok.chat_template, (str, dict, type(None))) else repr(tok.chat_template),
            "special_tokens_map": json.dumps(tok.special_tokens_map, sort_keys=True, default=repr),
            "added_tokens_decoder": [[int(i), repr(t)] for i, t in sorted(tok.added_tokens_decoder.items())],
            "ids": [tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id],
            "vocab_size": tok.vocab_size,
            "len": len(tok),
            "truncation_side": getattr(tok, "truncation_side", None),
            "padding_side": getattr(tok, "padding_side", None),
            "model_max_length": repr(getattr(tok, "model_max_length", None)),
            "init_kwargs": json.dumps(init_kwargs, sort_keys=True, default=repr),
            "promoted_reserved_slot_tokens": repr(getattr(tok, "promoted_reserved_slot_tokens", None)),
            "files_sha256": _tokenizer_file_hashes(tc.tokenizer_name_or_path, tc.tokenizer_revision),
        },
        "versions": {
            "python": platform.python_version(),
            **{d: _version(d) for d in ("datasets", "transformers", "tokenizers", "torch", "jinja2", "pyarrow", "numpy")},
        },
    }


def identity_digest(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- store


def _shard_features():
    import datasets

    lst = getattr(datasets, "List", None) or datasets.Sequence
    return datasets.Features(
        {
            KEY: datasets.Value("binary"),
            DIGEST: datasets.Value("binary"),
            MODE: datasets.Value("int8"),
            IDS: lst(datasets.Value("uint32")),
            MASK: lst(datasets.Value("uint8")),
            RAW_IDS: lst(datasets.Value("int64")),
            RAW_LABELS: lst(datasets.Value("int64")),
            RAW_ATTN: lst(datasets.Value("int64")),
        }
    )


def _shard_schema_ok(schema) -> bool:
    import pyarrow as pa

    want = {
        KEY: pa.binary(),
        DIGEST: pa.binary(),
        MODE: pa.int8(),
        IDS: pa.list_(pa.uint32()),
        MASK: pa.list_(pa.uint8()),
        RAW_IDS: pa.list_(pa.int64()),
        RAW_LABELS: pa.list_(pa.int64()),
        RAW_ATTN: pa.list_(pa.int64()),
    }
    if list(schema.names) != list(SHARD_COLUMNS):
        return False
    return all(schema.field(n).type.equals(t) for n, t in want.items())


def _open_shard(path: str):
    """The shard as a zero-copy Arrow table over a memory map (Hugging Face writes IPC streams)."""
    import pyarrow as pa

    table = pa.ipc.open_stream(pa.memory_map(path, "r")).read_all()
    if not _shard_schema_ok(table.schema):
        raise ValueError(f"unexpected shard schema {table.schema}")
    return table


def _mat32(arr) -> np.ndarray:
    """An Arrow binary array whose values are all 32 bytes long, as an (n, 32) uint8 matrix."""
    import pyarrow as pa

    fixed = arr.cast(pa.binary(32))
    data = np.frombuffer(fixed.buffers()[1], dtype=np.uint8)
    return data[fixed.offset * 32 : (fixed.offset + len(fixed)) * 32].reshape(-1, 32)


def _list_shards(shards_dir: str) -> list[str]:
    names = sorted(n for n in os.listdir(shards_dir) if _NAMED_SHARD.fullmatch(n) or _INCOMING_SHARD.fullmatch(n))
    return [os.path.join(shards_dir, n) for n in names]


def _index_one(path: str):
    import pyarrow as pa

    table = _open_shard(path)
    keys = table.column(KEY).combine_chunks() if table.num_rows else pa.array([], pa.binary())
    digests = table.column(DIGEST).combine_chunks() if table.num_rows else pa.array([], pa.binary())
    meta = {}
    side = os.path.splitext(path)[0] + ".json"
    if os.path.exists(side):
        try:
            with open(side) as f:
                meta = json.load(f)
        except Exception:  # noqa: BLE001
            meta = {}
    return keys, digests, os.path.getsize(path), meta


class StoreIndex:
    """Every readable shard's keys, deduplicated. A key stored twice with different digests is never served."""

    def __init__(self, shards_dir: str):
        import pyarrow as pa
        import pyarrow.compute as pc

        self.paths: list[str] = []
        self.bytes = 0
        self.rates: list[float] = []
        self.skipped: list[str] = []
        key_parts, digest_parts, file_parts, row_parts = [], [], [], []
        paths = _list_shards(shards_dir)
        with concurrent.futures.ThreadPoolExecutor(_INDEX_THREADS) as pool:
            results = list(pool.map(self._safe_index, paths))
        for path, res in zip(paths, results):
            if isinstance(res, Exception):
                self.skipped.append(path)
                _say(f"UNREADABLE SHARD skipped, its rows will be tokenized: {path}: {res!r}", logging.WARNING)
                continue
            keys, digests, size, meta = res
            fno = len(self.paths)
            self.paths.append(path)
            self.bytes += size
            if meta.get("proc_seconds_per_row"):
                self.rates.append(float(meta["proc_seconds_per_row"]))
            n = len(keys)
            key_parts.append(keys)
            digest_parts.append(digests)
            file_parts.append(np.full(n, fno, dtype=np.int32))
            row_parts.append(np.arange(n, dtype=np.int64))
        keys = pa.concat_arrays(key_parts) if key_parts else pa.array([], pa.binary())
        digests = pa.concat_arrays(digest_parts) if digest_parts else pa.array([], pa.binary())
        file_no = np.concatenate(file_parts) if file_parts else np.zeros(0, np.int32)
        row_no = np.concatenate(row_parts) if row_parts else np.zeros(0, np.int64)
        self.rows = len(keys)
        # Rows with no key (uncacheable input) or no digest are never served.
        lens = pc.binary_length(keys).to_numpy(zero_copy_only=False) if len(keys) else np.zeros(0, np.int64)
        dlens = pc.binary_length(digests).to_numpy(zero_copy_only=False) if len(keys) else np.zeros(0, np.int64)
        usable = np.flatnonzero((lens == 32) & (dlens == 32))
        keys, digests, file_no, row_no = keys.take(usable), digests.take(usable), file_no[usable], row_no[usable]
        self.unique_keys = pc.unique(keys) if len(keys) else pa.array([], pa.binary())
        group = pc.index_in(keys, value_set=self.unique_keys).to_numpy(zero_copy_only=False) if len(keys) else np.zeros(0, np.int64)
        _, first = np.unique(group, return_index=True)  # first stored row of each key, in file order
        self.first_file = file_no[first]
        self.first_row = row_no[first]
        # Conflicts: the same key with a different digest elsewhere (an impure transform or a corrupt shard).
        self.conflicts = 0
        if len(keys) > len(first):
            d = _mat32(digests)
            dup = np.flatnonzero(np.arange(len(keys)) != first[group])
            differs = (d[dup] != d[first[group[dup]]]).any(axis=1)
            bad = np.unique(group[dup[differs]])
            if bad.size:
                self.conflicts = int(bad.size)
                _say(f"{bad.size} KEYS STORED WITH DIFFERENT DIGESTS; they will be tokenized", logging.WARNING)
                valid = np.ones(len(first), bool)
                valid[bad] = False
                self.first_file = np.where(valid, self.first_file, -1)
        self.unique = len(self.unique_keys)

    @staticmethod
    def _safe_index(path):
        try:
            return _index_one(path)
        except Exception as e:  # noqa: BLE001
            return e


def _write_json_atomic(path: str, obj, fsync: bool = True) -> None:
    tmp = f"{path}.tmp-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)


def _prepare_store(root: str, identity: dict, digest: str) -> str:
    parent = os.path.dirname(os.path.abspath(root))
    if not os.path.isdir(parent):
        raise FileNotFoundError(f"cache root's parent {parent} does not exist (is the bucket mounted?)")
    id_dir = os.path.join(root, digest)
    os.makedirs(os.path.join(id_dir, "shards"), exist_ok=True)
    ident_path = os.path.join(id_dir, "identity.json")
    if os.path.exists(ident_path):
        with open(ident_path) as f:
            existing = json.load(f)
        if existing != json.loads(json.dumps(identity)):
            raise RuntimeError(f"{ident_path} does not match this identity although its digest does")
    else:
        _write_json_atomic(ident_path, identity)
    return id_dir


def _store_writable(shards_dir: str) -> bool:
    probe = os.path.join(shards_dir, f".probe-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with open(probe, "wb") as f:
            f.write(b"x")
        os.remove(probe)
        return True
    except OSError as e:
        _say(f"store {shards_dir} is not writable ({e!r}); serving hits only, new rows are not stored", logging.WARNING)
        return False


def _content_name(path: str) -> tuple[str, int]:
    """The sha256 of a shard's keys and digests, and its row count. Raises if a message body is cut short."""
    table = _open_shard(path)
    h = hashlib.sha256(b"trc-shard-v1")
    for col in (KEY, DIGEST):
        for v in table.column(col).to_pylist():
            h.update(struct.pack("<q", len(v)))
            h.update(v)
    return h.hexdigest(), table.num_rows


def _seal_shard(path: str, extra_meta: dict) -> str:
    """Rename a completed miss-map output to the sha256 of its keys and digests, with a json beside it."""
    digest, rows = _content_name(path)
    name = os.path.join(os.path.dirname(path), digest + ".arrow")
    meta = {**extra_meta, "rows": rows, "created_from": os.path.basename(path)}
    _write_json_atomic(os.path.splitext(name)[0] + ".json", meta)
    try:
        intact = os.path.exists(name) and _content_name(name)[0] == digest
    except Exception:  # noqa: BLE001
        intact = False
    if intact:  # another job stored exactly these rows
        os.remove(path)
    else:  # absent, or a damaged file under this name: replace it (atomic; open readers keep the old file)
        if os.path.exists(name):
            _say(f"replacing the damaged shard {name}", logging.WARNING)
        os.replace(path, name)
    return name


# ---------------------------------------------------------------- map functions


class MissFn:
    """Runs the original transform on a row and returns what the store keeps: key, digest and the outputs."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, row, **fn_kwargs):
        key = _row_key_or_empty(row)
        out = self.fn(row, **fn_kwargs)
        if out is not row:
            raise RuntimeError(f"{self.fn.__name__} returned a new object; the row cache assumes it updates the row")
        ids, labels, attn = (_as_int64_1d(row[k]) for k in OUT_KEYS)
        empty = np.zeros(0, np.int64)
        res = {KEY: key, DIGEST: out_digest(ids, labels, attn)}
        n = ids.shape[0]
        compact = (
            labels.shape[0] == n
            and attn.shape[0] == n
            and bool((attn == 1).all())
            and (n == 0 or (int(ids.min()) >= 0 and int(ids.max()) <= 0xFFFFFFFF))
            and bool(((labels == MASKED) | (labels == ids)).all())
        )
        if compact:
            res.update({MODE: 0, IDS: ids.astype(np.uint32), MASK: (labels != MASKED).astype(np.uint8)})
            res.update({RAW_IDS: empty, RAW_LABELS: empty, RAW_ATTN: empty})
        else:
            res.update({MODE: 1, IDS: np.zeros(0, np.uint32), MASK: np.zeros(0, np.uint8)})
            res.update({RAW_IDS: ids, RAW_LABELS: labels, RAW_ATTN: attn})
        for k in OUT_KEYS:
            del row[k]
        row.update(res)
        return row


class _ShardReader:
    """Random row access to shards, with per-chunk numpy views over the memory maps."""

    def __init__(self, paths):
        self.paths = paths
        self.tables = {}
        self.starts = {}
        self.views = {}

    def row(self, f: int, r: int):
        t = self.tables.get(f)
        if t is None:
            t = self.tables[f] = _open_shard(self.paths[f])
            self.starts[f] = np.cumsum([0] + [len(c) for c in t.column(KEY).chunks])
        starts = self.starts[f]
        c = int(np.searchsorted(starts, r, side="right")) - 1
        lr = r - int(starts[c])
        v = self.views.get((f, c))
        if v is None:
            v = self.views[(f, c)] = {
                name: (t.column(name).chunk(c), *self._list_view(t.column(name).chunk(c)))
                for name in (IDS, MASK, RAW_IDS, RAW_LABELS, RAW_ATTN)
            }
            v[KEY] = t.column(KEY).chunk(c)
            v[DIGEST] = t.column(DIGEST).chunk(c)
            v[MODE] = t.column(MODE).chunk(c).to_numpy(zero_copy_only=False)

        def get(name):
            _, offs, vals = v[name]
            return vals[offs[lr] : offs[lr + 1]]

        key = v[KEY][lr].as_py()
        digest = v[DIGEST][lr].as_py()
        mode = int(v[MODE][lr])
        if mode == 0:
            ids = get(IDS).astype(np.int64)
            mask = get(MASK)
            labels = np.where(mask != 0, ids, np.int64(MASKED))
            attn = np.ones(ids.shape[0], dtype=np.int64)
        elif mode == 1:
            ids, labels, attn = (np.array(get(n), dtype=np.int64) for n in (RAW_IDS, RAW_LABELS, RAW_ATTN))
        else:
            raise ValueError(f"bad mode {mode}")
        return key, digest, ids, labels, attn

    @staticmethod
    def _list_view(arr):
        if arr.null_count:
            raise ValueError("null list in shard")
        return arr.offsets.to_numpy(), arr.values.to_numpy(zero_copy_only=False)


class AssembleFn:
    """The function of the final map: serves each row from the store, or calls the original transform."""

    def __init__(self, fn, plan_dir: str, paths: list[str], num_rows: int, num_proc: int):
        self.fn = fn
        self.plan_dir = plan_dir
        self.paths = paths
        self.num_rows = num_rows
        self.num_proc = max(1, num_proc or 1)
        self._state = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_state"] = None
        return d

    def _load(self):
        p = self.plan_dir
        n, k = self.num_rows, self.num_proc
        div, mod = divmod(n, k)
        ends = np.array([div * i + min(i, mod) + div + (1 if i < mod else 0) for i in range(k)])
        self._state = {
            "file": np.load(os.path.join(p, "file.npy"), mmap_mode="r"),
            "row": np.load(os.path.join(p, "row.npy"), mmap_mode="r"),
            "key": np.load(os.path.join(p, "key.npy"), mmap_mode="r"),
            "has_key": np.load(os.path.join(p, "has_key.npy"), mmap_mode="r"),
            "reader": _ShardReader(self.paths),
            "ends": ends,
            "stats": {},
            "warned": 0,
        }

    def _count(self, idx: int, what: str) -> None:
        s = self._state
        rank = int(np.searchsorted(s["ends"], idx, side="right"))
        st = s["stats"].setdefault(rank, {"served": 0, "fallback": 0, "seen": 0})
        st[what] += 1
        st["seen"] += 1
        first_fallbacks = what == "fallback" and st["fallback"] <= 16
        if first_fallbacks or st["seen"] % 1000 == 0 or idx == int(s["ends"][min(rank, len(s["ends"]) - 1)]) - 1:
            try:
                _write_json_atomic(os.path.join(self.plan_dir, f"stats-{rank}-{os.getpid()}.json"), st, fsync=False)
            except Exception:  # noqa: BLE001
                pass

    def _serve(self, row, idx: int) -> bool:
        s = self._state
        f = int(s["file"][idx])
        if f < 0:
            return False
        skey, sdigest, ids, labels, attn = s["reader"].row(f, int(s["row"][idx]))
        if s["has_key"][idx]:
            key = _row_key_or_empty(row)
            if len(key) != 32 or key != bytes(s["key"][idx]) or key != skey:
                return False
        elif skey != b"":
            return False
        if len(sdigest) != 32 or out_digest(ids, labels, attn) != sdigest:
            return False
        row[OUT_KEYS[0]] = ids
        row[OUT_KEYS[1]] = labels
        row[OUT_KEYS[2]] = attn
        return True

    def __call__(self, row, idx, **fn_kwargs):
        if self._state is None:
            self._load()
        try:
            served = self._serve(row, idx)
        except Exception as e:  # noqa: BLE001
            served = False
            if self._state["warned"] < 5:
                self._state["warned"] += 1
                _say(f"row {idx}: store read failed ({e!r}); tokenizing it", logging.WARNING)
        if served:
            self._count(idx, "served")
            return row
        self._count(idx, "fallback")
        return self.fn(row, **fn_kwargs)


# ---------------------------------------------------------------- the map


def _fp(*parts) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()[:32]


def _hf_map_cache_complete(dataset, new_fingerprint: str) -> bool:
    """Whether dataset.map(new_fingerprint=...) would load a complete set of cache files instead of computing."""
    from datasets.fingerprint import is_caching_enabled

    if not is_caching_enabled() or not dataset.cache_files:
        return False
    path = dataset._get_cache_file_path(new_fingerprint)
    if os.path.exists(path):
        return True
    prefix, ext = os.path.splitext(path)
    pat = re.compile(re.escape(prefix) + r"_(\d{5})_of_(\d{5})" + re.escape(ext))
    found: dict[int, set] = {}
    for f in glob.glob(glob.escape(prefix) + "_*_of_*" + glob.escape(ext)):
        m = pat.fullmatch(f)
        if m:
            found.setdefault(int(m.group(2)), set()).add(int(m.group(1)))
    return any(len(ranks) == n for n, ranks in found.items())


def _resolve_root():
    setting = os.environ.get("TOK_ROW_CACHE")
    if setting is None:
        return DEFAULT_ROOT
    setting = setting.strip()
    if not setting or setting.lower() in OFF_VALUES:
        return None
    return setting


def cached_map(
    dataset,
    fn,
    *,
    fn_name: str,
    tc,
    cache_version: str,
    available_cpus: int,
    fn_kwargs: dict,
    remove_columns: list,
    num_proc,
    new_fingerprint: str,
):
    """dataset.map(fn, ...) with the given arguments, serving rows tokenized before from the row cache."""

    def original():
        return dataset.map(
            fn, fn_kwargs=fn_kwargs, remove_columns=remove_columns, num_proc=num_proc, new_fingerprint=new_fingerprint
        )

    root = _resolve_root()
    if root is None:
        _say(f"disabled by TOK_ROW_CACHE={os.environ.get('TOK_ROW_CACHE')!r}; running the original map")
        return original()
    if fn_name not in CACHEABLE_FNS or MESSAGES_KEY not in dataset.column_names:
        return original()
    if "strftime_now" in str(getattr(fn_kwargs.get("tokenizer"), "chat_template", "")):
        _say("the chat template reads the clock, so its output is not a function of the row; running the original map")
        return original()
    if _hf_map_cache_complete(dataset, new_fingerprint):
        _say(f"the Hugging Face map cache already holds {new_fingerprint}; loading it")
        return original()
    t0 = time.perf_counter()
    plan_dir = None
    try:
        prep = _prepare(dataset, fn, fn_name, tc, cache_version, available_cpus, fn_kwargs, root, new_fingerprint)
        plan_dir = prep["plan_dir"]
    except Exception:  # noqa: BLE001
        _say("SETUP FAILED; running the original map:\n" + traceback.format_exc(), logging.WARNING)
        if plan_dir:
            shutil.rmtree(plan_dir, ignore_errors=True)
        return original()
    t1 = time.perf_counter()
    try:
        result = dataset.map(
            AssembleFn(fn, plan_dir, prep["paths"], len(dataset), num_proc),
            fn_kwargs=fn_kwargs,
            remove_columns=remove_columns,
            num_proc=num_proc,
            new_fingerprint=new_fingerprint,
            with_indices=True,
        )
    except Exception:  # noqa: BLE001
        _say("ASSEMBLY FAILED; running the original map:\n" + traceback.format_exc(), logging.WARNING)
        shutil.rmtree(plan_dir, ignore_errors=True)
        return original()
    t2 = time.perf_counter()
    stats = {"served": 0, "fallback": 0}
    for p in glob.glob(os.path.join(plan_dir, "stats-*.json")):
        try:
            with open(p) as f:
                st = json.load(f)
            stats["served"] += st["served"]
            stats["fallback"] += st["fallback"]
        except Exception:  # noqa: BLE001
            pass
    shutil.rmtree(plan_dir, ignore_errors=True)
    s = prep["summary"]
    s.update(served=stats["served"], fallbacks=stats["fallback"], prep_s=round(t1 - t0, 2), assemble_s=round(t2 - t1, 2))
    rate = s.get("proc_s_per_row")
    if rate:
        saved_rows = s["rows"] - s["tokenized"] - stats["fallback"]
        s["est_saved_s"] = round(max(0, saved_rows) * rate / max(1, num_proc or 1), 1)
    if stats["fallback"]:
        _say(f"{stats['fallback']} rows failed a check and were tokenized instead", logging.WARNING)
    _say(
        f"{fn_name}: {s['rows']:,} rows, {s['hits']:,} from the store, {s['tokenized']:,} tokenized "
        f"({s['miss_rows']:,} miss rows), {stats['fallback']:,} fallbacks; store {s['store_shards']} shards, "
        f"{s['store_rows']:,} rows, {s['store_bytes'] / 1e9:.2f} GB; estimated time saved "
        f"{s.get('est_saved_s', 'unknown')} s"
    )
    _say("SUMMARY " + json.dumps(s, sort_keys=True))
    return result


def _prepare(dataset, fn, fn_name, tc, cache_version, available_cpus, fn_kwargs, root, new_fingerprint) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc

    from open_instruct import dataset_transformation as dt

    t0 = time.perf_counter()
    identity = compute_identity(fn, fn_name, fn_kwargs, tc, cache_version)
    digest = identity_digest(identity)
    id_dir = _prepare_store(root, identity, digest)
    shards_dir = os.path.join(id_dir, "shards")
    index = StoreIndex(shards_dir)
    _say(
        f"identity {digest[:16]} ({fn_name}, {fn_kwargs.get('max_seq_length', '-')}); store {shards_dir}: "
        f"{len(index.paths)} shards, {index.rows:,} rows, {index.unique:,} keys, {index.bytes / 1e9:.2f} GB"
        + (f", {len(index.skipped)} unreadable" if index.skipped else "")
    )
    t_index = time.perf_counter()

    n = len(dataset)
    has_tools = TOOLS_KEY in dataset.column_names
    cols = [MESSAGES_KEY] + ([TOOLS_KEY] if has_tools else [])
    keys_ds = dataset.map(
        _key_batch,
        batched=True,
        input_columns=cols,
        remove_columns=dataset.column_names,
        num_proc=num_proc_for(n, available_cpus, dt),
        keep_in_memory=True,
        new_fingerprint=_fp("trc-keys", dataset._fingerprint, digest),
        desc="Row cache keys",
    )
    qkeys = keys_ds.data.column(KEY).combine_chunks().cast(pa.binary())
    assert len(qkeys) == n
    del keys_ds
    has_key = pc.binary_length(qkeys).to_numpy(zero_copy_only=False) == 32
    t_keys = time.perf_counter()

    # Hits: rows whose key is stored (and not in conflict).
    file_no = np.full(n, -1, dtype=np.int32)
    row_no = np.zeros(n, dtype=np.int64)
    if index.unique:
        pos = pc.index_in(qkeys, value_set=index.unique_keys)
        hit_pos = pos.to_numpy(zero_copy_only=False) if pos.null_count == 0 else pc.fill_null(pos, -1).to_numpy()
        hit = has_key & (hit_pos >= 0)
        hp = hit_pos[hit]
        file_no[hit] = index.first_file[hp]
        row_no[hit] = index.first_row[hp]
        hit &= file_no >= 0
    else:
        hit = np.zeros(n, dtype=bool)
    miss = ~hit
    paths = list(index.paths)

    # Misses: tokenize each distinct key once, and every keyless row on its own.
    miss_keyed = np.flatnonzero(miss & has_key)
    miss_keyless = np.flatnonzero(miss & ~has_key)
    if miss_keyed.size:
        mk = qkeys.take(pa.array(miss_keyed))
        uniq = pc.unique(mk)
        group = pc.index_in(mk, value_set=uniq).to_numpy(zero_copy_only=False)
        _, first = np.unique(group, return_index=True)
        first_pos = miss_keyed[first]
    else:
        group = np.zeros(0, np.int64)
        first_pos = np.zeros(0, np.int64)
    todo = np.sort(np.concatenate([first_pos, miss_keyless]).astype(np.int64))
    summary = {
        "fn": fn_name,
        "identity": digest[:16],
        "rows": n,
        "hits": int(hit.sum()),
        "miss_rows": int(miss.sum()),
        "keyless_rows": int((~has_key).sum()),
        "tokenized": int(todo.size),
        "store_shards": len(index.paths),
        "store_rows": index.rows,
        "store_bytes": index.bytes,
        "store_unreadable_shards": len(index.skipped),
        "store_conflicts": index.conflicts,
        "index_s": round(t_index - t0, 2),
        "keys_s": round(t_keys - t_index, 2),
    }
    if index.rates:
        summary["proc_s_per_row"] = float(np.median(index.rates))

    if todo.size:
        writable = _store_writable(shards_dir)
        subset = dataset.select_columns(cols).select(todo)
        miss_procs = num_proc_for(len(subset), available_cpus, dt)
        tag = uuid.uuid4().hex
        t_m0 = time.perf_counter()
        miss_ds = subset.map(
            MissFn(fn),
            fn_kwargs=fn_kwargs,
            remove_columns=cols,
            num_proc=miss_procs,
            features=_shard_features(),
            new_fingerprint=_fp("trc-miss", new_fingerprint, digest, tag),
            cache_file_name=os.path.join(shards_dir, f"incoming-{tag}.arrow") if writable else None,
            desc="Row cache: tokenizing new rows",
        )
        miss_s = time.perf_counter() - t_m0
        rate = miss_s * (miss_procs or 1) / todo.size
        summary.update(miss_map_s=round(miss_s, 2), miss_procs=miss_procs, proc_s_per_row=rate)
        # The output files in rank order: in the store when it is writable, else in the Hugging Face cache.
        miss_files = [c["filename"] for c in miss_ds.cache_files]
        counts = [_open_shard(p).num_rows for p in miss_files]
        if sum(counts) != todo.size or len(miss_ds) != todo.size:
            raise RuntimeError(f"miss map output has {sum(counts)} rows in {len(miss_files)} files for {todo.size}")
        del miss_ds
        meta = {
            "proc_seconds_per_row": rate,
            "host": socket.gethostname(),
            "beaker_job": os.environ.get("BEAKER_JOB_ID"),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        base = len(paths)
        for p in miss_files:
            if writable:
                try:
                    p = _seal_shard(p, meta)
                except Exception as e:  # noqa: BLE001
                    _say(f"could not seal {p} ({e!r}); it stays as an incoming shard", logging.WARNING)
            paths.append(p)
        # Miss row j of the output is the tokenization of position todo[j]; find its file from the per-file counts.
        starts = np.cumsum([0] + counts)
        all_j = np.arange(todo.size)
        f_of_j = (np.searchsorted(starts, all_j, side="right") - 1).astype(np.int32)
        r_of_j = all_j - starts[f_of_j]
        if miss_keyed.size:
            j = np.searchsorted(todo, first_pos[group])
            file_no[miss_keyed] = base + f_of_j[j]
            row_no[miss_keyed] = r_of_j[j]
        if miss_keyless.size:
            j = np.searchsorted(todo, miss_keyless)
            file_no[miss_keyless] = base + f_of_j[j]
            row_no[miss_keyless] = r_of_j[j]
        summary["stored_rows"] = int(todo.size) if writable else 0

    plan_dir = tempfile.mkdtemp(prefix="trc-plan-")
    key_mat = _mat32(pc.if_else(pa.array(has_key), qkeys, pa.scalar(bytes(32), pa.binary())))
    np.save(os.path.join(plan_dir, "file.npy"), file_no)
    np.save(os.path.join(plan_dir, "row.npy"), row_no)
    np.save(os.path.join(plan_dir, "key.npy"), key_mat)
    np.save(os.path.join(plan_dir, "has_key.npy"), has_key)
    return {"plan_dir": plan_dir, "paths": paths, "summary": summary}


def num_proc_for(n: int, available_cpus: int, dt) -> int:
    return dt.get_num_proc(n, available_cpus, dt.APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU)


# ---------------------------------------------------------------- the patch


OLD_MAP = """        if fn_type == "map":
            dataset = dataset.map(
                fn,
                fn_kwargs=fn_kwargs,
                remove_columns=[col for col in dataset.column_names if col not in target_columns],
                num_proc=get_num_proc(len(dataset), num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
                new_fingerprint=new_fingerprint,
            )
"""
NEW_MAP = """        if fn_type == "map":
            from open_instruct import token_row_cache  # scripts/c12/tok_row_cache_patch.py

            dataset = token_row_cache.cached_map(
                dataset,
                fn,
                fn_name=fn_name,
                tc=tc,
                cache_version=DATASET_CACHE_VERSION,
                available_cpus=num_proc,
                fn_kwargs=fn_kwargs,
                remove_columns=[col for col in dataset.column_names if col not in target_columns],
                num_proc=get_num_proc(len(dataset), num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
                new_fingerprint=new_fingerprint,
            )
"""


def apply_patch(root: str) -> None:
    import pathlib

    root_p = pathlib.Path(root)
    p = root_p / "open_instruct/dataset_transformation.py"
    s = p.read_text()
    assert "token_row_cache" not in s, "row cache patch already applied"
    assert s.count(OLD_MAP) == 1, "get_dataset_v1's transform map does not match 37a119ffd"
    (root_p / "open_instruct/token_row_cache.py").write_text(pathlib.Path(__file__).read_text())
    p.write_text(s.replace(OLD_MAP, NEW_MAP))
    print("ROWCACHE-PATCHED", p)


if __name__ == "__main__":
    apply_patch(sys.argv[1])
