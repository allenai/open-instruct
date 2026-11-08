"""Disk-backed prompt templates with bounded byte prefetch and compact row IDs."""

import bisect
import hashlib
import json
import os
import random
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from open_instruct.miles.datasets import indexed_rows


class Catalog:
    def __init__(
        self, directory, sources, pools, *, sample_type, max_prompt_length=None, cache_rows=1280, cache_bytes=256 << 20
    ):
        self.sample_type = sample_type
        self.sources, self.pool_names = sources, list(pools)
        self.indexes, self.fds, self.ends = [], [], []
        codes = []
        self.cache_rows, self.cache_bytes = cache_rows, cache_bytes
        self.cache, self.pending = OrderedDict(), {}
        self.bytes = 0
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prompt-prefetch")
        self.reads, self.hits = 0, 0
        try:
            for source in sources:
                path = directory / source["path"]
                index, header = indexed_rows.open_index(path, source["sha256"], source["rows"], max_prompt_length)
                mapping = np.asarray([self.pool_names.index(name) for name in header["pools"]], dtype=np.uint32)
                codes.append(mapping[index["pool"]])
                self.indexes.append(index)
                self.fds.append(os.open(path, os.O_RDONLY))
                self.ends.append((self.ends[-1] if self.ends else 0) + len(index))
            self.pool_codes = np.concatenate(codes)
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def _read(self, row):
        source = bisect.bisect_right(self.ends, row)
        local = row - (self.ends[source - 1] if source else 0)
        entry = self.indexes[source][local]
        raw = os.pread(self.fds[source], int(entry["length"]), int(entry["offset"]))
        if len(raw) != int(entry["length"]):
            raise OSError("Prepared prompt read was incomplete")
        with self.lock:
            self.reads += 1
        return raw

    def _finished(self, row, future):
        with self.lock:
            if self.pending.get(row) is not future:
                return  # A demand read already consumed this future.
            self.pending.pop(row)
            if future.cancelled() or future.exception() is not None:
                return  # Demand reads propagate failures; speculative reads never select rows.
            raw = future.result()
            if len(raw) > self.cache_bytes:
                return
            previous = self.cache.pop(row, None)
            self.bytes -= len(previous) if previous is not None else 0
            self.cache[row] = raw
            self.bytes += len(raw)
            while self.bytes > self.cache_bytes or len(self.cache) > self.cache_rows:
                _, removed = self.cache.popitem(last=False)
                self.bytes -= len(removed)

    def prefetch(self, rows):
        for row in rows:
            row = int(row)
            with self.lock:
                if row in self.cache or row in self.pending:
                    continue
                if len(self.pending) >= self.cache_rows:
                    break
                future = self.executor.submit(self._read, row)
                self.pending[row] = future
            future.add_done_callback(lambda result, row=row: self._finished(row, result))

    def sample(self, row):
        row = int(row)
        with self.lock:
            raw = self.cache.pop(row, None)
            future = self.pending.pop(row, None)
            if raw is not None:
                self.hits += 1
                self.bytes -= len(raw)
        if raw is None:
            raw = future.result() if future is not None else self._read(row)
        data = json.loads(raw)
        source = bisect.bisect_right(self.ends, row)
        metadata = data["metadata"] | {"mixture_source": self.sources[source]["name"]}
        return self.sample_type(prompt=data["input"], label=data.get("label"), metadata=metadata)

    def split(self, policy, shuffle, seed):
        order = np.arange(len(self), dtype=np.uint64)
        if policy == "shuffle" and shuffle:
            random.Random(seed).shuffle(order)
        codes = self.pool_codes[order]
        members = {name: View(self, order[codes == code]) for code, name in enumerate(self.pool_names)}
        return PoolOrder(self.pool_names, codes) if policy == "shuffle" else [], members

    def metrics(self):
        with self.lock:
            return {
                "cache_bytes": self.bytes,
                "cache_rows": len(self.cache),
                "prefetch_pending": len(self.pending),
                "disk_reads": self.reads,
                "cache_hits": self.hits,
            }

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()
        self.indexes.clear()
        self.cache.clear()
        self.pending.clear()
        self.bytes = 0


class PoolOrder(Sequence):
    def __init__(self, names, codes):
        self.names, self.codes = names, codes

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.names[int(self.codes[index])]


class View(Sequence):
    def __init__(self, catalog, rows):
        self.catalog, self.rows = catalog, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.catalog.sample(row) for row in self.rows[index]]
        return self.catalog.sample(self.rows[index])

    def ordered(self, seed, shuffle):
        rows = self.rows.copy()
        if shuffle:
            random.Random(seed).shuffle(rows)
        return View(self.catalog, rows)

    def prefetch(self, offset):
        # Per-pool lookahead shares one byte/row-bounded cache across all pools.
        lookahead = min(64, max(1, self.catalog.cache_rows // len(self.catalog.pool_names)))
        self.catalog.prefetch(self.rows[offset : offset + lookahead])

    def replay_catalog(self, identity):
        entries, sources, tasks = {}, {}, {}
        for row in self.rows:
            key, source, task = identity(self.catalog.sample(row))
            entries.setdefault(key, int(row))
            sources[key], tasks[key] = source, task
        catalog_hash = hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()
        return ReplayRows(self.catalog, entries), sources, tasks, catalog_hash


class ReplayRows(Mapping):
    def __init__(self, catalog, entries):
        self.catalog, self.entries = catalog, entries

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def __getitem__(self, key):
        return self.catalog.sample(self.entries[key])
