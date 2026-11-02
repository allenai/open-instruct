"""H174 cache guard: move aside (never delete) any transformed-dataset dir under CACHE_ROOT that open-instruct would
load but cannot read.

open-instruct's LocalDatasetTransformationCache loads CACHE_ROOT/<config_hash> whenever the dir exists. A preemption
can leave it unreadable: on 2026-11-02 a dir holding dataset_statistics.json still had an arrow file that pyarrow
rejected ("Tried reading schema message, was null or length 0"). A dir passes only if it has dataset_statistics.json
and state.json, and every arrow file it names opens and reports its row count.

  python h174_cache_guard.py CACHE_ROOT
"""
import json
import os
import shutil
import sys
import time

import pyarrow as pa


def readable(d):
    for need in ("dataset_statistics.json", "state.json"):
        if not os.path.isfile(os.path.join(d, need)):
            return f"missing {need}"
    with open(os.path.join(d, "state.json")) as f:
        files = [x["filename"] for x in json.load(f).get("_data_files", [])]
    if not files:
        return "state.json names no data files"
    rows = 0
    for name in files:
        p = os.path.join(d, name)
        try:
            with pa.memory_map(p) as src:
                rows += pa.ipc.open_stream(src).read_all().num_rows
        except Exception as e:  # noqa: BLE001 - any unreadable file disqualifies the dir
            return f"{name}: {type(e).__name__}: {e}"
    print(f"H174-GUARD-OK {d} files={len(files)} rows={rows}", flush=True)
    return None


def main(root):
    for n in sorted(os.listdir(root)):
        d = os.path.join(root, n)
        if n == "numpy_sft" or not os.path.isdir(d):
            continue
        why = readable(d)
        if why:
            dest = os.path.join(os.path.dirname(root.rstrip("/")), f"cache-partial-{n}-{int(time.time())}")
            print(f"H174-GUARD-MOVED {d} -> {dest}: {why}", flush=True)
            shutil.move(d, dest)


if __name__ == "__main__":
    main(sys.argv[1])
