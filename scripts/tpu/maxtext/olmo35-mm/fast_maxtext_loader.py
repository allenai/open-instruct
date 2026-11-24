"""Fast MaxText Orbax params reader (non-OCDBT zarr v2 per-array dirs) for the TPU-eval converter.

From allenai/olmo-post-training-ledger@ca639ab0b01ed7cff57a411d73936ecc9f0d8914 experiments/mmsft/tpu/fast_maxtext_loader.py
(ledger H271's eval pipeline); only the formatting and the import order changed.

Orbax's restore + np.array() on this box copied at ~5 MB/s for a 48 GB training checkpoint (hours). Each
`params.params.<a>.<b>...` dir under `items/` is a plain zarr v2 array, so read them with tensorstore in parallel
and return the same mapping `load_maxtext_params_any` returns: {"a/b/...": float32 torch tensor}.

`patch()` swaps the converter module's loader for this one (falls back to the original if the layout differs).
"""

import concurrent.futures as cf
import os
import runpy
import sys
import time

import numpy as np
import tensorstore as ts
import torch
from maxtext.checkpoint_conversion.standalone_scripts import convert_olmo35_multimodal_ckpt as conv

PREFIX = "params.params."


def load_fast(items_dir: str, workers: int = 48):
    names = sorted(
        d for d in os.listdir(items_dir) if d.startswith(PREFIX) and os.path.isdir(os.path.join(items_dir, d))
    )
    if not names or not all(os.path.exists(os.path.join(items_dir, d, ".zarray")) for d in names):
        raise FileNotFoundError(f"{items_dir}: no zarr v2 params.params.* arrays")
    ctx = ts.Context({"data_copy_concurrency": {"limit": workers}, "file_io_concurrency": {"limit": workers}})

    def read(d):
        arr = ts.open(
            {"driver": "zarr", "kvstore": {"driver": "file", "path": os.path.join(items_dir, d)}},
            context=ctx,
            open=True,
        ).result()
        a = np.ascontiguousarray(arr.read().result(), dtype=np.float32)
        return d[len(PREFIX) :].replace(".", "/"), torch.from_numpy(a)

    t0 = time.time()
    out = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, v in ex.map(read, names):
            out[k] = v
    n = sum(v.numel() for v in out.values())
    print(
        f"[fast_maxtext_loader] {len(out)} arrays, {n / 1e9:.2f}B elements in {time.time() - t0:.0f}s",
        file=sys.stderr,
        flush=True,
    )
    return out


def patch():
    orig = conv.load_maxtext_params_any

    def load(items_dir):
        try:
            return load_fast(items_dir)
        except FileNotFoundError as e:
            print(f"[fast_maxtext_loader] fallback to orbax: {e}", file=sys.stderr, flush=True)
            return orig(items_dir)

    conv.load_maxtext_params_any = load
    return conv


if __name__ == "__main__":
    # Usage: python fast_maxtext_loader.py <converter args...>  == the converter CLI with the fast loader.
    patch()

    sys.argv = ["convert_olmo35_multimodal_checkpoint"] + sys.argv[1:]
    runpy.run_module("maxtext.checkpoint_conversion.convert_olmo35_multimodal_checkpoint", run_name="__main__")
