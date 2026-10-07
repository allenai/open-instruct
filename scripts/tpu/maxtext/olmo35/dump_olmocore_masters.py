"""Stage 1 of the OLMo-core -> MaxText conversion: dump the fp32 master copies (`module.<param>.main`) of an
OLMoDDP distcp checkpoint as flat .npy files. Runs where olmo_core is importable (its metadata is pickled with
olmo_core classes).   dump_olmocore_masters.py <checkpoint step dir> <out dir>"""

import os
import sys

import numpy as np
import torch
import torch.distributed.checkpoint as dcp

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
reader = dcp.FileSystemReader(os.path.join(src, "model_and_optim"))
_read_metadata = reader.read_metadata


def read_metadata(*args, **kwargs):
    # Checkpoints written by an older torch carry _StorageInfo records without `transform_descriptors`, which
    # newer torch's FileSystemReader reads; an absent field means no transforms. dcp.load re-reads the metadata,
    # so the fix has to sit in the reader.
    md = _read_metadata(*args, **kwargs)
    for info in md.storage_data.values():
        if not hasattr(info, "transform_descriptors"):
            object.__setattr__(info, "transform_descriptors", None)
    return md


reader.read_metadata = read_metadata
md = reader.read_metadata()
keys = [k for k in md.state_dict_metadata if k.startswith("module.") and k.endswith(".main")]
state = {k: torch.empty(tuple(md.state_dict_metadata[k].size), dtype=torch.float32) for k in keys}
dcp.load(state, storage_reader=reader)
n = 0
for k, v in state.items():
    np.save(os.path.join(out, k[len("module.") : -len(".main")] + ".npy"), v.numpy())
    n += v.numel()
print(f"dumped {len(state)} tensors, {n / 1e9:.3f}B params to {out}")
