"""Stage 1 for a multimodal checkpoint (H134): dump only the planned LM fp32 masters (`module.<param>.main`) of an
OLMoDDP distcp checkpoint as flat .npy files, one key at a time so memory stays at one tensor.

  dump_olmocore_masters_mm.py <checkpoint step dir> <plan json from strip_vision.py plan> <out dir>

Same reader fix as dump_olmocore_masters.py (absent `transform_descriptors` means no transforms). The metadata is read
with strip_vision's permissive unpickler, so classes from an uninstalled vision codebase don't stop the load. Writes
<out dir>.tmp and renames it when every key is dumped; never overwrites."""
import json
import os
import sys

import numpy as np
import torch
import torch.distributed.checkpoint as dcp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strip_vision import read_dcp_metadata  # noqa: E402

src, plan_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
plan = json.load(open(plan_path))
if os.path.exists(out) or os.path.exists(out + ".tmp"):
    sys.exit(f"{out} or {out}.tmp exists; not overwriting")
md, md_sha, _ = read_dcp_metadata(src)
if md_sha != plan["dcp_metadata_sha256"]:
    sys.exit(f"DCP metadata sha {md_sha[:16]} != plan's {plan['dcp_metadata_sha256'][:16]}")
for info in md.storage_data.values():
    try:
        info.transform_descriptors
    except AttributeError:
        object.__setattr__(info, "transform_descriptors", None)
reader = dcp.FileSystemReader(os.path.join(src, "model_and_optim"))
reader.read_metadata = lambda *a, **k: md
tmp = out + ".tmp"
os.makedirs(tmp)
n = 0
for name, info in sorted(plan["kept"].items()):
    k = info["src_key"]
    t = torch.empty(tuple(md.state_dict_metadata[k].size), dtype=torch.float32)
    dcp.load({k: t}, storage_reader=reader)
    np.save(os.path.join(tmp, k[len("module.") : -len(".main")] + ".npy"), t.numpy())
    n += t.numel()
os.rename(tmp, out)
print(f"dumped {len(plan['kept'])} tensors, {n / 1e9:.3f}B params to {out}")
