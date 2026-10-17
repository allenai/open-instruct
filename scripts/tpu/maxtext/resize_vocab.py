"""Pad or trim the vocabulary of a MaxText Olmo 3 parameter checkpoint (Orbax, as written by to_maxtext).

  resize_vocab.py <in params dir (…/0/items)> <out base dir> <new vocab size>

Olmo 3's vocabulary (100278) is not divisible by tensor parallelism; OLMo-core trains it padded to 100352.
Padding appends zero rows to the token embedding and zero columns to the output projection (padded token
ids never appear in the data, and their logits start at exactly 0). Trimming drops them again before HF
export. Writes <out base dir>/0/items like to_maxtext.
"""

import sys
from pathlib import Path

import jax
import numpy as np
from orbax import checkpoint as ocp

src, out, n = sys.argv[1], str(Path(sys.argv[2]).resolve()), int(sys.argv[3])
if n <= 0:
    raise ValueError("new vocab size must be positive")
ckpt = ocp.PyTreeCheckpointer()
meta = ckpt.metadata(src)
meta = meta.item_metadata.tree if hasattr(meta, "item_metadata") else meta.tree
# Restore every array as host numpy; no device sharding is needed for a reshape on CPU.
state = ckpt.restore(src, restore_args=jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta))
p = state["params"]["params"] if "params" in state else state


def resize(x, axis):
    # Tunix wraps parameter arrays in {"value": ...}; retain that checkpoint layout.
    if isinstance(x, dict) and set(x) == {"value"}:
        return {"value": resize(x["value"], axis)}
    cur = x.shape[axis]
    if n <= cur:
        return np.take(x, np.arange(n), axis=axis)
    pad = [(0, 0)] * x.ndim
    pad[axis] = (0, n - cur)
    return np.pad(x, pad)


embedding = p["token_embedder"]["embedding"]
old = (embedding["value"] if isinstance(embedding, dict) else embedding).shape[0]
p["token_embedder"]["embedding"] = resize(p["token_embedder"]["embedding"], 0)
p["decoder"]["logits_dense"]["kernel"] = resize(p["decoder"]["logits_dense"]["kernel"], 1)
mgr = ocp.CheckpointManager(out, item_names=("items",))
mgr.save(0, args=ocp.args.Composite(items=ocp.args.PyTreeSave(state)))
mgr.wait_until_finished()
print(f"vocab {old} -> {n}: wrote {out}/0/items")
