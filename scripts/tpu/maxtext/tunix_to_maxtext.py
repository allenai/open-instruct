"""Rewrite a Tunix DPO model_params checkpoint in the layout to_maxtext writes, for export_hf.sh.

  tunix_to_maxtext.py <in model_params dir> <out base dir>

Tunix's DPO trainer saves the policy under a top-level "base" key, wraps every leaf as {"value": ...} and keeps the
nnx RNG state ("rngs") beside the parameters. MaxText's to_huggingface reads {"params": {"params": tree}}. Arrays are
copied unchanged. Writes <out base dir>/0/items like to_maxtext.
"""

import sys

import jax
import numpy as np
import orbax.checkpoint as ocp

src, out = sys.argv[1], sys.argv[2]
ckpt = ocp.PyTreeCheckpointer()
meta = ckpt.metadata(src)
meta = meta.item_metadata.tree if hasattr(meta, "item_metadata") else meta.tree
state = ckpt.restore(src, restore_args=jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), meta))
tree = state["base"] if set(state) == {"base"} else state


def clean(d):
    if isinstance(d, dict):
        if set(d) == {"value"}:
            return d["value"]
        return {k: clean(v) for k, v in d.items() if k != "rngs"}
    return d


params = clean(tree)
leaves = jax.tree.leaves(params)
assert leaves and all(isinstance(x, np.ndarray) for x in leaves), "non-array leaf left after unwrapping"
mgr = ocp.CheckpointManager(out, item_names=("items",))
mgr.save(0, args=ocp.args.Composite(items=ocp.args.PyTreeSave({"params": {"params": params}})))
mgr.wait_until_finished()
print(f"{len(leaves)} arrays, {sum(x.size for x in leaves) / 1e9:.3f}B values: wrote {out}/0/items")
