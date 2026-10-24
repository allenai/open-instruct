from pathlib import Path

import jax
import numpy as np
from orbax import checkpoint as ocp

code = Path("/probe/convert_bounded.py").read_text().split("source_path =")[0]
namespace = {}
exec(compile(code, "/probe/convert_bounded.py", "exec"), namespace)
tree = {
    "token_embedder": {"embedding": np.arange(128, dtype=np.float32).reshape(16, 8)},
    "decoder": {
        "embedding_norm": {"scale": np.array([-0.0, 1.0], dtype=np.float32)},
        "layers_0": {"layer_0": {"kernel": np.arange(32, dtype=np.float32).reshape(4, 8).T}},
        "scanned_blocks": {"layer_0": {"kernel": np.arange(32, dtype=np.float32).reshape(4, 1, 8)}},
    },
}
tree = namespace["stage_tree"](tree, "/tmp/roundtrip")
jax.tree.map(lambda x: x.setflags(write=False), tree)
out = "/tmp/roundtrip"
mgr = ocp.CheckpointManager(
    out,
    item_handlers={"items": ocp.PyTreeCheckpointHandler(save_concurrent_gb=2, save_device_host_concurrent_gb=2)},
    options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False),
)
assert mgr.save(0, args=ocp.args.Composite(items=ocp.args.PyTreeSave({"params": {"params": tree}})))
mgr.wait_until_finished()
namespace["verify_saved"](tree, out)
print("Immutable handler and layerwise bitwise verification passed")
