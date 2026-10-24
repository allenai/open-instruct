"""Run the pinned tensor mapping with bounded Orbax serialization concurrency."""

import hashlib
import inspect
import json
import textwrap
from pathlib import Path

import jax
import numpy as np
from orbax import checkpoint as ocp
from orbax.checkpoint import type_handlers


def stage_tree(tree, out):
    """Keep mapped weights on disk so serialization cannot duplicate a resident model."""
    stage = Path(out + "-mapped")
    stage.mkdir(exist_ok=False)
    flat, structure = jax.tree.flatten(tree)
    staged = []
    for index, value in enumerate(flat):
        assert value.dtype == np.float32
        path = stage / f"{index:05d}.npy"
        target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=value.shape)
        row_bytes = max(1, int(np.prod(value.shape[1:])) * 4)
        stride = max(1, (16 * 1024 * 1024) // row_bytes)
        for begin in range(0, value.shape[0], stride):
            target[begin : begin + stride] = value[begin : begin + stride]
            assert np.array_equal(
                target[begin : begin + stride].view(np.uint32), value[begin : begin + stride].view(np.uint32)
            )
        target.flush()
        del target
        staged.append(np.load(path, mmap_mode="r"))
        if index % 50 == 0:
            print("STAGED", index, "of", len(flat), flush=True)
    return jax.tree.unflatten(structure, staged)


def verify_saved(tree, out):
    """Restore one layer at a time and compare every mapped value bit-for-bit."""
    groups = [{"token_embedder": tree["token_embedder"]}]
    for name, value in tree["decoder"].items():
        if name in ("layers_0", "scanned_blocks"):
            groups.extend({"decoder": {name: {layer: layer_tree}}} for layer, layer_tree in value.items())
        else:
            groups.append({"decoder": {name: value}})
    checker = ocp.PyTreeCheckpointer()
    leaves, count = 0, 0
    for group in groups:
        target = {"params": {"params": group}}
        restored = checker.restore(
            str(Path(out) / "0/items"), args=ocp.args.PyTreeRestore(item=target, partial_restore=True)
        )
        original_leaves, original_structure = jax.tree.flatten(target)
        saved_leaves, saved_structure = jax.tree.flatten(restored)
        assert original_structure == saved_structure
        for expected, actual in zip(original_leaves, saved_leaves, strict=True):
            assert expected.shape == actual.shape and expected.dtype == actual.dtype == np.float32
            assert np.array_equal(expected.view(np.uint32), actual.view(np.uint32))
            leaves += 1
            count += expected.size
        del restored, saved_leaves
    assert leaves == len(jax.tree.leaves(tree))
    receipt = {
        "verified_leaves": leaves,
        "verified_values": count,
        "bitwise_equal": True,
        "method": "Full mapped tree restored in bounded layer groups; FP32 bit comparison",
    }
    (Path(out) / "serialization-verification.json").write_text(json.dumps(receipt, indent=2))
    print("SERIALIZATION-VERIFIED", receipt, flush=True)


# Orbax's standard NumPy handler snapshots *all* values before its byte limiter.
# This process saves synchronously, retains the complete tree, and never mutates it.
# Skip only that redundant snapshot; retain Orbax's checked serialization path.
serialize = type_handlers.NumpyHandler.serialize
handler_source = textwrap.dedent(inspect.getsource(serialize))
snapshot = "copied_values = [copy.deepcopy(v) for v in values]"
assert handler_source.count(snapshot) == 1
namespace = {}
exec(
    compile(handler_source.replace(snapshot, "copied_values = values"), "<immutable_numpy_save>", "exec"),
    serialize.__globals__,
    namespace,
)
type_handlers.NumpyHandler.serialize = namespace["serialize"]

source_path = Path(__file__).parent / "bundle/convert_olmocore_to_maxtext.py"
source = source_path.read_text()
old = 'mgr = ocp.CheckpointManager(out, item_names=("items",))'
new = """jax.tree.map(lambda x: x.setflags(write=False), tree)
print("MAPPING-COMPLETE; immutable synchronous save, 2 GiB concurrency", flush=True)
mgr = ocp.CheckpointManager(
    out,
    item_handlers={"items": ocp.PyTreeCheckpointHandler(save_concurrent_gb=2, save_device_host_concurrent_gb=2)},
    options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False),
)"""
assert source.count(old) == 1
print("PINNED-MAPPING-SHA256", hashlib.sha256(source.encode()).hexdigest(), flush=True)
assert source.count("mgr.wait_until_finished()") == 1
# This 16-layer model has exactly one scanned repetition. Inserting its size-one
# scan axis is a view, bitwise equivalent to stacking one array, without keeping
# a second copy of the entire scanned half of the model alive during conversion.
stack = "np.stack(xs, axis=1)"
assert source.count(stack) == 1
source = source.replace(stack, "(np.expand_dims(xs[0], axis=1) if len(xs) == 1 else np.stack(xs, axis=1))")
contiguous = "tree = jax.tree.map(lambda x: np.ascontiguousarray(x, dtype=np.float32), tree)"
assert source.count(contiguous) == 1
source = source.replace(contiguous, "tree = stage_tree(tree, out)\ndel p\n")
source = source.replace(old, new).replace(
    "mgr.wait_until_finished()", "mgr.wait_until_finished()\nverify_saved(tree, out)"
)
exec(compile(source, str(source_path), "exec"))
