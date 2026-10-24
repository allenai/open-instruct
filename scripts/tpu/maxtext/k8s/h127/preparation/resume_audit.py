"""Audit a disposable Tunix resume without modifying training updates or data.

Run as the SFT entrypoint with H127_RESUME_STEP set to the required source step.
Verify every host's locally addressable restored model and optimizer leaves are
installed exactly, and that the optimizer step counter was restored too.
"""

import hashlib
import json
import os
import runpy

import jax
import numpy as np
from flax import nnx
from tunix.sft import checkpoint_manager


def verify_installed(expected, actual):
    before, tree_before = jax.tree_util.tree_flatten_with_path(expected)
    after, tree_after = jax.tree_util.tree_flatten_with_path(actual)
    assert tree_before == tree_after, "Restored/live tree structures differ"
    values = 0
    leaves = 0
    digest = hashlib.sha256()
    for (path_a, a), (path_b, b) in zip(before, after, strict=True):
        assert path_a == path_b
        assert a.shape == b.shape and a.dtype == b.dtype, str(path_a)
        shards_a, shards_b = a.addressable_shards, b.addressable_shards
        assert len(shards_a) == len(shards_b), str(path_a)
        for sa, sb in zip(shards_a, shards_b, strict=True):
            assert str(sa.index) == str(sb.index), str(path_a)
            av, bv = np.asarray(sa.data), np.asarray(sb.data)
            assert av.tobytes() == bv.tobytes(), f"Restored value changed: {path_a}"
            digest.update(str(path_a).encode())
            digest.update(str(sa.index).encode())
            digest.update(av.tobytes())
            values += av.size
        leaves += 1
    return {"leaves": leaves, "local_values_including_replicas": values, "sha256": digest.hexdigest()}


def install(expected_step):
    original_restore = checkpoint_manager.CheckpointManager.maybe_restore

    def audited_restore(manager, model, optimizer=None, **kwargs):
        assert optimizer is not None
        assert not kwargs.get("restore_only_lora_params", False)
        assert manager.latest_step() == expected_step, "Missing/wrong resume checkpoint"
        captured = {}
        loader_type = type(manager._checkpointer)
        original_load = loader_type.load_checkpointables

        def audited_load(loader, step, checkpointables, *args, **load_kwargs):
            if loader is manager._checkpointer:
                assert step == expected_step
                assert "optimizer_state" in checkpointables, "Optimizer omitted from restore request"
            loaded = original_load(loader, step, checkpointables, *args, **load_kwargs)
            if loader is manager._checkpointer:
                captured.update(loaded)
            return loaded

        loader_type.load_checkpointables = audited_load
        try:
            result = original_restore(manager, model, optimizer, **kwargs)
        finally:
            loader_type.load_checkpointables = original_load
        assert result[0] == expected_step
        assert {"model_params", "optimizer_state"} <= captured.keys()
        receipt = {
            "process": jax.process_index(),
            "restored_step": result[0],
            "model": verify_installed(captured["model_params"], nnx.state(model)),
            "optimizer": verify_installed(captured["optimizer_state"], nnx.state(optimizer, nnx.optimizer.OptState)),
        }
        optimizer_step = int(np.asarray(optimizer.step.value.addressable_shards[0].data))
        assert optimizer_step == expected_step, (optimizer_step, expected_step)
        receipt["optimizer_step"] = optimizer_step
        print("H127-RESUME-RESTORE " + json.dumps(receipt, sort_keys=True), flush=True)
        return result

    checkpoint_manager.CheckpointManager.maybe_restore = audited_restore


if __name__ == "__main__":
    install(int(os.environ["H127_RESUME_STEP"]))
    runpy.run_module("maxtext.trainers.post_train.sft.train_sft", run_name="__main__")
