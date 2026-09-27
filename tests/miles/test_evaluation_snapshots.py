"""CPU contracts for independent, lossy background evaluation."""

from miles.backends.core_utils import checkpoint

from open_instruct.miles.evaluation import evaluation


def test_eval_snapshots_are_outside_checkpoint_cleanup(tmp_path):
    from_checkpoint_root = tmp_path / "checkpoints"
    for update in (1, 2):
        path = from_checkpoint_root / "core" / f"rollout_{update:07d}"
        path.mkdir(parents=True)
        (path / "complete.json").write_text("{}")
    frozen = evaluation.snapshot(tmp_path, 1)
    frozen.mkdir(parents=True)
    (frozen / "model.safetensors").write_bytes(b"immutable")
    assert checkpoint.prune(from_checkpoint_root, 2, keep_last=1, keep_every=None) == [1]
    assert (frozen / "model.safetensors").read_bytes() == b"immutable"
