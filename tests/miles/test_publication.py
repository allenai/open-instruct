"""Check the flattened wire contract independently of CUDA placement."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from open_instruct.miles import publication


def test_flattened_broadcast_preserves_mixed_dtypes_and_order(monkeypatch):
    updater = publication.FlattenedDistributedUpdater.__new__(publication.FlattenedDistributedUpdater)
    updater._is_src_rank = True
    updater._group_name = "test"
    updater._model_update_groups = object()
    remote = Mock(return_value={"success": True})
    updater.rollout_engines = [SimpleNamespace(_make_request=SimpleNamespace(remote=remote))]
    broadcast = Mock(return_value=SimpleNamespace(wait=lambda: None))
    monkeypatch.setattr(publication.dist, "broadcast", broadcast)
    monkeypatch.setattr(publication.ray, "get", lambda values: values)
    # Non-contiguous expert transpose and mixed dtype exercise the byte contract.
    tensors = [
        ("expert", torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).T),
        ("decay", torch.tensor([0.125, 1.234567], dtype=torch.float32)),
    ]
    updater.update_bucket_weights(tensors, weight_version=7)
    assert broadcast.call_count == 1
    flat = broadcast.call_args.args[0]
    operation, payload = remote.call_args.args
    assert operation == "update_weights_from_distributed"
    assert payload["weight_version"] == "7"
    offset = 0
    for name, dtype, shape in zip(payload["names"], payload["dtypes"], payload["shapes"], strict=True):
        original = dict(tensors)[name]
        restored = flat[offset : offset + original.nbytes].view(getattr(torch, dtype)).reshape(shape)
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
        offset += original.nbytes
    assert offset == flat.numel()
    remote.return_value = {"success": False, "message": "load rejected"}
    with pytest.raises(RuntimeError, match="load rejected"):
        updater.update_bucket_weights(tensors, weight_version=8)
