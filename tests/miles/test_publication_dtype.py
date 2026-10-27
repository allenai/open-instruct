"""Initial and rolling publications must preserve FP32 KDA decay values."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import torch
from miles.backends.core_utils import actor
from miles.backends.core_utils.publication import engine_delivery
from miles.backends.core_utils.publication.state import PolicyClock
from miles.backends.fsdp_utils.update_weight_utils import FlattenedTensorBucket

from open_instruct.miles.configuration.config import CoreConfig


def weights():
    # An odd BF16 byte prefix also exercises FP32 view alignment in the wire format.
    return {
        "embedding": torch.tensor([1.0], dtype=torch.bfloat16),
        "linear_attn.A_log": torch.tensor([0.1234567, 1.234567], dtype=torch.float32),
        "linear_attn.dt_bias": torch.tensor([-0.0312345], dtype=torch.float32),
        "projection": torch.tensor([2.0, 3.0], dtype=torch.bfloat16),
    }


def test_initial_publication_preserves_mixed_dtype_values(monkeypatch):
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(olmo_core=CoreConfig(), colocate=False, update_weight_buffer_size=1024, save=None)
    worker.clock = SimpleNamespace(completed_steps=0, published_step=-1, published=Mock())
    worker.train_module = worker.hf_config = None
    original = weights()
    received = {}
    sizes = []

    def send(named_tensors, weight_version=None):
        assert weight_version == 0
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        sizes.append(bucket.get_flattened_tensor().numel())
        received.update({name: tensor.clone() for name, tensor in bucket.reconstruct_tensors()})

    worker.weight_updater = SimpleNamespace(
        connect_rollout_engines=Mock(), update_bucket_weights=send, last_bucket_timing=None
    )
    engine = SimpleNamespace(
        **{
            name: AsyncMock()
            for name in (
                "pause_generation",
                "begin_weight_update",
                "flush_cache",
                "end_weight_update",
                "update_weight_version",
                "continue_generation",
            )
        }
    )
    original_to = torch.Tensor.to

    def cpu_transport(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_transport)
    monkeypatch.setattr(actor.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(actor.dist, "barrier", lambda: None)
    monkeypatch.setattr(actor.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor.models, "iter_export_state", lambda *a, **kw: iter(original.items()))
    worker.update_weights(
        SimpleNamespace(
            rollout_engines=[engine], engine_gpu_counts=[1], engine_gpu_offsets=[0], snapshot_cell_id_to_hashes={}
        )
    )
    assert sum(sizes) == sum(value.nbytes for value in original.values())
    assert received.keys() == original.keys()
    for name, value in original.items():
        assert received[name].dtype == value.dtype
        assert torch.equal(received[name], value)


def test_rolling_snapshot_preserves_dtype_values_and_owned_bytes(monkeypatch):
    original = weights()
    expected = {name: value.clone() for name, value in original.items()}
    stored = {}

    def put(value):
        stored[len(stored)] = value.copy()
        return len(stored) - 1

    monkeypatch.setattr(engine_delivery.ray, "put", put)
    monkeypatch.setattr(engine_delivery.dist, "barrier", lambda: None)
    monkeypatch.setattr(engine_delivery.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(engine_delivery.models, "iter_export_state", lambda *a, **kw: iter(original.items()))
    worker = SimpleNamespace(
        args=SimpleNamespace(update_weight_buffer_size=1024, olmo_core=CoreConfig()),
        clock=PolicyClock(completed_steps=1),
        train_module=None,
        hf_config=None,
        _agree=lambda fn: fn(),
    )
    snapshot = engine_delivery.capture(worker)
    assert snapshot.nbytes == sum(value.nbytes for value in expected.values())
    for value in original.values():
        value.add_(100)
    received = {}
    for metadata, reference in snapshot.buckets:
        flat = torch.from_numpy(stored[reference])
        offset = 0
        for name, shape, dtype in metadata:
            size = expected[name].nbytes
            received[name] = flat[offset : offset + size].view(getattr(torch, dtype)).reshape(shape)
            offset += size
    for name, value in expected.items():
        assert received[name].dtype == value.dtype
        assert torch.equal(received[name], value)
