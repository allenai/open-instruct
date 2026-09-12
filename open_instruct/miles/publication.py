"""Core publication using the flattened NCCL protocol qualified by olmo-miles."""

import time

import ray
from miles.backends.fsdp_utils import update_weight_utils
from torch import distributed as dist


class FlattenedDistributedUpdater(update_weight_utils.UpdateWeightFromDistributed):
    """One NCCL broadcast per bucket, preserving the HF name/shape order.

    ``last_bucket_timing`` splits each bucket into the broadcast wait, which
    covers the engine accepting the request and receiving the bytes, and the
    engine wait, which covers reconstruction and ``load_weights`` until the
    engine responds. The split attributes publication time between transport
    and engine-side loading without instrumenting the engine.
    """

    last_bucket_timing = None

    def update_bucket_weights(self, named_tensors, weight_version=None):
        self.last_bucket_timing = None
        if not self._is_src_rank or not named_tensors:
            return
        names = [name for name, _ in named_tensors]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate names in a weight bucket")
        if any(tensor.numel() == 0 for _, tensor in named_tensors):
            raise ValueError("Empty tensor in a weight bucket")
        if len({tensor.device for _, tensor in named_tensors}) != 1:
            raise ValueError("Weight bucket must reside on one device")
        bucket = update_weight_utils.FlattenedTensorBucket(named_tensors=named_tensors)
        flat = bucket.get_flattened_tensor()
        expected = sum(tensor.nbytes for _, tensor in named_tensors)
        if flat.element_size() != 1 or flat.numel() != expected or not flat.is_contiguous():
            raise ValueError("Pinned SGLang flattened byte layout changed")
        payload = dict(
            names=names,
            dtypes=[str(tensor.dtype).removeprefix("torch.") for _, tensor in named_tensors],
            shapes=[list(tensor.shape) for _, tensor in named_tensors],
            group_name=self._group_name,
            weight_version=str(weight_version),
            load_format="flattened_bucket",
            flush_cache=False,
        )
        # The pinned engine's public wrapper does not expose load_format.
        # This is the same HTTP request contract used by olmo-miles direct export.
        started = time.perf_counter()
        pending = [
            engine._make_request.remote("update_weights_from_distributed", payload) for engine in self.rollout_engines
        ]
        dist.broadcast(flat, 0, group=self._model_update_groups, async_op=True).wait()
        broadcast_done = time.perf_counter()
        results = ray.get(pending)
        finished = time.perf_counter()
        self.last_bucket_timing = {
            "tensors": len(names),
            "bytes": expected,
            "broadcast_seconds": broadcast_done - started,
            "engine_seconds": finished - broadcast_done,
        }
        for result in results:
            success = result.get("success", True) if isinstance(result, dict) else getattr(result, "success", True)
            if not success:
                raise RuntimeError(f"SGLang rejected weight bucket: {result}")
