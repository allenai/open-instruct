"""Core publication using the flattened NCCL protocol qualified by olmo-miles."""

import ray
from miles.backends.fsdp_utils import update_weight_utils
from torch import distributed as dist


class FlattenedDistributedUpdater(update_weight_utils.UpdateWeightFromDistributed):
    """One NCCL broadcast per bucket, preserving the HF name/shape order."""

    def update_bucket_weights(self, named_tensors, weight_version=None):
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
        pending = [
            engine._make_request.remote("update_weights_from_distributed", payload) for engine in self.rollout_engines
        ]
        dist.broadcast(flat, 0, group=self._model_update_groups, async_op=True).wait()
        for result in ray.get(pending):
            success = result.get("success", True) if isinstance(result, dict) else getattr(result, "success", True)
            if not success:
                raise RuntimeError(f"SGLang rejected weight bucket: {result}")
