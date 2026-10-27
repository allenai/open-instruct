"""Real Ray roundtrip of the native ragged image transport and policy metadata."""

from types import SimpleNamespace

import ray
import torch
from miles.ray.rollout import train_data_conversion
from miles.utils import object_store, types


@ray.remote
def read_images(reference):
    store = object_store.RayObjectStore()
    with store.get(reference) as value:
        # Return the transported tensors from another process, not an alias to
        # the producer's original Python objects.
        return value


def test_real_ray_preserves_ragged_images_and_policy_boundaries():
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(num_cpus=1, include_dashboard=False, object_store_memory=100 * 1024**2)
    try:
        samples = []
        for index, crops in enumerate((1, 3)):
            sample = types.Sample(
                index=index,
                group_index=index,
                tokens=[0, 3, 3, 9],
                response="A",
                response_length=1,
                reward=float(index),
                rollout_log_probs=[-0.7],
                status=types.Sample.Status.COMPLETED,
                train_metadata={"prepared_sample_id": f"image-{index}", "vision": {"hash": str(index)}},
                multimodal_train_inputs={
                    "images": torch.randn(1, crops, 4, 12),
                    "pooled_patches_idx": torch.tensor([[[0, 1], [2, -1]]]),
                    "image_positions": torch.tensor([1, 2]),
                },
                weight_versions=[types.WeightVersionsPerCall([types.WeightVersionSpan("2", 3, 4)])],
            )
            samples.append(sample)
        args = SimpleNamespace(
            reward_key=None,
            advantage_estimator="grpo",
            rewards_normalization=False,
            use_dynamic_global_batch_size=False,
        )
        value = train_data_conversion.convert_samples_to_train_data(args, samples, {}, None, None)
        store = object_store.RayObjectStore()
        reference = store.put(value, train_data_conversion.ROLLOUT_DATA_VALUE_SPEC)
        actual = ray.get(read_images.remote(reference), timeout=60)
        assert actual["loss_masks"] == [[1], [1]]
        assert actual["response_lengths"] == [1, 1]
        assert actual["metadata"] == value["metadata"]
        assert actual["weight_versions"] == value["weight_versions"]
        for left, right in zip(actual["multimodal_train_inputs"], value["multimodal_train_inputs"], strict=True):
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
                assert left[name].dtype == right[name].dtype
    finally:
        if owns_ray:
            ray.shutdown()
