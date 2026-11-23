"""Checkpoint comparisons include scalar parameters without changing their shape."""

import pytest
import torch
from safetensors import torch as safetensors_torch
from scripts.miles import checkpoint_weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64])
def test_scalar_checkpoint_comparison(tmp_path, dtype):
    tensors = {"attention.ssmax_scale": torch.tensor(1, dtype=dtype), "weight": torch.tensor([2, 3], dtype=dtype)}
    safetensors_torch.save_file(tensors, tmp_path / "model.safetensors")
    with checkpoint_weights.SafeTensorState(tmp_path) as reference:
        result = checkpoint_weights.compare_stream(tensors.items(), reference)
    assert result["exact_match_after_export_cast"] is True
    assert result["tensor_count"] == 2
    assert result["bytes"] == 3 * tensors["weight"].element_size()
    assert result["attention_parameter_shapes"] == {"attention.ssmax_scale": []}
    vector = {"attention.ssmax_scale": tensors["attention.ssmax_scale"].reshape(1), "weight": tensors["weight"]}
    vector_result = checkpoint_weights.compare_stream(vector.items(), vector)
    assert result["ordered_tensor_content_sha256"] != vector_result["ordered_tensor_content_sha256"]


@pytest.mark.parametrize(
    "actual,expected,message",
    [
        (torch.tensor(1.0), torch.tensor(2.0), "Value mismatch"),
        (torch.tensor(1.0), torch.tensor([1.0]), "Shape mismatch"),
        (torch.tensor(float("nan")), torch.tensor(1.0), "Nonfinite"),
    ],
)
def test_scalar_checkpoint_still_rejects_mismatches(actual, expected, message):
    with pytest.raises(ValueError, match=message):
        checkpoint_weights.compare_stream([("scale", actual)], {"scale": expected})
