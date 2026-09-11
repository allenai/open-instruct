"""A conversion gate must reject incomplete, reshaped or corrupted weight streams."""

import json

import pytest
import torch
from safetensors.torch import save_file
from scripts.miles.checkpoint_weights import SafeTensorState, compare_stream
from scripts.miles.launch_hero_conversion import specification


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unexpected", "shape", "value", "nan"])
def test_conversion_comparison_rejects_corrupt_stream(fault):
    reference = {"a": torch.ones(2, 3), "b": torch.zeros(3)}
    stream = [(name, tensor.clone()) for name, tensor in reference.items()]
    if fault == "missing":
        stream.pop()
    elif fault == "duplicate":
        stream.append(stream[0])
    elif fault == "unexpected":
        stream.append(("c", torch.ones(1)))
    elif fault == "shape":
        stream[0] = ("a", torch.ones(3, 2))
    elif fault == "value":
        stream[0][1][0, 0] += 0.01
    elif fault == "nan":
        stream[0][1][0, 0] = float("nan")
    with pytest.raises(ValueError):
        compare_stream(iter(stream), reference)


def test_records_export_cast_and_only_explicit_vocabulary_padding():
    original = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7
    reference = {"model.embed_tokens.weight": original[:3].bfloat16()}
    stream = list({"model.embed_tokens.weight": original}.items())
    result = compare_stream(iter(stream), reference, native_vocab_size=4, hf_vocab_size=3)
    assert result["exact_match_after_export_cast"]
    assert result["dtype_conversions"] == ["torch.float32->torch.bfloat16"]
    assert result["trimmed_vocabulary_tensors"] == ["model.embed_tokens.weight"]
    with pytest.raises(ValueError, match="Shape mismatch"):
        compare_stream(iter(stream), reference)
    with pytest.raises(ValueError, match="Vocabulary size"):
        compare_stream(iter(stream), reference, native_vocab_size=5, hf_vocab_size=3)


def test_reads_all_shards_and_rejects_duplicate_names(tmp_path):
    save_file({"a": torch.ones(2)}, tmp_path / "one.safetensors")
    save_file({"b": torch.zeros(3)}, tmp_path / "two.safetensors")
    with SafeTensorState(tmp_path) as state:
        assert set(state) == {"a", "b"}
        assert compare_stream(iter(state.items()), state)["tensor_count"] == 2
    save_file({"a": torch.ones(2)}, tmp_path / "duplicate.safetensors")
    with pytest.raises(ValueError, match="Duplicate"):
        SafeTensorState(tmp_path)


def test_cpu_weka_audit_uses_saturn_with_reserved_runtime():
    task = specification("test-image")["tasks"][0]
    assert task["constraints"]["cluster"] == ["ai2/saturn"]
    assert task["resources"]["gpuCount"] == 0
    assert task["context"] == {"priority": "urgent", "minRuntime": "30m", "autoResume": False}
    assert "--report /output/conversion.json" in task["arguments"][0]


@pytest.mark.parametrize("fault", [None, "missing_shard", "swapped_shards", "missing_key", "extra_key", "invalid_map"])
def test_index_must_match_exact_tensor_to_shard_inventory(tmp_path, fault):
    save_file({"a": torch.ones(2)}, tmp_path / "one.safetensors")
    save_file({"b": torch.zeros(3)}, tmp_path / "two.safetensors")
    mapping = {"a": "one.safetensors", "b": "two.safetensors"}
    if fault == "missing_shard":
        mapping["a"] = "missing.safetensors"
    elif fault == "swapped_shards":
        mapping = {"a": "two.safetensors", "b": "one.safetensors"}
    elif fault == "missing_key":
        del mapping["b"]
    elif fault == "extra_key":
        mapping["c"] = "one.safetensors"
    elif fault == "invalid_map":
        mapping = []
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    if fault is None:
        with SafeTensorState(tmp_path) as state:
            assert state.tensor_shards == mapping
            assert set(state) == {"a", "b"}
    else:
        with pytest.raises(ValueError, match="Safetensors index"):
            SafeTensorState(tmp_path)
