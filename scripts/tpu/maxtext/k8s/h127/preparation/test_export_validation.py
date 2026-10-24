"""Verify that export audits reject nonfinite serialized BF16 values."""

import hashlib
import json
import struct

import pytest
from export_bounded import tensor_hashes


@pytest.mark.parametrize("bits", [0x7F80, 0xFF80, 0x7FC1])
def test_nonfinite_bf16_rejected(tmp_path, bits):
    path = tmp_path / "model.safetensors"
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<H", bits))
    with pytest.raises(AssertionError, match="Nonfinite BF16 tensor"):
        tensor_hashes(path)


def test_finite_bf16_hash_unchanged(tmp_path):
    path = tmp_path / "model.safetensors"
    data = struct.pack("<HHH", 0x3F80, 0xBF80, 0)
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    assert tensor_hashes(path) == {"weight": hashlib.sha256(data).hexdigest()}
