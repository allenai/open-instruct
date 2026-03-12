import pytest
import torch

from open_instruct.distillkit.compression import (
    DistributionQuantizationConfig,
    LogprobCompressor,
    TermDtype,
    pack_to_bytes,
    unpack_from_bytes,
)


@pytest.mark.parametrize("elem_bits", [1, 3, 8, 11, 16])
def test_bitpack_roundtrip(elem_bits: int) -> None:
    torch.manual_seed(1)
    max_val = (1 << elem_bits) - 1
    values = torch.randint(0, max_val + 1, size=(4, 7), dtype=torch.long)
    packed = pack_to_bytes(values, elem_bits=elem_bits)
    unpacked = unpack_from_bytes(packed, elem_bits=elem_bits, original_num_elements=values.shape[-1])
    assert torch.equal(unpacked, values)


def test_distribution_quantization_config_roundtrip() -> None:
    cfg_in = {
        "d": 128256,
        "k": 16,
        "exact_k": 8,
        "exact_dtype": "float32",
        "polynomial_terms": [0, 1, "sqrt"],
        "term_dtype": "float32",
        "residual_bins": [{"scale_dtype": "float32", "element_bits": 8, "num_elements": 4}],
        "delta_encoding": True,
        "error_diffusion": False,
        "normalize_t": True,
    }
    cfg = DistributionQuantizationConfig.from_dict(cfg_in)
    cfg_out = cfg.to_dict()
    assert cfg_out["d"] == 128256
    assert cfg_out["k"] == 16
    assert cfg_out["exact_k"] == 8
    assert cfg_out["term_dtype"] == "float32"
    assert cfg_out["polynomial_terms"] == [0, 1, "sqrt"]
    assert cfg_out["residual_bins"][0]["num_elements"] == 4


def test_sparse_compressor_lossless_when_exact_only() -> None:
    cfg = DistributionQuantizationConfig(
        d=64,
        k=4,
        exact_k=4,
        exact_dtype=TermDtype.FLOAT32,
        polynomial_terms=None,
        residual_bins=[],
        delta_encoding=False,
    )
    compressor = LogprobCompressor(cfg)

    indices = torch.tensor([[[2, 5, 7, 1], [3, 4, 9, 0]]], dtype=torch.long)
    logprobs = torch.tensor([[[-0.2, -0.8, -1.5, -2.0], [-0.1, -0.7, -1.2, -2.8]]], dtype=torch.float32)

    row = compressor.compress_from_sparse(indices, logprobs)
    out_indices, out_logprobs = compressor.decompress_to_sparse(row)

    sort_idx = torch.argsort(logprobs, dim=-1, descending=True)
    expected_logprobs = torch.gather(logprobs, -1, sort_idx)
    expected_indices = torch.gather(indices, -1, sort_idx)

    assert torch.equal(out_indices, expected_indices)
    assert torch.allclose(out_logprobs, expected_logprobs, atol=0.0, rtol=0.0)
