import unittest

import torch
from parameterized import parameterized

from open_instruct.distillkit.compression import (
    DistributionQuantizationConfig,
    LogprobCompressor,
    QuantizationBin,
    pack_to_bytes,
    unpack_from_bytes,
)


class TestCompressionPrimitives(unittest.TestCase):
    @parameterized.expand([(1,), (3,), (8,), (11,), (16,)])
    def test_bitpack_roundtrip(self, elem_bits: int) -> None:
        torch.manual_seed(1)
        max_val = (1 << elem_bits) - 1
        values = torch.randint(0, max_val + 1, size=(4, 7), dtype=torch.long)
        packed = pack_to_bytes(values, elem_bits=elem_bits)
        unpacked = unpack_from_bytes(packed, elem_bits=elem_bits, original_num_elements=values.shape[-1])
        self.assertTrue(torch.equal(unpacked, values))

    def test_pack_to_bytes_no_padding_noop_when_aligned(self) -> None:
        # 2 values * 4 bits = 8 bits exactly, so pad length is 0.
        values = torch.tensor([[0b1010, 0b0101]], dtype=torch.long)
        packed = pack_to_bytes(values, elem_bits=4)

        self.assertEqual(packed.shape, (1, 1))
        self.assertEqual(int(packed.item()), 0b10100101)

        unpacked = unpack_from_bytes(packed, elem_bits=4, original_num_elements=2)
        self.assertTrue(torch.equal(unpacked, values))

    def test_distribution_quantization_config_roundtrip(self) -> None:
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
        self.assertEqual(cfg_out["d"], 128256)
        self.assertEqual(cfg_out["k"], 16)
        self.assertEqual(cfg_out["exact_k"], 8)
        self.assertEqual(cfg_out["term_dtype"], "float32")
        self.assertEqual(cfg_out["polynomial_terms"], [0, 1, "sqrt"])
        self.assertEqual(cfg_out["residual_bins"][0]["num_elements"], 4)

    def test_distribution_quantization_config_accepts_torch_dtypes(self) -> None:
        cfg = DistributionQuantizationConfig(
            d=256,
            k=8,
            exact_k=4,
            exact_dtype=torch.float32,
            term_dtype=torch.float16,
            residual_bins=[QuantizationBin(scale_dtype=torch.bfloat16, element_bits=8, num_elements=4)],
            polynomial_terms=[0, 1],
        )
        cfg_out = cfg.to_dict()
        self.assertEqual(cfg_out["exact_dtype"], "float32")
        self.assertEqual(cfg_out["term_dtype"], "float16")
        self.assertEqual(cfg_out["residual_bins"][0]["scale_dtype"], "bfloat16")

    def test_sparse_compressor_lossless_when_exact_only(self) -> None:
        cfg = DistributionQuantizationConfig(
            d=64,
            k=4,
            exact_k=4,
            exact_dtype=torch.float32,
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

        self.assertTrue(torch.equal(out_indices, expected_indices))
        self.assertTrue(torch.allclose(out_logprobs, expected_logprobs, atol=0.0, rtol=0.0))
