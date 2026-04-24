"""Tests for offline distillation collator and distillkit loss wiring."""

import gzip
import json
import os
import shutil
import tempfile
import unittest

import torch
from parameterized import parameterized
from transformers import AutoTokenizer

from open_instruct.dataset_transformation import (
    ATTENTION_MASK_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    MASKED_TOKEN_VALUE,
    distill_pretokenized_filter_v1,
    distill_pretokenized_v1,
)
from open_instruct.distillation_collator import DistillationDataCollator
from open_instruct.distillkit.compression import DistributionQuantizationConfig, LogprobCompressor
from open_instruct.distillkit.distill_loss import DistillationLossComputer, HingeLoss, KLDLoss, create_loss_function


def _local_tokenizer_path() -> str:
    """Unpack the checked-in test tokenizer without importing `test_dataset_transformation` (heavy imports)."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "test_data", "tokenizer")
    dst_dir = tempfile.mkdtemp(prefix="distill_tok_")
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        if name.endswith(".gz"):
            dst = os.path.join(dst_dir, name[:-3])
            with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src, dst_dir)
    return dst_dir


def _tiny_compressor_config_dict() -> dict:
    """Minimal valid quantization config for tests (small vocab, top-k 8)."""
    return {
        "d": 128,
        "k": 8,
        "exact_k": 4,
        "exact_dtype": "float32",
        "polynomial_terms": [0, 1],
        "term_dtype": "float32",
        "residual_bins": [],
        "delta_encoding": False,
        "error_diffusion": False,
        "normalize_t": False,
    }


class TestCreateLossFunction(unittest.TestCase):
    @parameterized.expand([("cross_entropy",), ("kl",), ("jsd",), ("tvd",), ("hinge",), ("logistic_ranking",)])
    def test_known_functions_return_name_weight_callable(self, name: str) -> None:
        cfg: dict = {"function": name, "weight": 1.0}
        if name == "hinge":
            cfg["margin"] = 0.1
        if name in {"kl", "jsd", "tvd"}:
            cfg["missing_probability_handling"] = "zero"
        fn_name, weight, _fn = create_loss_function(cfg)
        self.assertEqual(fn_name, name)
        self.assertEqual(weight, 1.0)

    def test_unknown_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown loss function"):
            create_loss_function({"function": "not_a_real_loss"})

    def test_kl_config_fields_are_wired(self) -> None:
        _name, _weight, loss_fn = create_loss_function(
            {
                "function": "kl",
                "weight": 2.0,
                "temperature": 1.7,
                "missing_probability_handling": "zero",
                "sparse_chunk_length": 16,
            }
        )
        self.assertIsInstance(loss_fn, KLDLoss)
        self.assertEqual(loss_fn.temperature, 1.7)
        self.assertEqual(loss_fn.chunk_length, 16)

    def test_hinge_margin_is_wired(self) -> None:
        _name, _weight, loss_fn = create_loss_function({"function": "hinge", "weight": 0.5, "margin": 0.25})
        self.assertIsInstance(loss_fn, HingeLoss)
        self.assertEqual(loss_fn.margin, 0.25)


class TestDistillationLossComputer(unittest.TestCase):
    def test_zero_total_weight_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Sum of loss weights"):
            DistillationLossComputer([{"function": "cross_entropy", "weight": 0.0}], {}, vocab_size=100)

    def test_cross_entropy_only_forward(self) -> None:
        comp = DistillationLossComputer([{"function": "cross_entropy", "weight": 1.0}], {}, vocab_size=256)
        logits = torch.randn(2, 5, 256, dtype=torch.float32)
        labels = torch.full((2, 5), -100, dtype=torch.long)
        labels[:, 2:] = torch.randint(0, 256, (2, 3))
        total, loss_dict = comp.compute_loss(
            student_logits=logits, model_loss=torch.tensor(0.25, dtype=torch.float32), labels=labels, batch={}
        )
        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(loss_dict["total_loss"], total.item(), places=5)
        self.assertAlmostEqual(loss_dict["cross_entropy_loss"], 0.25, places=5)

    def test_kl_with_synthetic_compressed_batch(self) -> None:
        cfg_dict = _tiny_compressor_config_dict()
        cfg = DistributionQuantizationConfig.from_dict(cfg_dict)
        compressor = LogprobCompressor(config=cfg)
        seq = 4
        k = cfg.k
        indices = torch.stack([torch.arange(k, dtype=torch.long) % cfg.d for _ in range(seq)], dim=0)
        logprobs = torch.randn(seq, k)
        packed = compressor.compress_from_sparse(indices, logprobs)
        # Batch size 1; time axis matches labels[:, 1:] (seq positions).
        batch = {
            "compressed_logprobs": packed["compressed_logprobs"].unsqueeze(0),
            "bytepacked_indices": packed["bytepacked_indices"].unsqueeze(0),
        }
        comp = DistillationLossComputer([{"function": "kl", "weight": 1.0}], cfg_dict, vocab_size=cfg.d)
        vocab_size = cfg.d
        logits = torch.randn(1, seq + 1, vocab_size, dtype=torch.float32)
        labels = torch.full((1, seq + 1), -100, dtype=torch.long)
        labels[:, 1:] = torch.randint(0, vocab_size, (1, seq))
        total, loss_dict = comp.compute_loss(
            student_logits=logits, model_loss=torch.tensor(0.0, dtype=torch.float32), labels=labels, batch=batch
        )
        self.assertTrue(torch.isfinite(total))
        self.assertIn("kl_loss", loss_dict)
        self.assertIn("total_loss", loss_dict)

    def test_distill_loss_without_compressor_config_raises(self) -> None:
        with self.assertRaises((KeyError, ValueError)):
            DistillationLossComputer([{"function": "kl", "weight": 1.0}], {}, vocab_size=128)

    def test_missing_compressed_field_raises(self) -> None:
        cfg_dict = _tiny_compressor_config_dict()
        cfg = DistributionQuantizationConfig.from_dict(cfg_dict)
        compressor = LogprobCompressor(config=cfg)
        seq = 4
        indices = torch.stack([torch.arange(cfg.k, dtype=torch.long) % cfg.d for _ in range(seq)], dim=0)
        logprobs = torch.randn(seq, cfg.k)
        packed = compressor.compress_from_sparse(indices, logprobs)
        batch = {"compressed_logprobs": packed["compressed_logprobs"].unsqueeze(0)}
        comp = DistillationLossComputer([{"function": "kl", "weight": 1.0}], cfg_dict, vocab_size=cfg.d)
        logits = torch.randn(1, seq + 1, cfg.d, dtype=torch.float32)
        labels = torch.randint(0, cfg.d, (1, seq + 1), dtype=torch.long)
        with self.assertRaises(KeyError):
            comp.compute_loss(
                student_logits=logits, model_loss=torch.tensor(0.0, dtype=torch.float32), labels=labels, batch=batch
            )


class TestDistillationDataCollator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(_local_tokenizer_path())

    def test_standard_batch_without_distill_keys(self) -> None:
        collator = DistillationDataCollator(tokenizer=self.tokenizer)
        features = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "labels": [4, 5], "attention_mask": [1, 1]},
        ]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertNotIn("compressed_logprobs", batch)

    def test_pads_compressed_tensors_to_max_seq_minus_one(self) -> None:
        collator = DistillationDataCollator(tokenizer=self.tokenizer)
        # Length 4 tokens -> 3 distill positions; length 6 -> 5 positions.
        f1_compressed = torch.zeros(3, 7)
        f1_indices = torch.zeros(3, 5)
        f2_compressed = torch.zeros(5, 7)
        f2_indices = torch.zeros(5, 5)
        features = [
            {
                "input_ids": [10, 11, 12, 13],
                "labels": [10, 11, 12, 13],
                "attention_mask": [1, 1, 1, 1],
                "compressed_logprobs": f1_compressed,
                "bytepacked_indices": f1_indices,
            },
            {
                "input_ids": [20, 21, 22, 23, 24, 25],
                "labels": [20, 21, 22, 23, 24, 25],
                "attention_mask": [1, 1, 1, 1, 1, 1],
                "compressed_logprobs": f2_compressed,
                "bytepacked_indices": f2_indices,
            },
        ]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape[1], 6)
        self.assertEqual(batch["compressed_logprobs"].shape, (2, 5, 7))
        self.assertEqual(batch["bytepacked_indices"].shape, (2, 5, 5))
        self.assertTrue(torch.all(batch["compressed_logprobs"][0, 3:] == 0))

    def test_handles_single_distill_key_present(self) -> None:
        collator = DistillationDataCollator(tokenizer=self.tokenizer)
        features = [
            {
                "input_ids": [10, 11, 12],
                "labels": [10, 11, 12],
                "attention_mask": [1, 1, 1],
                "compressed_logprobs": torch.zeros(2, 3),
            },
            {
                "input_ids": [20, 21, 22, 23],
                "labels": [20, 21, 22, 23],
                "attention_mask": [1, 1, 1, 1],
                "compressed_logprobs": torch.zeros(3, 3),
            },
        ]
        batch = collator(features)
        self.assertIn("compressed_logprobs", batch)
        self.assertNotIn("bytepacked_indices", batch)
        self.assertEqual(batch["compressed_logprobs"].shape, (2, 3, 3))


class TestDistillDatasetTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer_dir = _local_tokenizer_path()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.tokenizer_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tokenizer_dir, ignore_errors=True)

    def test_distill_pretokenized_masks_non_assistant_prefix(self) -> None:
        messages = [{"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "2 + 2 is 4."}]
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        user_prefix = self.tokenizer.apply_chat_template(
            conversation=[messages[0]], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        row = {
            INPUT_IDS_KEY: input_ids,
            "messages": json.dumps(messages),
            "compressed_logprobs": [],
            "bytepacked_indices": [],
        }
        out = distill_pretokenized_v1(row, self.tokenizer, max_seq_length=4096)
        self.assertEqual(out[ATTENTION_MASK_KEY], [1] * len(input_ids))
        self.assertEqual(out[LABELS_KEY][: len(user_prefix)], [MASKED_TOKEN_VALUE] * len(user_prefix))
        self.assertEqual(out[LABELS_KEY][len(user_prefix) :], input_ids[len(user_prefix) :])

    def test_distill_pretokenized_raises_when_sequence_too_long(self) -> None:
        row = {INPUT_IDS_KEY: [1, 2, 3, 4], "messages": ""}
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_length"):
            distill_pretokenized_v1(row, self.tokenizer, max_seq_length=2)

    def test_distill_filter_behavior(self) -> None:
        self.assertFalse(distill_pretokenized_filter_v1({}, self.tokenizer))
        self.assertFalse(distill_pretokenized_filter_v1({INPUT_IDS_KEY: []}, self.tokenizer))
        self.assertFalse(
            distill_pretokenized_filter_v1(
                {INPUT_IDS_KEY: [1, 2], LABELS_KEY: [MASKED_TOKEN_VALUE, MASKED_TOKEN_VALUE]}, self.tokenizer
            )
        )
        self.assertTrue(
            distill_pretokenized_filter_v1(
                {INPUT_IDS_KEY: [1, 2], LABELS_KEY: [MASKED_TOKEN_VALUE, 2]}, self.tokenizer
            )
        )
        self.assertTrue(distill_pretokenized_filter_v1({INPUT_IDS_KEY: [1, 2]}, self.tokenizer))


if __name__ == "__main__":
    unittest.main()
