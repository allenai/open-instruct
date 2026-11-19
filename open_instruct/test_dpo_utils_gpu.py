"""GPU integration tests for DPO utils including TensorCache.

These tests require CUDA and will be skipped if not available.

To run:
    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh
"""

import pathlib
import tempfile
import unittest

import torch
from olmo_core.nn.transformer import TransformerConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct import dpo_utils, model_utils
from open_instruct.olmo_core_train_modules import DPOLMHead


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTensorCacheGPU(unittest.TestCase):
    def test_tensor_cache_gpu_indexing(self):
        cache = model_utils.TensorCache(
            tensors={
                "chosen_logps": torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda(),
                "rejected_logps": torch.tensor([5.0, 6.0, 7.0, 8.0]).cuda(),
            }
        )
        indices = torch.tensor([0, 2]).cuda()
        result = cache[indices]

        self.assertEqual(result["chosen_logps"].device.type, "cuda")
        self.assertTrue(torch.equal(result["chosen_logps"], torch.tensor([1.0, 3.0]).cuda()))
        self.assertTrue(torch.equal(result["rejected_logps"], torch.tensor([5.0, 7.0]).cuda()))

    def test_tensor_cache_disk_roundtrip_with_gpu(self):
        cache = model_utils.TensorCache(tensors={"chosen_logps": torch.tensor([1.0, 2.0, 3.0]).cuda()})
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir) / "cache.pt"
            cache.to_disk(cache_path)
            loaded_cache = model_utils.TensorCache.from_disk(cache_path, torch.device("cuda"))
            self.assertEqual(loaded_cache.tensors["chosen_logps"].device.type, "cuda")
            self.assertTrue(torch.allclose(cache.tensors["chosen_logps"], loaded_cache.tensors["chosen_logps"]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestDataCollatorDatasetIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, torch_dtype=torch.bfloat16).cuda()

    def test_collator_preserves_index(self):
        samples = [
            {
                "chosen_input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "chosen_labels": [-100, -100, 3, 4, 5],
                "chosen_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "rejected_input_ids": torch.tensor([1, 2, 6, 7, 8]),
                "rejected_labels": [-100, -100, 6, 7, 8],
                "rejected_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "index": i,
            }
            for i in range(4)
        ]

        collator = dpo_utils.DataCollatorForSeq2SeqDPO(tokenizer=self.tokenizer, model=self.model, padding="longest")
        batch = collator(samples)

        self.assertIn("index", batch)
        self.assertTrue(torch.equal(batch["index"], torch.tensor([0, 1, 2, 3])))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestForwardFunctionsOlmo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.olmo_model = TransformerConfig.olmo2_30M(vocab_size=1000).build(init_device="cuda").to(torch.bfloat16)
        cls.olmo_model.lm_head.__class__ = DPOLMHead
        cls.hf_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.hf_model = AutoModelForCausalLM.from_pretrained(cls.hf_model_name, torch_dtype=torch.bfloat16).cuda()

    def _make_batch(self, batch_size: int = 2, seq_len: int = 10, vocab_size: int = 1000):
        return {
            "chosen_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
            "chosen_labels": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
            "chosen_attention_mask": torch.ones(batch_size, seq_len).cuda(),
            "rejected_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
            "rejected_labels": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
            "rejected_attention_mask": torch.ones(batch_size, seq_len).cuda(),
        }

    def test_concatenated_forward_olmo(self):
        batch = self._make_batch()
        chosen_logps, rejected_logps, aux_loss = dpo_utils.concatenated_forward_olmo(self.olmo_model, batch)

        self.assertEqual(chosen_logps.shape, (2,))
        self.assertEqual(rejected_logps.shape, (2,))
        self.assertIsNone(aux_loss)
        self.assertTrue(torch.isfinite(chosen_logps).all())
        self.assertTrue(torch.isfinite(rejected_logps).all())

    def test_separate_forward_olmo(self):
        batch = self._make_batch()
        chosen_logps, rejected_logps, aux_loss = dpo_utils.separate_forward_olmo(self.olmo_model, batch)

        self.assertEqual(chosen_logps.shape, (2,))
        self.assertEqual(rejected_logps.shape, (2,))
        self.assertIsNone(aux_loss)
        self.assertTrue(torch.isfinite(chosen_logps).all())
        self.assertTrue(torch.isfinite(rejected_logps).all())

    def test_concatenated_forward_hf(self):
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        batch = self._make_batch(vocab_size=tokenizer.vocab_size)
        chosen_logps, rejected_logps, aux_loss = dpo_utils.concatenated_forward(self.hf_model, batch)

        self.assertEqual(chosen_logps.shape, (2,))
        self.assertEqual(rejected_logps.shape, (2,))
        self.assertIsNone(aux_loss)
        self.assertTrue(torch.isfinite(chosen_logps).all())
        self.assertTrue(torch.isfinite(rejected_logps).all())

    def test_separate_forward_hf(self):
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        batch = self._make_batch(vocab_size=tokenizer.vocab_size)
        chosen_logps, rejected_logps, aux_loss = dpo_utils.separate_forward(self.hf_model, batch)

        self.assertEqual(chosen_logps.shape, (2,))
        self.assertEqual(rejected_logps.shape, (2,))
        self.assertIsNone(aux_loss)
        self.assertTrue(torch.isfinite(chosen_logps).all())
        self.assertTrue(torch.isfinite(rejected_logps).all())

    def test_token_logps_out_frame_matches_sequence_logps(self):
        # SafeComp ablations (2026-11-03): the chosen per-token log-probs exported for template logging / gated NLL are
        # in _get_batch_logps' frame (index i = log p(labels[:, i + 1])), stay in the autograd graph, and the default
        # call (no token_logps_out) is unchanged.
        batch = self._make_batch()
        batch["chosen_labels"][:, :3] = -100  # an unlabeled prompt
        for fn in (dpo_utils.separate_forward_olmo, dpo_utils.concatenated_forward_olmo):
            out = {}
            chosen, rejected, _ = fn(self.olmo_model, batch, average_log_prob=True, token_logps_out=out)
            ref_chosen, ref_rejected, _ = fn(self.olmo_model, batch, average_log_prob=True)
            torch.testing.assert_close(chosen, ref_chosen)
            torch.testing.assert_close(rejected, ref_rejected)
            tok, lab = out["chosen_token_logps"], out["chosen_labels"]
            self.assertEqual(tuple(tok.shape), (2, 9))
            torch.testing.assert_close(lab, batch["chosen_labels"])
            mask = lab[:, 1:] != -100
            torch.testing.assert_close((tok * mask).sum(-1) / mask.sum(-1), chosen)
            self.assertTrue(tok.requires_grad)
            self.olmo_model.zero_grad(set_to_none=True)
            dpo_utils.template_gated_nll(tok, mask, tau=1.01).backward()  # tau > 1: every masked position gated in
            grads = [p.grad for p in self.olmo_model.parameters() if p.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            self.olmo_model.zero_grad(set_to_none=True)

    def test_olmo_and_hf_produce_different_results(self):
        batch = self._make_batch()
        olmo_chosen, olmo_rejected, _ = dpo_utils.concatenated_forward_olmo(self.olmo_model, batch)

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        hf_batch = self._make_batch(vocab_size=tokenizer.vocab_size)
        hf_chosen, hf_rejected, _ = dpo_utils.concatenated_forward(self.hf_model, hf_batch)

        self.assertFalse(torch.allclose(olmo_chosen, hf_chosen))


if __name__ == "__main__":
    unittest.main()
