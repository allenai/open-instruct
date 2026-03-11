"""GPU integration tests for DPO utils including TensorCache.

These tests require CUDA and will be skipped if not available.

To run:
    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh
"""

import os
import pathlib
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from olmo_core.nn.transformer import TransformerConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct import data_loader as data_loader_lib
from open_instruct import dpo_utils, model_utils, utils
from open_instruct.olmo_core_train_modules import DPOLMHead
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO


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

    def test_olmo_and_hf_produce_different_results(self):
        batch = self._make_batch()
        olmo_chosen, olmo_rejected, _ = dpo_utils.concatenated_forward_olmo(self.olmo_model, batch)

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        hf_batch = self._make_batch(vocab_size=tokenizer.vocab_size)
        hf_chosen, hf_rejected, _ = dpo_utils.concatenated_forward(self.hf_model, hf_batch)

        self.assertFalse(torch.allclose(olmo_chosen, hf_chosen))


def _make_dpo_dataset(num_samples: int, vocab_size: int, length_range: tuple[int, int]) -> Dataset:
    """Create a synthetic DPO dataset with variable-length sequences."""
    rng = torch.Generator().manual_seed(42)
    min_len, max_len = length_range
    data = {
        "chosen_input_ids": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_labels": [],
        "index": list(range(num_samples)),
    }
    for _ in range(num_samples):
        chosen_len = torch.randint(min_len, max_len + 1, (1,), generator=rng).item()
        rejected_len = torch.randint(min_len, max_len + 1, (1,), generator=rng).item()
        data["chosen_input_ids"].append(torch.randint(0, vocab_size, (chosen_len,), generator=rng))
        data["chosen_labels"].append(torch.randint(0, vocab_size, (chosen_len,), generator=rng))
        data["rejected_input_ids"].append(torch.randint(0, vocab_size, (rejected_len,), generator=rng))
        data["rejected_labels"].append(torch.randint(0, vocab_size, (rejected_len,), generator=rng))
    return Dataset.from_dict(data)


def _run_cache_builder(rank, world_size, dataset, model_config, max_seq_length, cache_dir, results_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device = torch.device("cuda:0")
    model = model_config.build(init_device="cuda").to(torch.bfloat16)
    model.lm_head.__class__ = DPOLMHead

    collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)
    per_rank_batch_size = 4
    global_batch_size = per_rank_batch_size * world_size

    dl = data_loader_lib.HFDataLoader(
        dataset=dataset,
        batch_size=global_batch_size,
        seed=42,
        dp_rank=rank,
        dp_world_size=world_size,
        work_dir=cache_dir,
        collator=collator,
        device=device,
        drop_last=False,
    )

    model_dims = utils.ModelDims(
        num_layers=1, hidden_size=128, intermediate_size=256, vocab_size=1000, num_attn_heads=2, head_dim=64
    )

    cache_path = pathlib.Path(cache_dir) / f"cache_rank{rank}.pt"
    cache = dpo_utils.build_reference_logprobs_cache(
        model=model,
        dataloader=dl,
        average_log_prob=False,
        forward_fn=dpo_utils.concatenated_forward_olmo,
        forward_kwargs={"packing": True},
        full_dataset_size=len(dataset),
        device=device,
        cache_path=cache_path,
        is_main_process=(rank == 0),
        model_dims=model_dims,
    )

    missing_chosen = torch.where(cache.tensors["chosen_logps"] == float("-inf"))[0]
    missing_rejected = torch.where(cache.tensors["rejected_logps"] == float("-inf"))[0]
    results_dict[rank] = {
        "num_batches": dl.total_batches,
        "missing_chosen": missing_chosen.tolist(),
        "missing_rejected": missing_rejected.tolist(),
        "chosen_logps": cache.tensors["chosen_logps"].cpu(),
        "rejected_logps": cache.tensors["rejected_logps"].cpu(),
    }

    dist.destroy_process_group()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestBuildReferenceCacheMultiRank(unittest.TestCase):
    def test_world_aware_packing_cache_no_missing_indices(self):
        vocab_size = 1000
        num_samples = 50
        max_seq_length = 200
        world_size = 2

        dataset = _make_dpo_dataset(num_samples, vocab_size, length_range=(20, 120))
        model_config = TransformerConfig.olmo2_30M(vocab_size=vocab_size)

        manager = mp.Manager()
        results_dict = manager.dict()

        with tempfile.TemporaryDirectory() as cache_dir:
            mp.spawn(
                _run_cache_builder,
                args=(world_size, dataset, model_config, max_seq_length, cache_dir, results_dict),
                nprocs=world_size,
                join=True,
            )

        for rank in range(world_size):
            result = results_dict[rank]
            self.assertEqual(result["missing_chosen"], [], f"Rank {rank} has missing chosen indices")
            self.assertEqual(result["missing_rejected"], [], f"Rank {rank} has missing rejected indices")

        torch.testing.assert_close(results_dict[0]["chosen_logps"], results_dict[1]["chosen_logps"])
        torch.testing.assert_close(results_dict[0]["rejected_logps"], results_dict[1]["rejected_logps"])


if __name__ == "__main__":
    unittest.main()
