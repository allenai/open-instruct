"""GPU integration tests for DPO utils including TensorCache.

These tests require CUDA and will be skipped if not available.

To run:
    ./scripts/train/build_image_and_launch.sh scripts/train/debug/run_gpu_pytest.sh
"""

import pathlib
import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct.dpo_utils import DataCollatorForSeq2SeqDPO
from open_instruct.model_utils import TensorCache


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTensorCacheGPU(unittest.TestCase):
    def test_tensor_cache_gpu_indexing(self):
        cache = TensorCache(
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
        cache = TensorCache(tensors={"chosen_logps": torch.tensor([1.0, 2.0, 3.0]).cuda()})
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir) / "cache.pt"
            cache.to_disk(cache_path)
            loaded_cache = TensorCache.from_disk(cache_path)
            self.assertEqual(loaded_cache.tensors["chosen_logps"].device.type, "cpu")
            self.assertTrue(torch.allclose(cache.tensors["chosen_logps"].cpu(), loaded_cache.tensors["chosen_logps"]))


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
                "chosen_labels": torch.tensor([-100, -100, 3, 4, 5]),
                "chosen_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "rejected_input_ids": torch.tensor([1, 2, 6, 7, 8]),
                "rejected_labels": torch.tensor([-100, -100, 6, 7, 8]),
                "rejected_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "index": i,
            }
            for i in range(4)
        ]

        collator = DataCollatorForSeq2SeqDPO(tokenizer=self.tokenizer, model=self.model, padding="longest")
        batch = collator(samples)

        self.assertIn("index", batch)
        self.assertTrue(torch.equal(batch["index"], torch.tensor([0, 1, 2, 3])))


if __name__ == "__main__":
    unittest.main()
