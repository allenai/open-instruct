import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import open_instruct.model_utils
from open_instruct.model_utils import Batch, TensorCache, uses_olmo3_generation_config


class TestBatchSlicing(unittest.TestCase):
    def test_batch_slicing_with_all_fields(self):
        batch = Batch(
            queries=[[1, 2], [3, 4], [5, 6]],
            ground_truths=[[7, 8], [9, 10], [11, 12]],
            datasets=["ds1", "ds2", "ds3"],
            raw_queries=["q1", "q2", "q3"],
            decoded_responses=["r1", "r2", "r3"],
            indices=[0, 1, 2],
            scores=[0.1, 0.2, 0.3],
            model_steps=[0, 0, 0],
        )

        sliced = batch[[0, 2]]
        self.assertEqual(len(sliced.queries), 2)
        self.assertEqual(sliced.queries, [[1, 2], [5, 6]])
        self.assertEqual(sliced.decoded_responses, ["r1", "r3"])
        self.assertEqual(sliced.scores, [0.1, 0.3])

    def test_batch_slicing_with_none_fields(self):
        batch = Batch(
            queries=[[1, 2], [3, 4], [5, 6]],
            ground_truths=[[7, 8], [9, 10], [11, 12]],
            datasets=["ds1", "ds2", "ds3"],
            raw_queries=None,
            decoded_responses=None,
            indices=None,
            scores=None,
            model_steps=[0, 0, 0],
        )

        sliced = batch[[0, 2]]
        self.assertEqual(len(sliced.queries), 2)
        self.assertEqual(sliced.queries, [[1, 2], [5, 6]])
        self.assertIsNone(sliced.decoded_responses)
        self.assertIsNone(sliced.scores)


class TestLogSoftmaxAndGather(unittest.TestCase):
    def test_log_softmax_and_gather_sliced_logits(self):
        batch_size, seq_len, vocab_size = 2, 160, 151936
        logits_full = torch.randn(batch_size, seq_len + 1, vocab_size)
        logits = logits_full[:, :-1, :]
        index_full = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
        index = index_full[:, 1:].clone()

        self.assertFalse(logits.is_contiguous())
        self.assertTrue(index.is_contiguous())

        result = open_instruct.model_utils.log_softmax_and_gather(logits, index)

        self.assertEqual(result.shape, (batch_size, seq_len))
        self.assertTrue(torch.all(result <= 0))
        self.assertTrue(torch.all(torch.isfinite(result)))


class TestTensorCache(unittest.TestCase):
    def test_getitem_returns_correct_tensors(self):
        chosen_logps = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rejected_logps = torch.tensor([[0.5, 1.5], [2.5, 3.5]])

        cache = TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        result = cache[torch.tensor([0])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[0.5, 1.5]])))

        result = cache[torch.tensor([1])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[3.0, 4.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[2.5, 3.5]])))

    def test_getitem_with_multiple_indices(self):
        chosen_logps = torch.tensor([[1.0], [2.0], [3.0]])
        rejected_logps = torch.tensor([[0.5], [1.5], [2.5]])

        cache = TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        result = cache[torch.tensor([0, 2])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[1.0], [3.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[0.5], [2.5]])))

    def test_to_disk_and_from_disk(self):
        chosen_logps = torch.tensor([1.0, 2.0, 3.0])
        rejected_logps = torch.tensor([0.5, 1.5, 2.5])

        cache = TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir) / "cache.pt"
            cache.to_disk(cache_path)

            self.assertTrue(cache_path.exists())

            loaded_cache = TensorCache.from_disk(cache_path, device="cpu")

            self.assertTrue(torch.allclose(loaded_cache.tensors["chosen_logps"], chosen_logps))
            self.assertTrue(torch.allclose(loaded_cache.tensors["rejected_logps"], rejected_logps))

    def test_from_disk_preserves_indexing(self):
        chosen_logps = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rejected_logps = torch.tensor([0.1, 0.2, 0.3, 0.4])

        cache = TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir) / "cache.pt"
            cache.to_disk(cache_path)
            loaded_cache = TensorCache.from_disk(cache_path, device="cpu")

            result = loaded_cache[torch.tensor([1, 3])]
            self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([2.0, 4.0])))
            self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([0.2, 0.4])))


class TestUsesOlmo3GenerationConfig(unittest.TestCase):
    def test_olmo_template_name(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "User: {{ content }}"
        self.assertTrue(uses_olmo3_generation_config("olmo123", tokenizer))

    def test_hybrid_model_type_with_tokenizer_default(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        model = SimpleNamespace(config=SimpleNamespace(model_type="olmo_hybrid"))
        self.assertTrue(uses_olmo3_generation_config("tokenizer_default", tokenizer, model))

    def test_olmo_tokenizer_with_resolved_im_end_template(self):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "allenai/olmo-3-tokenizer-instruct-dev"
        tokenizer.chat_template = "<|im_start|>assistant\n{{ content }}<|im_end|>"
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="/weka/HYBRID_INSTRUCT_SFT"))
        self.assertTrue(uses_olmo3_generation_config("tokenizer_default", tokenizer, model))

    def test_non_olmo_chatml_template_does_not_use_olmo_generation_config(self):
        tokenizer = MagicMock()
        tokenizer.name_or_path = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer.chat_template = "<|im_start|>assistant\n{{ content }}<|im_end|>"
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen2"))
        self.assertFalse(uses_olmo3_generation_config("tokenizer_default", tokenizer, model))

    def test_wrapped_olmo_hybrid_model_is_detected(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        model = SimpleNamespace(module=SimpleNamespace(config=SimpleNamespace(model_type="olmo_hybrid")))
        self.assertTrue(uses_olmo3_generation_config("tokenizer_default", tokenizer, model))

    def test_non_olmo_template_without_im_end(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "{{ messages }}"
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
        self.assertFalse(uses_olmo3_generation_config("tulu", tokenizer, model))
        self.assertFalse(uses_olmo3_generation_config("tokenizer_default", tokenizer, model))


if __name__ == "__main__":
    unittest.main()
