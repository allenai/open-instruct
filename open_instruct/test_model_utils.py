import unittest

import torch

import open_instruct.model_utils
from open_instruct.model_utils import Batch


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


if __name__ == "__main__":
    unittest.main()
