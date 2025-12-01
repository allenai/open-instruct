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


class TestMaskedMean(unittest.TestCase):
    def test_original_axis_int(self):
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        result = open_instruct.model_utils.masked_mean(values, mask, axis=1)
        expected = ((1.0 + 2.0) / 2 + 4.0 / 1) / 2
        self.assertAlmostEqual(result.item(), expected)

    def test_original_axis_none(self):
        values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        result = open_instruct.model_utils.masked_mean(values, mask, axis=None)
        expected = (1.0 + 2.0 + 4.0) / 3
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_vectorized_axis_int(self):
        kl_4BT = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
                [[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]],
            ]
        )
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        result = open_instruct.model_utils.masked_mean(kl_4BT, mask, axis=1)
        self.assertEqual(result.shape, (4,))
        expected_0 = ((1.0 + 2.0) / 2 + 4.0 / 1) / 2
        expected_1 = ((10.0 + 20.0) / 2 + 40.0 / 1) / 2
        expected_2 = ((100.0 + 200.0) / 2 + 400.0 / 1) / 2
        expected_3 = ((1000.0 + 2000.0) / 2 + 4000.0 / 1) / 2
        self.assertAlmostEqual(result[0].item(), expected_0)
        self.assertAlmostEqual(result[1].item(), expected_1)
        self.assertAlmostEqual(result[2].item(), expected_2)
        self.assertAlmostEqual(result[3].item(), expected_3)

    def test_vectorized_axis_none(self):
        kl_4BT = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                [[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]],
                [[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]],
            ]
        )
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        result = open_instruct.model_utils.masked_mean(kl_4BT, mask, axis=None)
        self.assertEqual(result.shape, (4,))
        expected = torch.tensor(
            [
                (1.0 + 2.0 + 4.0) / 3,
                (10.0 + 20.0 + 40.0) / 3,
                (100.0 + 200.0 + 400.0) / 3,
                (1000.0 + 2000.0 + 4000.0) / 3,
            ]
        )
        self.assertTrue(torch.allclose(result, expected))


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
