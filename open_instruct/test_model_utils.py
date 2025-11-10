import unittest

import torch

import open_instruct.model_utils


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
