import random
import unittest

import torch

from open_instruct import subtb_utils


class TestSubTBUtils(unittest.TestCase):
    def test_compute_subtb_g_values_matches_manual_formula(self):
        logits = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]]], dtype=torch.float32)
        tokens = torch.tensor([[0, 1]])

        actual = subtb_utils.compute_subtb_g_values(logits, tokens, q=0.5, alpha=1.0, omega=1.0)

        lp_policy = torch.log_softmax(1.5 * logits, dim=-1)
        lp_sharp = torch.log_softmax(logits, dim=-1)
        expected = lp_policy.gather(-1, tokens.unsqueeze(-1)).squeeze(-1) - 0.5 * lp_sharp.gather(
            -1, tokens.unsqueeze(-1)
        ).squeeze(-1)

        torch.testing.assert_close(actual, expected)

    def test_split_response_positions_uses_done_boundaries(self):
        response_mask = torch.tensor([0, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
        dones = torch.tensor([0, 0, 1, 0, 0, 0, 1], dtype=torch.bool)

        sequences = subtb_utils.split_response_positions(response_mask, dones)

        self.assertEqual([seq.tolist() for seq in sequences], [[1, 2], [4, 5, 6]])

    def test_sample_subtb_windows_includes_terminal_window(self):
        windows = subtb_utils.sample_subtb_windows(
            sequence_length=10,
            num_windows=4,
            min_window_size=2,
            max_window_size=5,
            include_terminal_window=True,
            rng=random.Random(7),
        )

        self.assertEqual(len(windows), 4)
        self.assertTrue(any(end == 10 for _, end in windows))
        for start, end in windows:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, 10)
            self.assertGreater(end, start)

    def test_compute_subtb_loss_zero_when_constraints_match(self):
        flow_values = torch.tensor([1.0, 0.75, 0.25], dtype=torch.float32)
        g_values = torch.tensor([-0.25, -0.50, 0.0], dtype=torch.float32)
        response_mask = torch.tensor([1, 1, 0], dtype=torch.bool)
        dones = torch.tensor([0, 1, 0], dtype=torch.bool)
        rewards = torch.tensor([0.0, 0.25, 0.0], dtype=torch.float32)

        loss, stats = subtb_utils.compute_subtb_loss(
            flow_values=flow_values,
            g_values=g_values,
            response_mask=response_mask,
            dones=dones,
            rewards=rewards,
            reward_scale=1.0,
            num_windows=3,
            min_window_size=1,
            max_window_size=2,
            lambda_decay=0.9,
            rng=random.Random(0),
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertGreater(stats["num_windows"], 0.0)
        self.assertAlmostEqual(stats["mean_abs_residual"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
