import unittest

import torch

from open_instruct.distillkit.losses import forward_kl_topk_from_logprobs


class TestForwardKLTopK(unittest.TestCase):
    def test_unnormalized_forward_kl_matches_hand_computation(self):
        teacher_logprobs = torch.log(torch.tensor([[[0.5, 0.25]]], dtype=torch.float32))
        student_logprobs = torch.log(torch.tensor([[[0.25, 0.5]]], dtype=torch.float32))

        output = forward_kl_topk_from_logprobs(student_logprobs, teacher_logprobs)

        expected = 0.5 * (torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.25)))
        expected += 0.25 * (torch.log(torch.tensor(0.25)) - torch.log(torch.tensor(0.5)))
        torch.testing.assert_close(output.loss, expected.reshape(1, 1))
        torch.testing.assert_close(output.teacher_topk_mass, torch.tensor([[0.75]]))

    def test_missing_entries_contribute_zero(self):
        teacher_logprobs = torch.tensor([[[-0.5, float("-inf")]]], dtype=torch.float32)
        student_logprobs = torch.tensor([[[-0.25, -100.0]]], dtype=torch.float32)

        output = forward_kl_topk_from_logprobs(student_logprobs, teacher_logprobs)

        expected = torch.exp(torch.tensor(-0.5)) * (-0.5 + 0.25)
        torch.testing.assert_close(output.loss, expected.reshape(1, 1))
        torch.testing.assert_close(output.teacher_topk_mass, torch.exp(torch.tensor([[-0.5]])))

    def test_missing_entry_with_neg_inf_student_logprob_stays_finite(self):
        # Regression: at a missing teacher slot (-inf) the student logprob may also be
        # -inf; teacher_probs is 0 there, so without masking the student term this would
        # compute 0 * inf = NaN.
        teacher_logprobs = torch.tensor([[[-0.5, float("-inf")]]], dtype=torch.float32)
        student_logprobs = torch.tensor([[[-0.25, float("-inf")]]], dtype=torch.float32)

        output = forward_kl_topk_from_logprobs(student_logprobs, teacher_logprobs)

        expected = torch.exp(torch.tensor(-0.5)) * (-0.5 + 0.25)
        self.assertTrue(torch.isfinite(output.loss).all())
        torch.testing.assert_close(output.loss, expected.reshape(1, 1))


if __name__ == "__main__":
    unittest.main()
