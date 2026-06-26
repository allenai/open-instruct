import unittest

import torch

from open_instruct.distillkit.losses import forward_kl_topk_from_logprobs
from open_instruct.distillkit.signals import SparseTeacherSignal


class TestForwardKLTopK(unittest.TestCase):
    def test_unnormalized_forward_kl_matches_hand_computation(self):
        teacher_logprobs = torch.log(torch.tensor([[[0.5, 0.25]]], dtype=torch.float32))
        student_logprobs = torch.log(torch.tensor([[[0.25, 0.5]]], dtype=torch.float32))
        signal = SparseTeacherSignal(token_ids=torch.tensor([[[3, 5]]], dtype=torch.long), logprobs=teacher_logprobs)

        output = forward_kl_topk_from_logprobs(student_logprobs, signal)

        expected = 0.5 * (torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.25)))
        expected += 0.25 * (torch.log(torch.tensor(0.25)) - torch.log(torch.tensor(0.5)))
        torch.testing.assert_close(output.loss, expected.reshape(1, 1))
        torch.testing.assert_close(output.teacher_topk_mass, torch.tensor([[0.75]]))

    def test_missing_entries_contribute_zero(self):
        signal = SparseTeacherSignal(
            token_ids=torch.tensor([[[1, 0]]], dtype=torch.long),
            logprobs=torch.tensor([[[-0.5, float("-inf")]]], dtype=torch.float32),
        )
        student_logprobs = torch.tensor([[[-0.25, -100.0]]], dtype=torch.float32)

        output = forward_kl_topk_from_logprobs(student_logprobs, signal)

        expected = torch.exp(torch.tensor(-0.5)) * (-0.5 + 0.25)
        torch.testing.assert_close(output.loss, expected.reshape(1, 1))
        torch.testing.assert_close(output.teacher_topk_mass, torch.exp(torch.tensor([[-0.5]])))

    def test_missing_entry_with_neg_inf_student_logprob_stays_finite(self):
        # Regression: at a missing teacher slot (-inf) the student logprob may also be
        # -inf; teacher_probs is 0 there, so without masking the student term this would
        # compute 0 * inf = NaN.
        signal = SparseTeacherSignal(
            token_ids=torch.tensor([[[1, 0]]], dtype=torch.long),
            logprobs=torch.tensor([[[-0.5, float("-inf")]]], dtype=torch.float32),
        )
        student_logprobs = torch.tensor([[[-0.25, float("-inf")]]], dtype=torch.float32)

        output = forward_kl_topk_from_logprobs(student_logprobs, signal)

        expected = torch.exp(torch.tensor(-0.5)) * (-0.5 + 0.25)
        self.assertTrue(torch.isfinite(output.loss).all())
        torch.testing.assert_close(output.loss, expected.reshape(1, 1))

    def test_normalized_topk_changes_mass_to_one_in_loss(self):
        signal = SparseTeacherSignal(
            token_ids=torch.tensor([[[1, 2]]], dtype=torch.long),
            logprobs=torch.log(torch.tensor([[[0.2, 0.3]]], dtype=torch.float32)),
        )
        student_logprobs = torch.log(torch.tensor([[[0.4, 0.6]]], dtype=torch.float32))

        output = forward_kl_topk_from_logprobs(student_logprobs, signal, normalize_topk=True)

        torch.testing.assert_close(output.loss, torch.zeros((1, 1)))
        torch.testing.assert_close(output.teacher_topk_mass, torch.tensor([[0.5]]))


if __name__ == "__main__":
    unittest.main()
