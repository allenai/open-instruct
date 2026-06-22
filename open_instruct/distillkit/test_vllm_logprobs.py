import unittest
from dataclasses import dataclass

import torch

from open_instruct.distillkit.vllm_logprobs import extract_response_topk_from_prompt_logprobs


@dataclass
class FakeFlatLogprobs:
    start_indices: list[int]
    end_indices: list[int]
    ranks: list[int | None]
    token_ids: list[int]
    logprobs: list[float]

    def __len__(self):
        return len(self.start_indices)


class TestExtractResponseTopK(unittest.TestCase):
    def test_extracts_response_rows_when_first_prompt_entry_is_empty(self):
        # Combined token sequence length is 5: prompt length 3 + response length 2.
        # vLLM often emits an empty row for token 0 and then rows for tokens 1..4.
        prompt_logprobs = FakeFlatLogprobs(
            start_indices=[0, 0, 2, 4, 6],
            end_indices=[0, 2, 4, 6, 8],
            ranks=[1, 2, 1, 2, 1, 2, 1, 2],
            token_ids=[10, 11, 20, 21, 30, 31, 40, 41],
            logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        )

        token_ids, logprobs = extract_response_topk_from_prompt_logprobs(
            prompt_logprobs, prompt_len=3, response_len=2, k=2
        )

        torch.testing.assert_close(token_ids, torch.tensor([[30, 31], [40, 41]]))
        torch.testing.assert_close(logprobs, torch.tensor([[-0.5, -0.6], [-0.7, -0.8]]))

    def test_extracts_response_rows_when_all_positions_are_present(self):
        prompt_logprobs = FakeFlatLogprobs(
            start_indices=[0, 2, 4, 6],
            end_indices=[2, 4, 6, 8],
            ranks=[1, 2, 1, 2, 1, 2, 1, 2],
            token_ids=[0, 1, 10, 11, 20, 21, 30, 31],
            logprobs=[-1.0, -1.1, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
        )

        token_ids, logprobs = extract_response_topk_from_prompt_logprobs(
            prompt_logprobs, prompt_len=2, response_len=2, k=2
        )

        torch.testing.assert_close(token_ids, torch.tensor([[20, 21], [30, 31]]))
        torch.testing.assert_close(logprobs, torch.tensor([[-0.3, -0.4], [-0.5, -0.6]]))


if __name__ == "__main__":
    unittest.main()
