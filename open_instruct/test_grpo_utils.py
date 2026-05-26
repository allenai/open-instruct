import unittest

import parameterized
import torch

from open_instruct import olmo_core_utils


class ComputeOlmoCoreDocLensTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (
                "single_doc_no_pad",
                torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
                torch.tensor([[5]], dtype=torch.int32),
                [5],
            ),
            (
                "two_docs_no_pad",
                torch.tensor([[1, 1, 1, 2, 2]], dtype=torch.long),
                torch.tensor([[3, 2]], dtype=torch.int32),
                [3],
            ),
            (
                "three_docs_with_pad",
                torch.tensor([[1, 1, 2, 2, 2, 3, 0, 0]], dtype=torch.long),
                torch.tensor([[2, 3, 1, 2]], dtype=torch.int32),
                [3],
            ),
            ("all_pad", torch.tensor([[0, 0, 0]], dtype=torch.long), torch.tensor([[3]], dtype=torch.int32), [3]),
        ]
    )
    def test_single_row(self, _name, attention_mask, expected_doc_lens, expected_max):
        doc_lens, max_doc_lens = olmo_core_utils.doc_lens_from_attention_mask(attention_mask)
        torch.testing.assert_close(doc_lens, expected_doc_lens)
        self.assertEqual(max_doc_lens, expected_max)

    def test_batch_padding_to_max_docs(self):
        attention_mask = torch.tensor([[1, 1, 1, 2, 2, 0, 0], [1, 1, 2, 2, 3, 3, 3]], dtype=torch.long)
        doc_lens, max_doc_lens = olmo_core_utils.doc_lens_from_attention_mask(attention_mask)
        expected = torch.tensor([[3, 2, 2], [2, 2, 3]], dtype=torch.int32)
        torch.testing.assert_close(doc_lens, expected)
        self.assertEqual(max_doc_lens, [3, 3])

    def test_row_sums_equal_seq_len(self):
        attention_mask = torch.tensor([[1, 1, 1, 2, 2, 0, 0], [1, 1, 2, 2, 3, 3, 3]], dtype=torch.long)
        doc_lens, _ = olmo_core_utils.doc_lens_from_attention_mask(attention_mask)
        torch.testing.assert_close(doc_lens.sum(dim=1, dtype=torch.int32), torch.tensor([7, 7], dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
