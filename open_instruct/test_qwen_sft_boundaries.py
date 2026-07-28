"""Regression tests for opt-in SFT record-boundary handling."""

import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from olmo_core.data.utils import get_document_lengths, iter_document_indices

from open_instruct import dataset_transformation, olmo_core_utils


class _FakeSftTokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self, conversation, *, add_generation_prompt=False, truncation=False, max_length=None, **_kwargs
    ):
        token_ids = []
        for message in conversation:
            if message["role"] == "user":
                token_ids.extend([10, 11])
            elif message["role"] == "assistant":
                token_ids.extend([20, 30, 30, 30, 30, 30, 30, 30, self.eos_token_id])
            else:
                raise ValueError(f"Unsupported test role: {message['role']}")
        if add_generation_prompt:
            token_ids.append(20)
        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]
        return torch.tensor([token_ids], dtype=torch.long)


class _FakeBoundaryTokenizer:
    def __init__(self, *, eos_token_id, bos_token_id, pad_token_id, vocab_size, encodings=None):
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self._encodings = encodings or {}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return self._encodings.get(text, [7, 8])


class TerminalEosTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _FakeSftTokenizer()
        self.row = {
            "messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "long answer"}]
        }

    def _tokenize(self, **kwargs):
        tokenized = (
            torch.tensor([[10, 11, 20, 30, 30, 30]], dtype=torch.long),
            torch.ones((1, 6), dtype=torch.long),
            torch.tensor([[-100, -100, 20, 30, 30, 30]], dtype=torch.long),
        )
        with mock.patch.object(
            dataset_transformation, "_tokenize_tulu_sft_with_assistant_labels", return_value=tokenized
        ):
            return dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
                copy.deepcopy(self.row), self.tokenizer, max_seq_length=6, **kwargs
            )

    def test_default_matches_explicit_false(self):
        default = self._tokenize()
        explicit_false = self._tokenize(ensure_terminal_eos_after_truncation=False)

        for key in dataset_transformation.TOKENIZED_SFT_DATASET_KEYS:
            torch.testing.assert_close(default[key], explicit_false[key])
        self.assertNotEqual(default[dataset_transformation.INPUT_IDS_KEY][-1].item(), self.tokenizer.eos_token_id)

    def test_opt_in_replaces_last_truncated_token_with_trainable_eos(self):
        tokenized = self._tokenize(ensure_terminal_eos_after_truncation=True)

        self.assertEqual(len(tokenized[dataset_transformation.INPUT_IDS_KEY]), 6)
        self.assertEqual(tokenized[dataset_transformation.INPUT_IDS_KEY][-1].item(), self.tokenizer.eos_token_id)
        self.assertEqual(tokenized[dataset_transformation.LABELS_KEY][-1].item(), self.tokenizer.eos_token_id)

    def test_opt_in_requires_eos_token(self):
        self.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, "requires a tokenizer EOS token"):
            self._tokenize(ensure_terminal_eos_after_truncation=True)


class DocumentBoundaryTest(unittest.TestCase):
    QWEN_IM_START = 151644
    QWEN_IM_END = 151645

    def _qwen_tokenizer_config(self):
        tokenizer = _FakeBoundaryTokenizer(
            eos_token_id=self.QWEN_IM_END,
            bos_token_id=None,
            pad_token_id=151643,
            vocab_size=151936,
            encodings={"<|im_start|>": [self.QWEN_IM_START]},
        )
        return SimpleNamespace(tokenizer_name_or_path="Qwen/Qwen3-30B-A3B", tokenizer=tokenizer)

    def test_qwen_boundary_hint_does_not_mutate_hf_tokenizer(self):
        tc = self._qwen_tokenizer_config()
        config = olmo_core_utils.to_oc_tokenizer_config(tc, document_boundary_start_token="<|im_start|>")

        self.assertEqual(config.bos_token_id, self.QWEN_IM_START)
        self.assertIsNone(tc.tokenizer.bos_token_id)

    def test_qwen_internal_eos_tokens_do_not_split_packed_records(self):
        tc = self._qwen_tokenizer_config()
        config = olmo_core_utils.to_oc_tokenizer_config(tc, document_boundary_start_token="<|im_start|>")
        newline = 198
        document_a = [self.QWEN_IM_START, 10, self.QWEN_IM_END, newline, self.QWEN_IM_START, 11, self.QWEN_IM_END]
        document_b = [self.QWEN_IM_START, 20, self.QWEN_IM_END, newline, self.QWEN_IM_START, 21, self.QWEN_IM_END]
        packed = np.asarray(document_a + document_b, dtype=np.uint32)

        lengths = get_document_lengths(packed, config.eos_token_id, config.bos_token_id)
        self.assertEqual(lengths.tolist(), [len(document_a), len(document_b)])

        with tempfile.TemporaryDirectory() as tmp:
            data_path = os.path.join(tmp, "token_ids.npy")
            packed.tofile(data_path)
            indices = list(
                iter_document_indices(
                    data_path,
                    use_array_if_local=True,
                    eos_token_id=config.eos_token_id,
                    bos_token_id=config.bos_token_id,
                    dtype=np.uint32,
                )
            )
        self.assertEqual(indices, [(0, len(document_a)), (len(document_a), len(packed))])

    def test_existing_olmo_eos_only_behavior_is_unchanged(self):
        eos_id = 100257
        tokenizer = _FakeBoundaryTokenizer(
            eos_token_id=eos_id, bos_token_id=eos_id, pad_token_id=100277, vocab_size=100278
        )
        tc = SimpleNamespace(tokenizer_name_or_path="allenai/olmo-3-tokenizer", tokenizer=tokenizer)

        config = olmo_core_utils.to_oc_tokenizer_config(tc)

        self.assertIsNone(config.bos_token_id)
        self.assertEqual(config.eos_token_id, eos_id)
        self.assertEqual(tokenizer.bos_token_id, eos_id)
        packed = np.asarray([100264, 1, 100265, 2, eos_id, 100264, 3, 100265, 4, eos_id], dtype=np.uint32)
        self.assertEqual(get_document_lengths(packed, config.eos_token_id, config.bos_token_id).tolist(), [5, 5])

    def test_boundary_start_marker_must_be_one_token(self):
        tc = self._qwen_tokenizer_config()
        with self.assertRaisesRegex(ValueError, "must encode to exactly one token"):
            olmo_core_utils.to_oc_tokenizer_config(tc, document_boundary_start_token="not-special")


if __name__ == "__main__":
    unittest.main()
