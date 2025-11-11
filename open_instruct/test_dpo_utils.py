import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Olmo3Config

from open_instruct.dpo_utils import DataCollatorForSeq2SeqDPO, concatenated_forward, separate_forward


class TestDPOForwardConsistency(unittest.TestCase):
    """Test that concatenated_forward and separate_forward produce equivalent results."""

    def setUp(self):
        """Set up a small model and tokenizer for testing."""
        # Use a small model config for testing
        self.config = Olmo3Config(vocab_size=1000, n_layer=2, n_embd=64, n_head=2)

        self.model = AutoModelForCausalLM.from_config(self.config)
        self.model.eval()  # Set to eval mode for consistent results

        # Use a real tokenizer, but ensure pad_token_id is within model vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Set pad_token_id to 0 (within model vocab_size) instead of eos_token
        self.tokenizer.pad_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2

    def create_test_batch(self, batch_size=2, chosen_length=10, rejected_length=12):
        """Create a test batch with chosen and rejected sequences."""
        # Create sequences with different lengths to test padding behavior
        # Use valid token IDs within the tokenizer's vocabulary
        vocab_size = min(self.config.vocab_size, self.tokenizer.vocab_size)
        chosen_input_ids = torch.randint(3, vocab_size - 1, (batch_size, chosen_length))
        rejected_input_ids = torch.randint(3, vocab_size - 1, (batch_size, rejected_length))

        # Create labels (mask prompt tokens with -100, keep response tokens)
        # For simplicity, mask first 3 tokens as prompt
        chosen_labels = chosen_input_ids.clone()
        chosen_labels[:, :3] = -100

        rejected_labels = rejected_input_ids.clone()
        rejected_labels[:, :3] = -100

        # Create attention masks
        chosen_attention_mask = torch.ones_like(chosen_input_ids)
        rejected_attention_mask = torch.ones_like(rejected_input_ids)

        batch = {
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_attention_mask,
        }

        return batch

    def test_forward_consistency_same_length(self):
        """Test that both forward methods produce the same results when sequences are same length."""
        batch = self.create_test_batch(batch_size=2, chosen_length=10, rejected_length=10)

        # Use DataCollator to ensure proper padding
        collator = DataCollatorForSeq2SeqDPO(tokenizer=self.tokenizer, model=self.model, padding="longest")
        # Convert batch to format expected by collator
        features = [
            {
                "chosen_input_ids": batch["chosen_input_ids"][i].tolist(),
                "chosen_labels": batch["chosen_labels"][i].tolist(),
                "chosen_attention_mask": batch["chosen_attention_mask"][i].tolist(),
                "rejected_input_ids": batch["rejected_input_ids"][i].tolist(),
                "rejected_labels": batch["rejected_labels"][i].tolist(),
                "rejected_attention_mask": batch["rejected_attention_mask"][i].tolist(),
            }
            for i in range(batch["chosen_input_ids"].shape[0])
        ]
        collated_batch = collator(features)

        with torch.no_grad():
            concat_chosen_logps, concat_rejected_logps, concat_aux_loss = concatenated_forward(
                self.model, collated_batch, average_log_prob=False
            )
            sep_chosen_logps, sep_rejected_logps, sep_aux_loss = separate_forward(
                self.model, collated_batch, average_log_prob=False
            )

        # Check that results are close (within numerical precision)
        # With the fix, they should be very close. Without the fix, there can be larger differences
        # due to extra padding affecting model computation
        torch.testing.assert_close(concat_chosen_logps, sep_chosen_logps, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(concat_rejected_logps, sep_rejected_logps, rtol=1e-4, atol=1e-5)
        self.assertEqual(concat_aux_loss, sep_aux_loss)

        # Also verify that the lengths match (this will fail without the fix)
        self.assertEqual(
            collated_batch["chosen_input_ids"].shape[1],
            collated_batch["rejected_input_ids"].shape[1],
            "Chosen and rejected should have same length for consistent forward passes",
        )  # Both should be None

    def test_forward_consistency_different_lengths(self):
        """Test that both forward methods produce the same results when sequences have different lengths."""
        batch = self.create_test_batch(batch_size=2, chosen_length=8, rejected_length=12)

        collator = DataCollatorForSeq2SeqDPO(tokenizer=self.tokenizer, model=self.model, padding="longest")
        features = [
            {
                "chosen_input_ids": batch["chosen_input_ids"][i].tolist(),
                "chosen_labels": batch["chosen_labels"][i].tolist(),
                "chosen_attention_mask": batch["chosen_attention_mask"][i].tolist(),
                "rejected_input_ids": batch["rejected_input_ids"][i].tolist(),
                "rejected_labels": batch["rejected_labels"][i].tolist(),
                "rejected_attention_mask": batch["rejected_attention_mask"][i].tolist(),
            }
            for i in range(batch["chosen_input_ids"].shape[0])
        ]
        collated_batch = collator(features)

        with torch.no_grad():
            concat_chosen_logps, concat_rejected_logps, concat_aux_loss = concatenated_forward(
                self.model, collated_batch, average_log_prob=False
            )
            sep_chosen_logps, sep_rejected_logps, sep_aux_loss = separate_forward(
                self.model, collated_batch, average_log_prob=False
            )

        # Check that results are close (within numerical precision)
        # With the fix, they should be very close. Without the fix, there can be larger differences
        # due to extra padding affecting model computation
        torch.testing.assert_close(concat_chosen_logps, sep_chosen_logps, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(concat_rejected_logps, sep_rejected_logps, rtol=1e-4, atol=1e-5)
        self.assertEqual(concat_aux_loss, sep_aux_loss)

    def test_forward_consistency_average_log_prob(self):
        """Test consistency when using average_log_prob=True."""
        batch = self.create_test_batch(batch_size=2, chosen_length=10, rejected_length=12)

        collator = DataCollatorForSeq2SeqDPO(tokenizer=self.tokenizer, model=self.model, padding="longest")
        features = [
            {
                "chosen_input_ids": batch["chosen_input_ids"][i].tolist(),
                "chosen_labels": batch["chosen_labels"][i].tolist(),
                "chosen_attention_mask": batch["chosen_attention_mask"][i].tolist(),
                "rejected_input_ids": batch["rejected_input_ids"][i].tolist(),
                "rejected_labels": batch["rejected_labels"][i].tolist(),
                "rejected_attention_mask": batch["rejected_attention_mask"][i].tolist(),
            }
            for i in range(batch["chosen_input_ids"].shape[0])
        ]
        collated_batch = collator(features)

        with torch.no_grad():
            concat_chosen_logps, concat_rejected_logps, concat_aux_loss = concatenated_forward(
                self.model, collated_batch, average_log_prob=True
            )
            sep_chosen_logps, sep_rejected_logps, sep_aux_loss = separate_forward(
                self.model, collated_batch, average_log_prob=True
            )

        # Check that results are close
        torch.testing.assert_close(concat_chosen_logps, sep_chosen_logps, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(concat_rejected_logps, sep_rejected_logps, rtol=1e-4, atol=1e-5)
        self.assertEqual(concat_aux_loss, sep_aux_loss)

    def test_collator_pads_to_same_length(self):  # fails
        """Test that DataCollatorForSeq2SeqDPO pads chosen and rejected to the same length."""
        batch = self.create_test_batch(batch_size=2, chosen_length=8, rejected_length=12)

        collator = DataCollatorForSeq2SeqDPO(tokenizer=self.tokenizer, model=self.model, padding="longest")
        features = [
            {
                "chosen_input_ids": batch["chosen_input_ids"][i].tolist(),
                "chosen_labels": batch["chosen_labels"][i].tolist(),
                "chosen_attention_mask": batch["chosen_attention_mask"][i].tolist(),
                "rejected_input_ids": batch["rejected_input_ids"][i].tolist(),
                "rejected_labels": batch["rejected_labels"][i].tolist(),
                "rejected_attention_mask": batch["rejected_attention_mask"][i].tolist(),
            }
            for i in range(batch["chosen_input_ids"].shape[0])
        ]
        collated_batch = collator(features)

        # Check that chosen and rejected have the same length after collation
        chosen_length = collated_batch["chosen_input_ids"].shape[1]
        rejected_length = collated_batch["rejected_input_ids"].shape[1]
        self.assertEqual(
            chosen_length, rejected_length, "Chosen and rejected sequences should be padded to the same length"
        )  # AssertionError: 8 != 12 : Chosen and rejected sequences should be padded to the same length

        # Check that labels are also the same length
        self.assertEqual(
            collated_batch["chosen_labels"].shape[1],
            collated_batch["rejected_labels"].shape[1],
            "Chosen and rejected labels should be padded to the same length",
        )
