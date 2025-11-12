import shutil
import tempfile
import time
import unittest

import numpy as np
import torch
import transformers

from open_instruct import rl_utils

PACK_LENGTH = 40
PROMPT_MAX_LEN = 20
GENERATE_MAX_LEN = 20
GAMMA = 1.0
LAMBDA = 1.0
MODEL_NAME = "EleutherAI/pythia-14m"


def get_test_data():
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    prompts = [
        "User: Hello, how are you?\nAssistant: <think>",
        "User: What is the capital of France?\nAssistant: <think>",
        "User: What is the capital of Germany?\nAssistant: <think>",
    ]
    outputs = ["I'm good, thank you!", "Paris", "Berlin"]
    queries = [tokenizer.encode(prompt) for prompt in prompts]
    responses = [tokenizer.encode(response) for response in outputs]
    assert all(len(query) <= PROMPT_MAX_LEN for query in queries)
    assert all(len(response) <= GENERATE_MAX_LEN for response in responses)
    return queries, responses, tokenizer.pad_token_id


class TestRLUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_timer_context_manager(self):
        """Test Timer works as a context manager and measures time."""
        with self.assertLogs("open_instruct.rl_utils", level="INFO") as cm:  # noqa: SIM117 - nested to capture Timer's log output
            with rl_utils.Timer("test operation"):
                time.sleep(0.1)

        self.assertEqual(len(cm.output), 1)
        self.assertIn("test operation", cm.output[0])
        self.assertIn("seconds", cm.output[0])
        log_message = cm.output[0]
        duration = float(log_message.split(": ")[1].split(" ")[0])
        self.assertGreaterEqual(duration, 0.1)
        self.assertLess(duration, 0.2)

    def test_timer_decorator(self):
        """Test Timer works as a decorator and measures time."""

        @rl_utils.Timer("decorated function")
        def slow_function():
            time.sleep(0.1)
            return 42

        with self.assertLogs("open_instruct.rl_utils", level="INFO") as cm:
            result = slow_function()

        self.assertEqual(result, 42)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("decorated function", cm.output[0])
        self.assertIn("seconds", cm.output[0])
        log_message = cm.output[0]
        duration = float(log_message.split(": ")[1].split(" ")[0])
        self.assertGreaterEqual(duration, 0.1)
        self.assertLess(duration, 0.2)

    def test_timer_noop(self):
        """Test Timer with noop=True does not log."""
        with self.assertNoLogs("open_instruct.rl_utils", level="INFO"):  # noqa: SIM117 - nested to verify no log output
            with rl_utils.Timer("should not log", noop=True):
                time.sleep(0.05)

    def test_timer_decorator_noop(self):
        """Test Timer decorator with noop=True does not log."""

        @rl_utils.Timer("should not log", noop=True)
        def silent_function():
            time.sleep(0.05)
            return "done"

        with self.assertNoLogs("open_instruct.rl_utils", level="INFO"):
            result = silent_function()

        self.assertEqual(result, "done")

    def test_pack_sequences(self):
        """Test that pack_sequences correctly concatenates queries and responses into packed format."""
        queries, responses, pad_token_id = get_test_data()
        masks = [[1] * len(response) for response in responses]
        vllm_logprobs = [[0.0] * len(response) for response in responses]
        with rl_utils.Timer("pack_sequences"):
            packed_sequences = rl_utils.pack_sequences(
                queries=queries,
                responses=responses,
                masks=masks,
                pack_length=PACK_LENGTH,
                pad_token_id=pad_token_id,
                vllm_logprobs=vllm_logprobs,
            )

        self._assert_sequence_packed_correctly(packed_sequences, queries, responses, 0, 0, 0)
        self._assert_sequence_packed_correctly(
            packed_sequences, queries, responses, 0, 1, len(queries[0]) + len(responses[0])
        )
        self._assert_sequence_packed_correctly(packed_sequences, queries, responses, 1, 2, 0)

    def _assert_sequence_packed_correctly(self, packed_sequences, queries, responses, pack_idx, seq_idx, offset):
        query = queries[seq_idx]
        response = responses[seq_idx]
        expected = np.array(query + response)
        sequence_length = len(query) + len(response)
        actual = packed_sequences.query_responses[pack_idx][offset : offset + sequence_length]
        np.testing.assert_allclose(actual, expected)

    def test_calculate_advantages_packed(self):
        """Test that calculate_advantages_packed produces same results as unpacked version."""
        _, _, pad_token_id = get_test_data()

        unpacked_values = np.array(
            [
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ]
        )
        unpacked_rewards = np.zeros((3, 13))
        unpacked_rewards[0, 5] = 10
        unpacked_rewards[1, 6] = 20
        unpacked_rewards[2, 12] = 30
        unpacked_advantages, unpacked_returns = rl_utils.calculate_advantages(
            unpacked_values, unpacked_rewards, GAMMA, LAMBDA
        )

        packed_values, packed_rewards, packed_dones, packed_response_masks = self._create_packed_arrays_with_padding()
        packed_values_masked = np.where(packed_response_masks[:, 1:] == 0, 0, packed_values)

        packed_advantages, packed_returns = rl_utils.calculate_advantages_packed(
            packed_values_masked,
            packed_rewards[:, 1:],
            GAMMA,
            LAMBDA,
            packed_dones[:, 1:],
            packed_response_masks[:, 1:],
        )

        packed_values_v2, packed_rewards_v2, packed_dones_v2, packed_response_masks_v2 = (
            self._create_packed_arrays_no_padding(pad_token_id)
        )
        packed_advantages_v2, packed_returns_v2 = rl_utils.calculate_advantages_packed(
            packed_values_v2, packed_rewards_v2, GAMMA, LAMBDA, packed_dones_v2, packed_response_masks_v2
        )

        np.testing.assert_allclose(unpacked_advantages[0, :5], packed_advantages_v2[0, 4:9])
        np.testing.assert_allclose(unpacked_returns[1, :6], packed_returns_v2[0, 12:18])
        np.testing.assert_allclose(unpacked_advantages[2, :12], packed_advantages_v2[0, 24:36])

    def _create_packed_arrays_with_padding(self):
        packed_response_masks = np.zeros((1, 42), dtype=int)
        packed_response_masks[0, 5:11] = 1
        packed_response_masks[0, 15:22] = 1
        packed_response_masks[0, 29:42] = 1

        packed_values = np.full((1, 41), 5)
        packed_values[0, 4:10] = 1
        packed_values[0, 14:21] = 2
        packed_values[0, 28:41] = 3

        packed_rewards = np.zeros((1, 42))
        packed_rewards[0, 10] = 10
        packed_rewards[0, 21] = 20
        packed_rewards[0, 41] = 30

        packed_dones = np.zeros((1, 42))
        packed_dones[0, 10] = 1
        packed_dones[0, 21] = 2
        packed_dones[0, 41] = 3

        return packed_values, packed_rewards, packed_dones, packed_response_masks

    def _create_packed_arrays_no_padding(self, pad_token_id):
        packed_values = np.full((1, 37), -1)
        packed_values[0, 4:9] = 1
        packed_values[0, 12:18] = 2
        packed_values[0, 24:36] = 3
        packed_values[0, 36] = pad_token_id

        packed_rewards = np.zeros((1, 37))
        packed_rewards[0, 8] = 10
        packed_rewards[0, 17] = 20
        packed_rewards[0, 35] = 30
        packed_rewards[0, 36] = pad_token_id

        packed_dones = np.zeros((1, 37))
        packed_dones[0, 8] = 1
        packed_dones[0, 17] = 1
        packed_dones[0, 35] = 1
        packed_dones[0, 36] = pad_token_id

        packed_response_masks = np.zeros((1, 37), dtype=int)
        packed_response_masks[0, 4:9] = 1
        packed_response_masks[0, 12:18] = 1
        packed_response_masks[0, 24:36] = 1
        packed_response_masks[0, 36] = pad_token_id

        return packed_values, packed_rewards, packed_dones, packed_response_masks

    def test_pack_sequences_logits(self):
        """Test packed sequence processing with value model and advantage calculation."""
        queries, responses, pad_token_id = get_test_data()

        value_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=1, torch_dtype=torch.bfloat16
        )
        value_model.train()
        value_head = value_model.score
        config = value_model.config
        torch.manual_seed(2)
        torch.nn.init.normal_(value_head.weight, std=1 / (config.hidden_size + 1))

        masks = [[1] * len(response) for response in responses]
        vllm_logprobs = [[0.0] * len(response) for response in responses]
        with rl_utils.Timer("pack_sequences"):
            packed_sequences = rl_utils.pack_sequences(
                queries=queries,
                responses=responses,
                masks=masks,
                pack_length=PACK_LENGTH,
                pad_token_id=pad_token_id,
                vllm_logprobs=vllm_logprobs,
            )

        lm_backbone = getattr(value_model, value_model.base_model_prefix)
        torch.manual_seed(2)
        value_model_output = lm_backbone(
            input_ids=packed_sequences.query_responses[0].unsqueeze(0),
            attention_mask=packed_sequences.attention_masks[0].unsqueeze(0).clamp(0, 1),
            position_ids=packed_sequences.position_ids[0].unsqueeze(0),
        )
        values = value_head(value_model_output.last_hidden_state).squeeze(-1)
        values = torch.where(packed_sequences.response_masks[0].unsqueeze(0) == 0, 0, values)

        rewards = np.zeros_like(values.detach().float().numpy())
        rewards[:, 21] = 0.1
        rewards[:, -1] = 1
        advantages, returns = rl_utils.calculate_advantages_packed(
            values=values.detach().float().numpy(),
            rewards=rewards,
            gamma=GAMMA,
            lam=LAMBDA,
            dones=packed_sequences.dones[0].unsqueeze(0).numpy(),
            response_masks=packed_sequences.response_masks[0].unsqueeze(0).numpy(),
        )


if __name__ == "__main__":
    unittest.main()
