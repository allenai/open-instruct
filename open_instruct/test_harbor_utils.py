"""Tests for Harbor integration utilities."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from open_instruct.data_types import EnvConfig, GenerationResult, RequestInfo
from open_instruct.model_utils import Batch

try:
    from open_instruct.harbor_utils import harbor_output_to_completions
    from open_instruct.vllm_utils import CompletionOutput, RequestOutput, process_completed_request

    _vllm_available = True
except ImportError:
    _vllm_available = False


@unittest.skipUnless(_vllm_available, "requires vllm")
class TestHarborOutputToCompletions(unittest.TestCase):
    """Tests for harbor_output_to_completions."""

    def setUp(self):
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = 0

    def _make_trial_result(self, rollout_details, reward=1.0):
        result = MagicMock()
        result.verifier_result = MagicMock()
        result.verifier_result.rewards = {"reward": reward}
        result.agent_result = MagicMock()
        result.agent_result.rollout_details = rollout_details
        return result

    def test_single_segment_single_turn(self):
        rollout_details = [
            {
                "prompt_token_ids": [[10, 20, 30]],
                "completion_token_ids": [[100, 101, 102]],
                "logprobs": [[-0.5, -0.3, -0.1]],
            }
        ]
        trial_result = self._make_trial_result(rollout_details, reward=0.75)

        completions, reward = harbor_output_to_completions(trial_result, self.tokenizer, max_seq_len=1024)

        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].token_ids, [100, 101, 102])
        self.assertEqual(completions[0].logprobs, [-0.5, -0.3, -0.1])
        self.assertEqual(completions[0].mask, [1, 1, 1])
        self.assertEqual(completions[0].finish_reason, "stop")
        self.assertAlmostEqual(reward, 0.75)
        self.assertAlmostEqual(completions[0].rollout_state["harbor_reward"], 0.75)

    def test_single_segment_multi_turn(self):
        rollout_details = [
            {
                "prompt_token_ids": [[10, 20, 30], [10, 20, 30, 100, 101, 200, 201]],
                "completion_token_ids": [[100, 101], [300, 301]],
                "logprobs": [[-0.5, -0.3], [-0.2, -0.4]],
            }
        ]
        trial_result = self._make_trial_result(rollout_details, reward=1.0)

        completions, reward = harbor_output_to_completions(trial_result, self.tokenizer, max_seq_len=1024)

        self.assertEqual(len(completions), 1)
        comp = completions[0]
        # Turn 1 assistant tokens: [100, 101]
        # Tool/user tokens between turns: [200, 201]
        # Turn 2 assistant tokens: [300, 301]
        self.assertEqual(comp.token_ids, [100, 101, 200, 201, 300, 301])
        self.assertEqual(comp.logprobs, [-0.5, -0.3, 0.0, 0.0, -0.2, -0.4])
        self.assertEqual(comp.mask, [1, 1, 0, 0, 1, 1])
        self.assertEqual(comp.finish_reason, "stop")

    def test_multiple_segments(self):
        rollout_details = [
            {"prompt_token_ids": [[10, 20]], "completion_token_ids": [[100]], "logprobs": [[-0.5]]},
            {"prompt_token_ids": [[50, 60]], "completion_token_ids": [[200]], "logprobs": [[-0.3]]},
        ]
        trial_result = self._make_trial_result(rollout_details, reward=0.5)

        completions, reward = harbor_output_to_completions(trial_result, self.tokenizer, max_seq_len=1024)

        self.assertEqual(len(completions), 2)
        self.assertEqual(completions[0].finish_reason, "summarized")
        self.assertEqual(completions[1].finish_reason, "stop")
        self.assertAlmostEqual(reward, 0.5)
        for comp in completions:
            self.assertAlmostEqual(comp.rollout_state["harbor_reward"], 0.5)

    def test_empty_rollout_details(self):
        trial_result = self._make_trial_result(rollout_details=None, reward=0.0)
        trial_result.agent_result.rollout_details = None

        completions, reward = harbor_output_to_completions(trial_result, self.tokenizer, max_seq_len=1024)

        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].token_ids, [0])
        self.assertEqual(completions[0].finish_reason, "stop")

    def test_truncation(self):
        rollout_details = [
            {
                "prompt_token_ids": [[10]],
                "completion_token_ids": [[100, 101, 102, 103, 104]],
                "logprobs": [[-0.1, -0.2, -0.3, -0.4, -0.5]],
            }
        ]
        trial_result = self._make_trial_result(rollout_details)

        completions, _ = harbor_output_to_completions(trial_result, self.tokenizer, max_seq_len=3)

        self.assertEqual(len(completions[0].token_ids), 3)
        self.assertEqual(len(completions[0].logprobs), 3)
        self.assertEqual(len(completions[0].mask), 3)


class TestIDBasedAdvantageNormalization(unittest.TestCase):
    """Tests for ID-based advantage normalization matching the reshape approach."""

    def test_uniform_groups_matches_reshape(self):
        """When all groups have equal size, ID-based matches reshape-based."""
        n_prompts = 4
        n_per_prompt = 3
        np.random.seed(42)
        scores = np.random.rand(n_prompts * n_per_prompt)

        # Reshape-based (original)
        scores_per_prompt = scores.reshape(-1, n_per_prompt)
        expected_mean = np.repeat(scores_per_prompt.mean(axis=-1), n_per_prompt)
        expected_std = np.repeat(scores_per_prompt.std(axis=-1), n_per_prompt)

        # ID-based
        group_ids = []
        for i in range(n_prompts):
            group_ids.extend([f"prompt_{i}"] * n_per_prompt)

        unique_groups = list(dict.fromkeys(group_ids))
        group_indices = {g: [] for g in unique_groups}
        for i, gid in enumerate(group_ids):
            group_indices[gid].append(i)

        id_mean = np.zeros(len(scores))
        id_std = np.zeros(len(scores))
        for _gid, indices in group_indices.items():
            group_scores = scores[indices]
            id_mean[indices] = group_scores.mean()
            id_std[indices] = group_scores.std()

        np.testing.assert_allclose(id_mean, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(id_std, expected_std, rtol=1e-10)

    def test_variable_group_sizes(self):
        """Variable group sizes compute correct per-group statistics."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        group_ids = ["a", "a", "b", "b", "b", "c", "c"]

        unique_groups = list(dict.fromkeys(group_ids))
        group_indices = {g: [] for g in unique_groups}
        for i, gid in enumerate(group_ids):
            group_indices[gid].append(i)

        mean_rewards = np.zeros(len(scores))
        std_rewards = np.zeros(len(scores))
        for _gid, indices in group_indices.items():
            group_scores = scores[indices]
            mean_rewards[indices] = group_scores.mean()
            std_rewards[indices] = group_scores.std()

        # Group "a": [1, 2] → mean=1.5
        np.testing.assert_allclose(mean_rewards[0], 1.5)
        np.testing.assert_allclose(mean_rewards[1], 1.5)
        # Group "b": [3, 4, 5] → mean=4.0
        np.testing.assert_allclose(mean_rewards[2], 4.0)
        np.testing.assert_allclose(mean_rewards[3], 4.0)
        np.testing.assert_allclose(mean_rewards[4], 4.0)
        # Group "c": [6, 7] → mean=6.5
        np.testing.assert_allclose(mean_rewards[5], 6.5)
        np.testing.assert_allclose(mean_rewards[6], 6.5)

        # Centered advantages
        advantages = scores - mean_rewards
        np.testing.assert_allclose(advantages[0], -0.5)
        np.testing.assert_allclose(advantages[1], 0.5)
        np.testing.assert_allclose(advantages[2], -1.0)
        np.testing.assert_allclose(advantages[3], 0.0)
        np.testing.assert_allclose(advantages[4], 1.0)

    def test_single_element_groups(self):
        """Single-element groups have zero std and zero centered advantage."""
        scores = np.array([5.0, 3.0, 8.0])
        group_ids = ["a", "b", "c"]

        unique_groups = list(dict.fromkeys(group_ids))
        group_indices = {g: [] for g in unique_groups}
        for i, gid in enumerate(group_ids):
            group_indices[gid].append(i)

        mean_rewards = np.zeros(len(scores))
        for _gid, indices in group_indices.items():
            mean_rewards[indices] = scores[indices].mean()

        advantages = scores - mean_rewards
        np.testing.assert_allclose(advantages, [0.0, 0.0, 0.0])


class TestBatchGroupIds(unittest.TestCase):
    """Tests for Batch slicing with group_ids field."""

    def _make_batch(self, n, group_ids=None):
        return Batch(
            queries=[[i] for i in range(n)],
            ground_truths=[[i] for i in range(n)],
            datasets=[f"ds_{i}" for i in range(n)],
            raw_queries=[f"q_{i}" for i in range(n)],
            decoded_responses=[f"r_{i}" for i in range(n)],
            indices=list(range(n)),
            scores=[float(i) for i in range(n)],
            group_ids=group_ids,
        )

    def test_slice_with_group_ids(self):
        batch = self._make_batch(6, group_ids=["a", "a", "b", "b", "c", "c"])
        sliced = batch[2:4]
        self.assertEqual(sliced.group_ids, ["b", "b"])

    def test_index_with_group_ids(self):
        batch = self._make_batch(4, group_ids=["x", "y", "x", "y"])
        single = batch[1]
        self.assertEqual(single.group_ids, ["y"])

    def test_list_index_with_group_ids(self):
        batch = self._make_batch(5, group_ids=["a", "b", "c", "d", "e"])
        selected = batch[[0, 3, 4]]
        self.assertEqual(selected.group_ids, ["a", "d", "e"])

    def test_slice_without_group_ids(self):
        batch = self._make_batch(4)
        sliced = batch[1:3]
        self.assertIsNone(sliced.group_ids)


@unittest.skipUnless(_vllm_available, "requires vllm")
class TestHarborRewardsInFinalize(unittest.TestCase):
    """Tests that Harbor rewards flow through the finalize pipeline."""

    def test_process_completed_request_preserves_rollout_state(self):
        outs = [
            RequestOutput(
                request_id="req_0_0",
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    CompletionOutput(
                        index=0,
                        token_ids=[10, 11],
                        logprobs=[-0.5, -0.3],
                        finish_reason="stop",
                        mask=[1, 1],
                        rollout_state={"harbor_reward": 0.8, "step_count": 2},
                    )
                ],
            ),
            RequestOutput(
                request_id="req_0_1",
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    CompletionOutput(
                        index=0,
                        token_ids=[20, 21],
                        logprobs=[-0.2, -0.4],
                        finish_reason="stop",
                        mask=[1, 1],
                        rollout_state={"harbor_reward": 0.8, "step_count": 3},
                    )
                ],
            ),
        ]

        metadata = {
            "req_0": {
                "is_eval": False,
                "index": 0,
                "prompt_id": "prompt_0",
                "prompt_token_ids": [1, 2, 3],
                "start_time": 0.0,
            }
        }

        result, is_eval = process_completed_request("req_0", outs, 1.0, True, metadata)

        self.assertEqual(len(result.request_info.rollout_states), 2)
        self.assertEqual(result.request_info.rollout_states[0]["harbor_reward"], 0.8)
        self.assertEqual(result.request_info.rollout_states[1]["harbor_reward"], 0.8)


class TestGenerationResultGroupId(unittest.TestCase):
    """Tests for group_id on GenerationResult."""

    def test_group_id_default_none(self):
        result = GenerationResult(
            responses=[[1, 2]],
            finish_reasons=["stop"],
            masks=[[1, 1]],
            request_info=RequestInfo(
                num_calls=[0],
                timeouts=[False],
                tool_errors=[""],
                tool_outputs=[""],
                tool_runtimes=[0.0],
                tool_calleds=[False],
            ),
            index=0,
            prompt_id="p0",
        )
        self.assertIsNone(result.group_id)

    def test_group_id_set(self):
        result = GenerationResult(
            responses=[[1, 2]],
            finish_reasons=["stop"],
            masks=[[1, 1]],
            request_info=RequestInfo(
                num_calls=[0],
                timeouts=[False],
                tool_errors=[""],
                tool_outputs=[""],
                tool_runtimes=[0.0],
                tool_calleds=[False],
            ),
            index=0,
            prompt_id="p0",
            group_id="group_abc",
        )
        self.assertEqual(result.group_id, "group_abc")


class TestEnvConfigHarborTaskPath(unittest.TestCase):
    """Tests for harbor_task_path on EnvConfig."""

    def test_default_none(self):
        cfg = EnvConfig()
        self.assertIsNone(cfg.harbor_task_path)

    def test_set_path(self):
        cfg = EnvConfig(harbor_task_path="/tasks/fix-bug")
        self.assertEqual(cfg.harbor_task_path, "/tasks/fix-bug")


if __name__ == "__main__":
    unittest.main()
