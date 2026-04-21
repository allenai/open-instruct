import tempfile
import unittest
from queue import Queue

import numpy as np
import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader, data_types, utils
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO


def _make_dpo_dataset(num_samples: int, max_seq_length: int) -> Dataset:
    rng = torch.Generator().manual_seed(42)
    data = {
        "chosen_input_ids": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_labels": [],
        "index": list(range(num_samples)),
    }
    for _ in range(num_samples):
        chosen_len = torch.randint(1, max_seq_length + 1, (1,), generator=rng).item()
        rejected_len = torch.randint(1, max_seq_length + 1, (1,), generator=rng).item()
        data["chosen_input_ids"].append(torch.randint(0, 1000, (chosen_len,), generator=rng))
        data["chosen_labels"].append(torch.randint(0, 1000, (chosen_len,), generator=rng))
        data["rejected_input_ids"].append(torch.randint(0, 1000, (rejected_len,), generator=rng))
        data["rejected_labels"].append(torch.randint(0, 1000, (rejected_len,), generator=rng))
    ds = Dataset.from_dict(data)
    ds.set_format(type="pt")
    return ds


def _make_rlvr_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            INPUT_IDS_PROMPT_KEY: [[1, 2, 3]],
            GROUND_TRUTHS_KEY: ["answer"],
            VERIFIER_SOURCE_KEY: ["dataset_0"],
            RAW_PROMPT_KEY: ["prompt"],
            "index": [0],
        }
    )


def _make_generation_result(
    prompt_id: str, reward_scores: list[float], model_step: int
) -> data_types.GenerationResult:
    total_responses = len(reward_scores)
    return data_types.GenerationResult(
        responses=[[1, 2, 3] for _ in range(total_responses)],
        finish_reasons=["stop"] * total_responses,
        masks=[[1, 1, 1] for _ in range(total_responses)],
        request_info=data_types.RequestInfo(
            num_calls=[0] * total_responses,
            timeouts=[0] * total_responses,
            tool_errors=[""] * total_responses,
            tool_outputs=[""] * total_responses,
            tool_runtimes=[0.0] * total_responses,
            tool_calleds=[False] * total_responses,
        ),
        index=0,
        prompt_id=prompt_id,
        token_statistics=data_types.TokenStatistics(
            num_prompt_tokens=10, num_response_tokens=3 * total_responses, generation_time=0.1
        ),
        start_time=0.0,
        logprobs=[[0.0, 0.0, 0.0] for _ in range(total_responses)],
        reward_scores=reward_scores,
        reward_metrics={"time/reward": 0.0},
        model_step=model_step,
    )


class _FakeTokenizer:
    eos_token_id = 0

    def batch_decode(self, responses, skip_special_tokens=False):
        return [" ".join(map(str, response)) for response in responses]


class TestWorldAwarePacking(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("olmo3_7b_dp2", 16384, 8, 2, True, 200),
            ("olmo3_7b_dp4", 16384, 16, 4, True, 200),
            ("olmo3_32b_dp4", 8192, 8, 4, True, 200),
            ("olmo3_32b_dp8", 8192, 16, 8, True, 200),
            ("debug_multi_node", 16384, 32, 2, True, 200),
            ("olmo3_7b_dp2_no_drop", 16384, 8, 2, False, 200),
            ("olmo3_32b_dp4_no_drop", 8192, 8, 4, False, 200),
        ]
    )
    def test_packing_equal_batches_across_ranks(
        self, _name, max_seq_length, global_batch_size, dp_world_size, drop_last, num_samples
    ):
        dataset = _make_dpo_dataset(num_samples, max_seq_length)
        collator = TensorDataCollatorWithFlatteningDPO(max_seq_length=max_seq_length)

        with tempfile.TemporaryDirectory() as work_dir:
            loaders = [
                data_loader.HFDataLoader(
                    dataset=dataset,
                    batch_size=global_batch_size,
                    seed=42,
                    dp_rank=rank,
                    dp_world_size=dp_world_size,
                    work_dir=work_dir,
                    collator=collator,
                    drop_last=drop_last,
                )
                for rank in range(dp_world_size)
            ]

            batch_counts = [loader.total_batches for loader in loaders]
            self.assertTrue(
                all(c == batch_counts[0] for c in batch_counts), f"Batch counts differ across ranks: {batch_counts}"
            )

            all_indices = set()
            for loader in loaders:
                for batch in loader:
                    if "index" in batch:
                        all_indices.update(batch["index"].tolist())

            if not drop_last:
                expected_indices = set(range(num_samples))
                self.assertEqual(all_indices, expected_indices, f"Missing indices: {expected_indices - all_indices}")


class TestGroupedAdvantageScales(unittest.TestCase):
    def test_expand_grouped_advantage_scales_uses_effective_baseline_counts(self):
        advantage_scales = data_loader.expand_grouped_advantage_scales(
            prompt_sample_counts=[4], prompt_baseline_sample_counts=[8]
        )

        self.assertTrue(np.allclose(advantage_scales, np.full(4, 0.5)))


class TestGroupedAdvantages(unittest.TestCase):
    def test_compute_grouped_advantages_rescales_by_baseline_counts(self):
        advantages = data_loader.compute_grouped_advantages(
            np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
            prompt_sample_counts=[4],
            prompt_baseline_sample_counts=[8],
            prompt_baseline_reward_sums=[2.0],
            advantage_normalization_type="centered",
            ngu_count_rescale="ratio",
        )

        self.assertTrue(np.allclose(advantages, np.array([-0.125, 0.375, -0.125, 0.375], dtype=float)))

    def test_compute_grouped_advantages_can_disable_baseline_count_rescale(self):
        advantages = data_loader.compute_grouped_advantages(
            np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
            prompt_sample_counts=[4],
            prompt_baseline_sample_counts=[8],
            prompt_baseline_reward_sums=[2.0],
            advantage_normalization_type="centered",
            ngu_count_rescale=None,
        )

        self.assertTrue(np.allclose(advantages, np.array([-0.25, 0.75, -0.25, 0.75], dtype=float)))

    def test_compute_grouped_advantages_max_rl_divides_by_mean(self):
        advantages = data_loader.compute_grouped_advantages(
            np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
            prompt_sample_counts=[4],
            prompt_baseline_sample_counts=[8],
            prompt_baseline_reward_sums=[2.0],
            advantage_normalization_type="max_rl",
        )

        self.assertTrue(np.allclose(advantages, np.array([-1.0, 3.0, -1.0, 3.0], dtype=float), atol=1e-6))

    def test_compute_grouped_advantages_anchor_pos_matches_centered_when_baseline_matches_batch(self):
        """When baseline mean equals batch mean, advantages already sum to zero; anchor scaling is identity."""
        scores = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        centered = data_loader.compute_grouped_advantages(
            scores, prompt_sample_counts=[4], advantage_normalization_type="centered", ngu_count_rescale=None
        )
        anchored = data_loader.compute_grouped_advantages(
            scores, prompt_sample_counts=[4], advantage_normalization_type="centered", ngu_count_rescale="anchor_pos"
        )
        self.assertTrue(np.allclose(anchored, centered))

    def test_compute_grouped_advantages_anchor_pos_rescales_when_ngu_baseline_differs(self):
        """With inflated baseline counts, centered advantages need not sum to zero; anchor_pos zeros the group sum."""
        advantages = data_loader.compute_grouped_advantages(
            np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
            prompt_sample_counts=[4],
            prompt_baseline_sample_counts=[8],
            prompt_baseline_reward_sums=[2.0],
            advantage_normalization_type="centered",
            ngu_count_rescale="anchor_pos",
        )
        self.assertTrue(np.allclose(advantages, np.array([-0.75, 0.75, -0.75, 0.75], dtype=float)))
        self.assertAlmostEqual(float(advantages.sum()), 0.0, places=6)


class TestNeverGiveUpPendingAge(unittest.TestCase):
    def test_streaming_config_defaults_pending_age_to_two(self):
        config = data_loader.StreamingDataLoaderConfig()
        self.assertEqual(config.maintain_pending_ngu_age, 2)
        self.assertIsNone(config.maintain_pending_ngu_count_rescale)

    def test_accumulate_inference_batches_drops_stale_pending_counts(self):
        inference_results_Q = Queue(maxsize=1)
        inference_results_Q.put(_make_generation_result("0_0_1", [0.0, 1.0, 0.0, 1.0], model_step=3))

        never_give_up_state = data_loader.NeverGiveUpAccumulationState(
            pending_response_counts={"0_0": 4},
            pending_reward_sums={"0_0": 0.0},
            pending_best_reward={"0_0": 0.0},
            pending_last_model_step={"0_0": 0},
        )

        _, batch, _, batch_stats = data_loader.accumulate_inference_batches(
            inference_results_Q=inference_results_Q,
            generation_config=type("GenerationConfig", (), {"n": 4})(),
            num_prompts=1,
            model_dims=utils.ModelDims(
                num_layers=1, hidden_size=1, intermediate_size=1, vocab_size=1, num_attn_heads=1, head_dim=1
            ),
            tokenizer=_FakeTokenizer(),
            dataset=_make_rlvr_dataset(),
            base_env_config=data_types.EnvConfig(),
            active_sampling=True,
            filter_zero_std_samples=True,
            never_give_up=1.0,
            show_progress_bar=False,
            never_give_up_state=never_give_up_state,
            maintain_pending_ngu_age=2,
            maintain_pending_ngu_counts=True,
        )

        self.assertEqual(batch.scores, [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(batch_stats.prompt_sample_counts, [4])
        self.assertEqual(batch_stats.prompt_baseline_sample_counts, [4])
        self.assertEqual(batch_stats.prompt_baseline_reward_sums, [2.0])
        self.assertEqual(never_give_up_state.pending_response_counts, {})
        self.assertEqual(never_give_up_state.pending_reward_sums, {})


if __name__ == "__main__":
    unittest.main()
