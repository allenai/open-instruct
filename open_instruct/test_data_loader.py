import tempfile
import unittest
from queue import Queue
from unittest.mock import Mock

import numpy as np
import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader
from open_instruct.data_loader_utils import (
    compute_grouped_advantages,
    get_never_give_up_chain_id,
    get_never_give_up_retry_suffix,
)
from open_instruct.data_types import EnvConfig, GenerationResult, RequestInfo, TokenStatistics
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


class TestGroupedAdvantages(unittest.TestCase):
    def test_compute_grouped_advantages_anchor_pos_matches_centered_when_baseline_matches_batch(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        centered = compute_grouped_advantages(
            scores, prompt_sample_counts=[4], advantage_normalization_type="centered"
        )
        anchored = compute_grouped_advantages(
            scores, prompt_sample_counts=[4], advantage_normalization_type="centered", ngu_count_rescale="anchor_pos"
        )

        self.assertTrue(np.allclose(anchored, centered))

    def test_compute_grouped_advantages_anchor_pos_rescales_when_ngu_baseline_differs(self):
        scores = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        anchored = compute_grouped_advantages(
            scores,
            prompt_sample_counts=[3],
            prompt_baseline_sample_counts=[5],
            prompt_baseline_reward_sums=[1.0],
            advantage_normalization_type="centered",
            ngu_count_rescale="anchor_pos",
        )

        self.assertTrue(np.allclose(anchored.sum(), 0.0))
        self.assertTrue(np.allclose(anchored[-1], 0.8))

    def test_compute_grouped_advantages_ignores_ngu_baseline_when_disabled(self):
        scores = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # use_ngu_baseline=True applies the NGU baseline (rescales the advantages).
        with_baseline = compute_grouped_advantages(
            scores,
            prompt_sample_counts=[3],
            prompt_baseline_sample_counts=[5],
            prompt_baseline_reward_sums=[1.0],
            advantage_normalization_type="centered",
            use_ngu_baseline=True,
        )
        # use_ngu_baseline=False falls back to the regular grouped mean.
        keep_all = compute_grouped_advantages(
            scores,
            prompt_sample_counts=[3],
            prompt_baseline_sample_counts=[5],
            prompt_baseline_reward_sums=[1.0],
            advantage_normalization_type="centered",
            use_ngu_baseline=False,
        )
        regular = compute_grouped_advantages(scores, prompt_sample_counts=[3], advantage_normalization_type="centered")

        self.assertTrue(np.allclose(keep_all, regular))
        self.assertFalse(np.allclose(with_baseline, regular))

    def test_get_never_give_up_retry_suffix_increments_existing_suffix(self):
        self.assertEqual(get_never_give_up_retry_suffix("7_0", epoch_number=7, index=0), "_1")
        self.assertEqual(get_never_give_up_retry_suffix("7_0_1", epoch_number=7, index=0), "_2")

    def test_get_never_give_up_chain_id_strips_retry_suffix(self):
        self.assertEqual(get_never_give_up_chain_id("7_0"), "7_0")
        self.assertEqual(get_never_give_up_chain_id("7_0_1"), "7_0")

    def test_accumulate_inference_batches_merges_never_give_up_retry(self):
        class MockTokenizer:
            eos_token_id = 0

            def batch_decode(self, responses, skip_special_tokens=False):
                return [str(response) for response in responses]

        def make_result(prompt_id, reward_scores):
            return GenerationResult(
                responses=[[1], [2]],
                finish_reasons=["stop", "stop"],
                masks=[[1], [1]],
                request_info=RequestInfo(
                    num_calls=[0, 0],
                    timeouts=[0, 0],
                    tool_errors=["", ""],
                    tool_outputs=["", ""],
                    tool_runtimes=[0.0, 0.0],
                    tool_calleds=[False, False],
                    tool_call_stats=[[], []],
                    rollout_states=[{}, {}],
                ),
                index=0,
                prompt_id=prompt_id,
                token_statistics=TokenStatistics(num_prompt_tokens=1, num_response_tokens=2, generation_time=1.0),
                logprobs=[[0.0], [0.0]],
                reward_scores=reward_scores,
                reward_metrics={},
                model_step=0,
            )

        inference_results = Queue()
        inference_results.put(make_result("0_0", [0.0, 0.0]))
        inference_results.put(make_result("0_0_1", [0.0, 1.0]))
        generation_config = Mock(n=2)
        dataset = Dataset.from_dict(
            {
                INPUT_IDS_PROMPT_KEY: [[10]],
                GROUND_TRUTHS_KEY: [[11]],
                VERIFIER_SOURCE_KEY: ["unit"],
                RAW_PROMPT_KEY: ["prompt"],
                "index": [0],
            }
        )

        result, batch, _, batch_stats = data_loader.accumulate_inference_batches(
            inference_results,
            generation_config,
            num_prompts=1,
            model_dims=Mock(),
            tokenizer=MockTokenizer(),
            dataset=dataset,
            base_env_config=EnvConfig(),
            training_step=0,
            active_sampling=True,
            filter_zero_std_samples=True,
            never_give_up=1.0,
            maintain_pending_ngu_completions=True,
        )

        self.assertEqual(len(result.responses), 4)
        self.assertEqual(batch.scores, [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(batch_stats.prompt_sample_counts, [4])


if __name__ == "__main__":
    unittest.main()
