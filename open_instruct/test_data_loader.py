import tempfile
import unittest

import numpy as np
import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader
from open_instruct.data_types import GenerationResult, RequestInfo
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


class TestResultIsStale(unittest.TestCase):
    def test_disabled_when_max_age_none(self):
        self.assertFalse(data_loader.result_is_stale(model_step=0, training_step=100, max_result_age_steps=None))

    def test_disabled_when_inputs_missing(self):
        self.assertFalse(data_loader.result_is_stale(model_step=None, training_step=100, max_result_age_steps=4))
        self.assertFalse(data_loader.result_is_stale(model_step=0, training_step=None, max_result_age_steps=4))

    def test_stale_when_lag_exceeds_threshold(self):
        # lag = 100 - 95 = 5 > 4 -> stale
        self.assertTrue(data_loader.result_is_stale(model_step=95, training_step=100, max_result_age_steps=4))

    def test_not_stale_at_threshold(self):
        # lag = 100 - 96 = 4, not > 4 -> fresh
        self.assertFalse(data_loader.result_is_stale(model_step=96, training_step=100, max_result_age_steps=4))

    def test_not_stale_when_fresh(self):
        self.assertFalse(data_loader.result_is_stale(model_step=100, training_step=100, max_result_age_steps=4))

    def test_max_result_age_requires_replenish_prompts(self):
        # The guard fires before any of the (here-dummy) inputs are used.
        with self.assertRaisesRegex(ValueError, "replenish_prompts"):
            data_loader.accumulate_inference_batches(
                inference_results_Q=None,
                generation_config=None,
                num_prompts=1,
                model_dims=None,
                tokenizer=None,
                dataset=None,
                base_env_config=None,
                training_step=0,
                replenish_prompts=False,
                max_result_age_steps=4,
            )


class TestMaskTruncatedCompletions(unittest.TestCase):
    def _make_result(self, finish_reasons: list[str]) -> GenerationResult:
        n = len(finish_reasons)
        return GenerationResult(
            responses=[[i] for i in range(n)],
            finish_reasons=finish_reasons,
            masks=[[1] for _ in range(n)],
            request_info=RequestInfo(
                num_calls=[0] * n,
                timeouts=[0] * n,
                tool_errors=[""] * n,
                tool_outputs=[""] * n,
                tool_runtimes=[0.0] * n,
                tool_calleds=[False] * n,
            ),
            index=None,
            prompt_id="0",
            logprobs=[[0.0] for _ in range(n)],
        )

    def _make_batch(self, n: int) -> data_loader.Batch:
        return data_loader.Batch(
            queries=[[0]] * n,
            ground_truths=[[0]] * n,
            datasets=["ds"] * n,
            raw_queries=["q"] * n,
            decoded_responses=["r"] * n,
            indices=[0] * n,
            scores=[1.0] * n,
            model_steps=[0] * n,
        )

    def test_disabled_is_a_no_op(self):
        finish_reasons = ["stop", "length", "stop"]
        result = self._make_result(finish_reasons)
        batch = self._make_batch(3)
        advantages = np.array([1.0, 2.0, 3.0])

        new_batch, new_advantages = data_loader.maybe_mask_truncated_completions(
            result, batch, advantages, enabled=False
        )

        self.assertEqual(len(new_batch.scores), 3)
        self.assertTrue(np.array_equal(new_advantages, advantages))

    def test_filters_batch_and_advantages_by_the_same_surviving_indices(self):
        # 2 prompts x 2 samples; the second sample of the first prompt is truncated.
        finish_reasons = ["stop", "length", "stop", "stop"]
        result = self._make_result(finish_reasons)
        batch = self._make_batch(4)
        # Pre-filter, group-computed advantages (as the caller now builds them before masking).
        advantages = np.array([-1.0, 1.0, -0.5, 0.5])

        new_batch, new_advantages = data_loader.maybe_mask_truncated_completions(
            result, batch, advantages, enabled=True
        )

        # Index 1 (the truncated sample) is dropped; the rest survive in order.
        self.assertEqual(len(result.responses), 3)
        self.assertEqual(len(new_batch.scores), 3)
        self.assertTrue(np.array_equal(new_advantages, np.array([-1.0, -0.5, 0.5])))
        # advantages must stay index-aligned with the filtered batch/result for every caller
        # downstream (lookup_advantages, save_rollouts_to_disk, val/advantages_* metrics).
        self.assertEqual(len(new_advantages), len(new_batch.scores))
        self.assertEqual(len(new_advantages), len(result.responses))


if __name__ == "__main__":
    unittest.main()
