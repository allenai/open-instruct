import tempfile
import unittest

import numpy as np
import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader, data_types, model_utils
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


def _make_result_and_batch(scores: list[float], finish_reasons: list[str]):
    n = len(scores)
    result = data_types.GenerationResult(
        responses=[[i] for i in range(n)],
        finish_reasons=list(finish_reasons),
        masks=[[1] for _ in range(n)],
        request_info=data_types.RequestInfo(
            num_calls=[0] * n,
            timeouts=[0] * n,
            tool_errors=[""] * n,
            tool_outputs=[""] * n,
            tool_runtimes=[0.0] * n,
            tool_calleds=[False] * n,
        ),
        index=None,
        prompt_id=None,
        logprobs=[[0.0] for _ in range(n)],
    )
    batch = model_utils.Batch(
        queries=[[0] for _ in range(n)],
        ground_truths=[[0] for _ in range(n)],
        datasets=["d"] * n,
        raw_queries=None,
        decoded_responses=None,
        indices=list(range(n)),
        scores=list(scores),
        model_steps=[0] * n,
    )
    return result, batch


class TestMaskTruncatedCompletions(unittest.TestCase):
    # Two prompts, four samples each: A = [1, 1, 0, 1], B = [0, 0, 1, 0].
    SCORES = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    R, S = 1 / np.sqrt(3), np.sqrt(3)

    @parameterized.parameterized.expand(
        [
            # 6 survivors don't fill whole groups of 4; regrouping them can't even be reshaped.
            (
                "uneven_survivors",
                ["stop", "length", "stop", "stop", "stop", "stop", "stop", "length"],
                [R, -S, R, -R, -R, S],
            ),
            # 4 survivors would reshape into one group, pooling prompt A's and prompt B's rewards.
            (
                "pooled_survivors",
                ["stop", "length", "length", "stop", "stop", "length", "stop", "length"],
                [R, R, -R, S],
            ),
        ]
    )
    def test_advantages_keep_full_prompt_groups(self, _name, finish_reasons, expected):
        result, batch = _make_result_and_batch(self.SCORES, finish_reasons)
        advantages = data_loader.compute_group_advantages(np.array(self.SCORES), 4, "standard")

        batch, advantages = data_loader.maybe_mask_truncated_completions(result, batch, advantages, enabled=True)

        kept = [i for i, fr in enumerate(finish_reasons) if fr == "stop"]
        np.testing.assert_allclose(advantages, expected, rtol=1e-6)
        self.assertEqual(batch.scores, [self.SCORES[i] for i in kept])
        self.assertEqual(result.responses, [[i] for i in kept])
        self.assertEqual(result.finish_reasons, ["stop"] * len(kept))

    def test_disabled_keeps_everything(self):
        finish_reasons = ["stop", "length"] * 4
        result, batch = _make_result_and_batch(self.SCORES, finish_reasons)
        advantages = data_loader.compute_group_advantages(np.array(self.SCORES), 4, "centered")

        new_batch, new_advantages = data_loader.maybe_mask_truncated_completions(
            result, batch, advantages, enabled=False
        )

        self.assertIs(new_batch, batch)
        self.assertIs(new_advantages, advantages)
        self.assertEqual(result.finish_reasons, finish_reasons)


if __name__ == "__main__":
    unittest.main()
