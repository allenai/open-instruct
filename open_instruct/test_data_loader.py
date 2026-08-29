import tempfile
import unittest

import numpy as np
import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader, data_types
from open_instruct.data_loader_utils import NeverGiveUpAccumulationState
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO


class _FakeTokenizer:
    def batch_decode(self, sequences, skip_special_tokens=False):
        return [f"decoded_{i}" for i in range(len(sequences))]


def _make_result(reward_scores: list[float], prompt_id: str, index: int = 0, model_step: int = 0):
    n = len(reward_scores)
    return data_types.GenerationResult(
        responses=[[1, 2, 3] for _ in range(n)],
        finish_reasons=["stop"] * n,
        masks=[[1, 1, 1] for _ in range(n)],
        request_info=data_types.RequestInfo(
            num_calls=[0] * n,
            timeouts=[0] * n,
            tool_errors=[""] * n,
            tool_outputs=[""] * n,
            tool_runtimes=[0.0] * n,
            tool_calleds=[False] * n,
            tool_call_stats=[[] for _ in range(n)],
            rollout_states=[{} for _ in range(n)],
        ),
        index=index,
        prompt_id=prompt_id,
        token_statistics=data_types.TokenStatistics(
            num_prompt_tokens=3 * n, num_response_tokens=3 * n, generation_time=1.0
        ),
        logprobs=[[0.0, 0.0, 0.0] for _ in range(n)],
        reward_scores=list(reward_scores),
        reward_metrics={},
        model_step=model_step,
    )


def _make_group(result):
    return data_loader.Group(
        result=result,
        query=[9, 9],
        ground_truth=[1],
        dataset="ds",
        raw_query="q",
        active_tools=None,
        index=result.index,
        decoded_responses=[f"r{i}" for i in range(len(result.responses))],
        reward_scores=result.reward_scores,
        reward_metrics={},
        percent_solved=float(np.mean(result.reward_scores)),
        sample_count=len(result.responses),
        baseline_sample_count=len(result.responses),
        baseline_reward_sum=float(np.sum(result.reward_scores)),
    )


class TestMaybeFilterGroup(unittest.TestCase):
    def _filter(self, group, state, *, never_give_up, max_possible_score=1.0):
        return data_loader.maybe_filter_group(
            group=group,
            tokenizer=_FakeTokenizer(),
            max_possible_score=max_possible_score,
            filter_zero_std_samples=True,
            active_sampling=False,
            never_give_up=never_give_up,
            never_give_up_state=state,
            never_give_up_state_lock=None,
            maintain_pending_ngu_age=4,
            maintain_pending_ngu_completions=True,
        )

    def test_ngu_disabled_filters_zero_std_group(self):
        result = _make_result([0.0, 0.0, 0.0, 0.0], prompt_id="0_0")
        out = self._filter(_make_group(result), NeverGiveUpAccumulationState(), never_give_up=0.0)
        self.assertIsNone(out.group)
        self.assertEqual(out.filtered_results, [result])
        self.assertFalse(out.never_give_up)

    def test_ngu_requeues_unsolved_zero_std_group(self):
        state = NeverGiveUpAccumulationState()
        result = _make_result([0.0, 0.0, 0.0, 0.0], prompt_id="0_0")
        out = self._filter(_make_group(result), state, never_give_up=1.0, max_possible_score=10.0)
        self.assertIsNone(out.group)
        self.assertTrue(out.never_give_up)
        self.assertEqual(out.filtered_results, [])
        # Chain state buffered the attempt.
        self.assertIn("0_0", state.pending_response_counts)
        self.assertEqual(state.pending_response_counts["0_0"], 4)
        self.assertEqual(state.pending_attempt_counts["0_0"], 1)

    def test_ngu_accepts_and_merges_retry_with_signal(self):
        state = NeverGiveUpAccumulationState()
        # First attempt: all zeros, unsolved -> buffered.
        first = _make_result([0.0, 0.0, 0.0, 0.0], prompt_id="0_0")
        self._filter(_make_group(first), state, never_give_up=1.0, max_possible_score=10.0)
        # Retry attempt (prompt id gained a _1 suffix) now has a positive sample.
        retry = _make_result([0.0, 10.0, 0.0, 0.0], prompt_id="0_0_1")
        out = self._filter(_make_group(retry), state, never_give_up=1.0, max_possible_score=10.0)
        self.assertIsNotNone(out.group)
        self.assertFalse(out.never_give_up)
        self.assertEqual(out.group.sample_count, 8)
        self.assertEqual(out.group.baseline_sample_count, 8)
        self.assertEqual(out.group.attempt_count, 2)
        self.assertEqual(len(out.group.result.responses), 8)
        # Chain state consumed.
        self.assertNotIn("0_0", state.pending_response_counts)


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


if __name__ == "__main__":
    unittest.main()
