import dataclasses
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
    NeverGiveUpAccumulationState,
    compute_grouped_advantages,
    compute_reinforce_ada_est_samples,
    get_never_give_up_chain_id,
    get_never_give_up_retry_suffix,
)
from open_instruct.data_types import EnvConfig, GenerationResult, RequestInfo, TokenStatistics
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    PASS_COUNT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO
from open_instruct.vllm_utils import SamplingConfig


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

        # filtered_ngu_baseline=True applies the NGU baseline (shifts the advantages).
        with_baseline = compute_grouped_advantages(
            scores,
            prompt_sample_counts=[3],
            prompt_baseline_sample_counts=[5],
            prompt_baseline_reward_sums=[1.0],
            advantage_normalization_type="centered",
            filtered_ngu_baseline=True,
        )
        # filtered_ngu_baseline=False falls back to the regular grouped mean.
        keep_all = compute_grouped_advantages(
            scores,
            prompt_sample_counts=[3],
            prompt_baseline_sample_counts=[5],
            prompt_baseline_reward_sums=[1.0],
            advantage_normalization_type="centered",
            filtered_ngu_baseline=False,
        )
        regular = compute_grouped_advantages(scores, prompt_sample_counts=[3], advantage_normalization_type="centered")

        self.assertTrue(np.allclose(keep_all, regular))
        self.assertFalse(np.allclose(with_baseline, regular))


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

    def _make_batch_stats(self, prompt_sample_counts: list[int]) -> data_loader.BatchStatistics:
        num_groups = len(prompt_sample_counts)
        return data_loader.BatchStatistics(
            prompt_lengths=[10] * num_groups,
            response_lengths=[5] * sum(prompt_sample_counts),
            filtered_prompts=0,
            filtered_prompts_zero=0,
            filtered_prompts_solved=0,
            filtered_prompts_nonzero=0,
            filtered_prompts_pct=0.0,
            percent_solved_mean=0.0,
            percent_solved_hist=np.array([]),
            no_resampled_prompts=0,
            total_prompts=num_groups,
            prompt_sample_counts=prompt_sample_counts,
            prompt_attempt_counts=[1] * num_groups,
            prompt_baseline_sample_counts=list(prompt_sample_counts),
            prompt_baseline_reward_sums=[float(c) for c in prompt_sample_counts],
        )

    def test_partial_group_truncation_keeps_counts_in_sync_with_scores(self):
        # Two groups of 3 and 2; one sample from the first group is truncated.
        finish_reasons = ["stop", "stop", "length", "stop", "stop"]
        result = self._make_result(finish_reasons)
        batch = self._make_batch(5)
        batch_stats = self._make_batch_stats(prompt_sample_counts=[3, 2])

        new_batch, new_batch_stats = data_loader.maybe_mask_truncated_completions(
            result, batch, batch_stats, enabled=True
        )

        self.assertEqual(new_batch_stats.prompt_sample_counts, [2, 2])
        self.assertEqual(sum(new_batch_stats.prompt_sample_counts), len(new_batch.scores))
        self.assertEqual(new_batch_stats.prompt_baseline_sample_counts, [3, 2])

        # Would previously raise "Mismatch between prompt_sample_counts and scores".
        compute_grouped_advantages(np.array(new_batch.scores), new_batch_stats.prompt_sample_counts)

    def test_fully_truncated_group_is_dropped_and_stays_aligned(self):
        # First group (size 1) is fully truncated; second group (size 2) survives intact.
        finish_reasons = ["length", "stop", "stop"]
        result = self._make_result(finish_reasons)
        batch = self._make_batch(3)
        batch_stats = self._make_batch_stats(prompt_sample_counts=[1, 2])
        batch_stats = dataclasses.replace(batch_stats, prompt_baseline_reward_sums=[9.0, 2.0])

        new_batch, new_batch_stats = data_loader.maybe_mask_truncated_completions(
            result, batch, batch_stats, enabled=True
        )

        self.assertEqual(new_batch_stats.prompt_sample_counts, [2])
        self.assertEqual(sum(new_batch_stats.prompt_sample_counts), len(new_batch.scores))
        # The surviving group's baseline reward sum (2.0), not the dropped group's (9.0).
        self.assertEqual(new_batch_stats.prompt_baseline_reward_sums, [2.0])

        compute_grouped_advantages(np.array(new_batch.scores), new_batch_stats.prompt_sample_counts)

    def test_no_truncation_is_a_no_op(self):
        finish_reasons = ["stop", "stop", "stop"]
        result = self._make_result(finish_reasons)
        batch = self._make_batch(3)
        batch_stats = self._make_batch_stats(prompt_sample_counts=[3])

        new_batch, new_batch_stats = data_loader.maybe_mask_truncated_completions(
            result, batch, batch_stats, enabled=True
        )

        self.assertEqual(new_batch_stats.prompt_sample_counts, [3])
        self.assertIs(new_batch_stats, batch_stats)

    def test_get_never_give_up_retry_suffix_increments_existing_suffix(self):
        self.assertEqual(get_never_give_up_retry_suffix("7_0", epoch_number=7, index=0), "_1")
        self.assertEqual(get_never_give_up_retry_suffix("7_0_1", epoch_number=7, index=0), "_2")

    def test_get_never_give_up_chain_id_strips_retry_suffix(self):
        self.assertEqual(get_never_give_up_chain_id("7_0"), "7_0")
        self.assertEqual(get_never_give_up_chain_id("7_0_1"), "7_0")

    def test_accumulate_inference_batches_reports_per_dataset_breakdown(self):
        class MockTokenizer:
            eos_token_id = 0

            def batch_decode(self, responses, skip_special_tokens=False):
                return [str(response) for response in responses]

        def make_result(index, prompt_id, reward_scores):
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
                index=index,
                prompt_id=prompt_id,
                token_statistics=TokenStatistics(num_prompt_tokens=1, num_response_tokens=2, generation_time=1.0),
                logprobs=[[0.0], [0.0]],
                reward_scores=reward_scores,
                reward_metrics={},
                model_step=0,
            )

        inference_results = Queue()
        # quartile0 prompt is all-solved -> filtered as "solved"; quartile1 prompt has nonzero std -> accepted.
        inference_results.put(make_result(0, "0_0", [1.0, 1.0]))
        inference_results.put(make_result(1, "0_1", [0.0, 1.0]))
        generation_config = Mock(n=2)
        dataset = Dataset.from_dict(
            {
                INPUT_IDS_PROMPT_KEY: [[10], [20]],
                GROUND_TRUTHS_KEY: [[11], [21]],
                VERIFIER_SOURCE_KEY: ["math_deepscaler_quartile0", "math_deepscaler_quartile1"],
                RAW_PROMPT_KEY: ["prompt0", "prompt1"],
                "index": [0, 1],
            }
        )

        _, _, _, batch_stats = data_loader.accumulate_inference_batches(
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
        )

        self.assertEqual(batch_stats.filtered_prompts_solved, 1)
        self.assertEqual(batch_stats.filtered_prompts_solved_by_dataset, {"math_deepscaler_quartile0": 1})
        self.assertEqual(batch_stats.nonzero_prompts_by_dataset, {"math_deepscaler_quartile1": 1})
        # The accepted quartile1 prompt contributes its 2 completions to the batch.
        self.assertEqual(batch_stats.completions_used_by_dataset, {"math_deepscaler_quartile1": 2})

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


def _make_ngu_result(prompt_id, responses, finish_reasons, reward_scores):
    num_samples = len(responses)
    return GenerationResult(
        responses=[list(response) for response in responses],
        finish_reasons=list(finish_reasons),
        masks=[[1] * len(response) for response in responses],
        request_info=RequestInfo(
            num_calls=[0] * num_samples,
            timeouts=[0] * num_samples,
            tool_errors=[""] * num_samples,
            tool_outputs=[""] * num_samples,
            tool_runtimes=[0.0] * num_samples,
            tool_calleds=[False] * num_samples,
            tool_call_stats=[[] for _ in range(num_samples)],
            rollout_states=[{} for _ in range(num_samples)],
        ),
        index=0,
        prompt_id=prompt_id,
        token_statistics=TokenStatistics(
            num_prompt_tokens=1, num_response_tokens=sum(len(response) for response in responses), generation_time=1.0
        ),
        logprobs=[[0.0] * len(response) for response in responses],
        reward_scores=list(reward_scores),
        reward_metrics={},
        model_step=0,
    )


class TestNguSeqMultiplier(unittest.TestCase):
    def _make_group(self, result):
        return data_loader.Group(
            result=result,
            query=[10],
            ground_truth=[11],
            dataset="unit",
            raw_query="prompt",
            active_tools=None,
            index=0,
            decoded_responses=[str(response) for response in result.responses],
            reward_scores=result.reward_scores,
            reward_metrics={},
            percent_solved=0.0,
            sample_count=len(result.responses),
            baseline_sample_count=len(result.responses),
            baseline_reward_sum=0.0,
        )

    def _filter(self, result, state, ngu_seq_multiplier=2, response_length=4):
        return data_loader.maybe_filter_group(
            group=self._make_group(result),
            tokenizer=Mock(),
            max_possible_score=1.0,
            filter_zero_std_samples=True,
            active_sampling=True,
            never_give_up=1.0,
            never_give_up_accept_on="better",
            never_give_up_state=state,
            never_give_up_state_lock=None,
            maintain_pending_ngu_age=-1,
            maintain_pending_ngu_completions=True,
            ngu_seq_multiplier=ngu_seq_multiplier,
            response_length=response_length,
        )

    def test_maybe_filter_group_continues_unfinished_and_buffers_finished(self):
        state = NeverGiveUpAccumulationState()
        result = _make_ngu_result("0_0", [[1, 2, 3, 4], [5]], ["length", "stop"], [0.0, 0.0])

        filter_result = self._filter(result, state)

        self.assertTrue(filter_result.never_give_up)
        self.assertEqual(len(filter_result.continuations), 1)
        continuation = filter_result.continuations[0]
        self.assertEqual(continuation.tokens, [1, 2, 3, 4])
        self.assertEqual(continuation.masks, [1, 1, 1, 1])
        self.assertEqual(continuation.max_tokens, 8)
        # Only the finished (stop) completion is buffered and counted toward the NGU baseline.
        self.assertEqual(state.pending_response_counts["0_0"], 1)
        self.assertEqual(len(state.pending_results["0_0"]), 1)
        self.assertEqual(state.pending_results["0_0"][0].responses, [[5]])

    def test_maybe_filter_group_does_not_continue_completions_at_the_cap(self):
        state = NeverGiveUpAccumulationState()
        # Both completions already used response_length * multiplier tokens: no continuations.
        result = _make_ngu_result(
            "0_0", [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]], ["length", "length"], [0.0, 0.0]
        )

        filter_result = self._filter(result, state)

        self.assertTrue(filter_result.never_give_up)
        self.assertEqual(filter_result.continuations, [])
        self.assertEqual(state.pending_response_counts["0_0"], 2)

    def test_maybe_filter_group_multiplier_one_keeps_existing_behavior(self):
        state = NeverGiveUpAccumulationState()
        result = _make_ngu_result("0_0", [[1, 2, 3, 4], [5]], ["length", "stop"], [0.0, 0.0])

        filter_result = self._filter(result, state, ngu_seq_multiplier=1)

        self.assertTrue(filter_result.never_give_up)
        self.assertEqual(filter_result.continuations, [])
        self.assertEqual(state.pending_response_counts["0_0"], 2)

    def test_accumulate_inference_batches_requeues_and_merges_continuations(self):
        class MockTokenizer:
            eos_token_id = 0

            def batch_decode(self, responses, skip_special_tokens=False):
                return [str(response) for response in responses]

        class FakeIterDataloader:
            _epoch = 0

            def __next__(self):
                return {"index": 0, INPUT_IDS_PROMPT_KEY: [10]}

        inference_results = Queue()
        # Attempt 1: one length-truncated completion (continued), one finished wrong one (buffered).
        inference_results.put(_make_ngu_result("0_0", [[1, 2, 3, 4], [5]], ["length", "stop"], [0.0, 0.0]))
        # Retry: the continuation comes back stitched (prefix + new tokens) and solved.
        inference_results.put(_make_ngu_result("0_0_1", [[1, 2, 3, 4, 9, 10], [6]], ["stop", "stop"], [1.0, 0.0]))
        param_prompt_Q = Queue()
        generation_config = Mock(n=2, max_tokens=4)
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
            ngu_seq_multiplier=2,
            replenish_prompts=True,
            iter_dataloader=FakeIterDataloader(),
            param_prompt_Q=param_prompt_Q,
        )

        # The NGU retry request resumes the unfinished completion with a doubled budget.
        retry_request = param_prompt_Q.get_nowait()
        self.assertEqual(retry_request.prompt_id, "0_0_1")
        self.assertEqual(len(retry_request.continuations), 1)
        self.assertEqual(retry_request.continuations[0].tokens, [1, 2, 3, 4])
        self.assertEqual(retry_request.continuations[0].max_tokens, 8)

        # Merged group: buffered stop completion + both retry completions, with no double count
        # of the continued completion in the batch or the NGU baseline.
        self.assertEqual(len(result.responses), 3)
        self.assertEqual(sorted(batch.scores), [0.0, 0.0, 1.0])
        self.assertEqual(batch_stats.prompt_sample_counts, [3])
        self.assertEqual(batch_stats.prompt_baseline_sample_counts, [3])
        # 3 is not a multiple of generation_config.n=2: continuations can make a round finalize
        # fewer than n samples, so utilization accounting must use the round count (2), not
        # sample_count // n (which would be 3 // 2 == 1, silently wrong instead of a crash).
        self.assertEqual(batch_stats.prompt_attempt_counts, [2])

    def test_streaming_config_validates_ngu_seq_multiplier(self):
        with self.assertRaises(ValueError):
            data_loader.StreamingDataLoaderConfig(ngu_seq_multiplier=0)
        with self.assertRaises(ValueError):
            data_loader.StreamingDataLoaderConfig(ngu_seq_multiplier=2, never_give_up=0.0)
        with self.assertRaises(AssertionError):
            data_loader.StreamingDataLoaderConfig(
                ngu_seq_multiplier=2,
                never_give_up=0.5,
                max_prompt_token_length=256,
                response_length=256,
                pack_length=512,
            )
        config = data_loader.StreamingDataLoaderConfig(
            ngu_seq_multiplier=2, never_give_up=0.5, max_prompt_token_length=256, response_length=256, pack_length=768
        )
        self.assertEqual(config.total_response_length, 512)


class TestReinforceAdaEst(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            ("zero", 0, 32),
            ("one", 1, 32),
            ("two", 2, 16),
            ("three", 3, 16),
            ("four", 4, 8),
            ("seven", 7, 8),
            ("eight", 8, 4),
            ("max", 32, 4),
        ]
    )
    def test_compute_reinforce_ada_est_samples_buckets(self, _name, pass_count, expected_n):
        self.assertEqual(compute_reinforce_ada_est_samples(pass_count), expected_n)

    def test_streaming_config_validates_reinforce_ada_est(self):
        with self.assertRaises(ValueError):
            data_loader.StreamingDataLoaderConfig(reinforce_ada_est=True, batch_by="completions")
        with self.assertRaises(ValueError):
            data_loader.StreamingDataLoaderConfig(reinforce_ada_est=True, never_give_up=0.5)
        config = data_loader.StreamingDataLoaderConfig(reinforce_ada_est=True, batch_by="prompts")
        self.assertTrue(config.reinforce_ada_est)

    def test_add_prompt_to_generator_sets_n_from_pass_count(self):
        example = {"index": 0, INPUT_IDS_PROMPT_KEY: [10], PASS_COUNT_KEY: 8}
        generation_config = SamplingConfig(temperature=1.0, n=16)
        param_prompt_Q = Queue()

        data_loader.add_prompt_to_generator(
            example,
            epoch_number=0,
            param_prompt_Q=param_prompt_Q,
            generation_config=generation_config,
            is_eval=False,
            base_env_config=EnvConfig(),
            reinforce_ada_est=True,
        )

        request = param_prompt_Q.get_nowait()
        self.assertEqual(request.generation_config.n, 4)
        # The shared base config passed in must not be mutated.
        self.assertEqual(generation_config.n, 16)

    def test_add_prompt_to_generator_ignores_pass_count_for_eval(self):
        example = {"index": 0, INPUT_IDS_PROMPT_KEY: [10], PASS_COUNT_KEY: 8}
        generation_config = SamplingConfig(temperature=1.0, n=16)
        param_prompt_Q = Queue()

        data_loader.add_prompt_to_generator(
            example,
            epoch_number=0,
            param_prompt_Q=param_prompt_Q,
            generation_config=generation_config,
            is_eval=True,
            base_env_config=EnvConfig(),
            reinforce_ada_est=True,
        )

        request = param_prompt_Q.get_nowait()
        self.assertEqual(request.generation_config.n, 16)

    def test_process_group_uses_pass_count_as_expected_n(self):
        class MockTokenizer:
            eos_token_id = 0

            def batch_decode(self, responses, skip_special_tokens=False):
                return [str(response) for response in responses]

        result = GenerationResult(
            responses=[[1], [2], [3], [4]],
            finish_reasons=["stop"] * 4,
            masks=[[1]] * 4,
            request_info=RequestInfo(
                num_calls=[0] * 4,
                timeouts=[0] * 4,
                tool_errors=[""] * 4,
                tool_outputs=[""] * 4,
                tool_runtimes=[0.0] * 4,
                tool_calleds=[False] * 4,
                tool_call_stats=[[] for _ in range(4)],
                rollout_states=[{} for _ in range(4)],
            ),
            index=0,
            prompt_id="0_0",
            token_statistics=TokenStatistics(num_prompt_tokens=1, num_response_tokens=4, generation_time=1.0),
            logprobs=[[0.0]] * 4,
            reward_scores=[0.0, 0.0, 0.0, 1.0],
            reward_metrics={},
            model_step=0,
        )
        # pass_count=8 buckets to 4 samples, but the shared generation_config.n is 16: without
        # reinforce_ada_est this mismatch would (correctly) raise.
        dataset = Dataset.from_dict(
            {
                INPUT_IDS_PROMPT_KEY: [[10]],
                GROUND_TRUTHS_KEY: [[11]],
                VERIFIER_SOURCE_KEY: ["unit"],
                RAW_PROMPT_KEY: ["prompt"],
                PASS_COUNT_KEY: [8],
                "index": [0],
            }
        )
        generation_config = SamplingConfig(temperature=1.0, n=16)

        group = data_loader.process_group(
            result=result,
            generation_config=generation_config,
            tokenizer=MockTokenizer(),
            dataset=dataset,
            max_possible_score=1.0,
            reinforce_ada_est=True,
        )
        self.assertEqual(group.sample_count, 4)

        with self.assertRaises(AssertionError):
            data_loader.process_group(
                result=result,
                generation_config=generation_config,
                tokenizer=MockTokenizer(),
                dataset=dataset,
                max_possible_score=1.0,
                reinforce_ada_est=False,
            )


if __name__ == "__main__":
    unittest.main()
