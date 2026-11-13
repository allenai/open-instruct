import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader, data_types, model_utils, rl_utils
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO
from open_instruct.rubrics import rubric_utils


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

    def test_saved_traces_ignore_later_truncation_filter(self):
        finish_reasons = ["stop", "length", "stop", "stop", "stop", "stop", "stop", "length"]
        result, batch = _make_result_and_batch(self.SCORES, finish_reasons)
        advantages = data_loader.compute_group_advantages(np.array(self.SCORES), 4, "standard")

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rl_utils._rollout_executor, "submit") as submit:
            rl_utils.save_rollouts_to_disk(tmp, "run", 0, batch, result, advantages, 4, 0)
            data_loader.maybe_mask_truncated_completions(result, batch, advantages, enabled=True)
            # The saver runs asynchronously, i.e. after the filter above has mutated `result`.
            fn, *args = submit.call_args.args
            fn(*args)

            with open(os.path.join(tmp, "run_rollouts_000000.jsonl")) as f:
                records = [json.loads(line) for line in f]

        self.assertEqual(len(records), 8)
        self.assertEqual([r["finish_reason"] for r in records], finish_reasons)
        self.assertEqual([r["response_tokens"] for r in records], [[i] for i in range(8)])

    @parameterized.parameterized.expand([("immediate", True), ("delayed", False)])
    def test_saved_traces_keep_scoring_rubrics(self, _name, save_immediately):
        result, batch = _make_result_and_batch(self.SCORES, ["stop"] * 8)
        batch.ground_truths = [
            [json.dumps({"query": f"Q{i // 4}", "rubrics": [{"description": "Original", "weight": 1.0}]})]
            for i in range(8)
        ]
        original_ground_truths = list(batch.ground_truths)
        advantages = data_loader.compute_group_advantages(np.array(self.SCORES), 4, "standard")
        evolving_rubrics = [
            {"positive_rubrics": [{"description": "New", "title": "New"}], "negative_rubrics": []} for _ in range(2)
        ]

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(rl_utils._rollout_executor, "submit") as submit:
            if save_immediately:
                submit.side_effect = lambda fn, *args: fn(*args)
            rl_utils.save_rollouts_to_disk(tmp, "run", 0, batch, result, advantages, 4, 0)
            # RubricManager's update replaces entries in the live ground-truth list after saving is queued.
            rubric_utils.update_ground_truths_with_evolving_rubrics(batch.ground_truths, evolving_rubrics, 4)
            if not save_immediately:
                fn, *args = submit.call_args.args
                fn(*args)

            with open(os.path.join(tmp, "run_rollouts_000000.jsonl")) as f:
                records = [json.loads(line) for line in f]

        self.assertTrue(all(len(json.loads(gt[0])["rubrics"]) == 2 for gt in batch.ground_truths))
        self.assertEqual([record["ground_truth"] for record in records], original_ground_truths)


def _make_packs(packs: list[list[tuple[int, int]]]) -> tuple[rl_utils.PackedSequences, list[tuple[int, int]]]:
    """Build packs of (query length, response length) documents in the layout `pack_sequences` produces.

    Tokens, logprobs and advantages of document d all hold d (tokens hold d + 1).
    """
    fields = ("query_responses", "attention_masks", "position_ids", "advantages", "response_masks", "vllm_logprobs")
    columns = {name: [] for name in fields}
    doc_lens = []
    for pack in packs:
        pieces = {name: [] for name in fields}
        for doc_id, (query_len, response_len) in enumerate(pack, start=1):
            d = len(doc_lens)
            doc_lens.append((query_len, response_len))
            length = query_len + response_len
            response_mask = torch.arange(length) >= query_len
            pieces["query_responses"].append(torch.full((length,), d + 1))
            pieces["attention_masks"].append(torch.full((length,), doc_id))
            pieces["position_ids"].append(torch.arange(length))
            pieces["advantages"].append(response_mask.float() * d)
            pieces["response_masks"].append(response_mask)
            pieces["vllm_logprobs"].append(torch.where(response_mask, float(d), float("nan")))
        for name in fields:
            columns[name].append(torch.cat(pieces[name]))
    return rl_utils.PackedSequences(original_responses=[], **columns), doc_lens


def _pack_sequences_with_doc_ids(doc_lens: list[tuple[int, int]], pack_length: int, dp_world_size: int):
    """Pack documents with `pack_sequences`; tokens, logprobs and advantages hold the document index."""
    queries = [[d + 1] * query_len for d, (query_len, _) in enumerate(doc_lens)]
    responses = [[d + 1] * response_len for d, (_, response_len) in enumerate(doc_lens)]
    packed = rl_utils.pack_sequences(
        queries=queries,
        responses=responses,
        masks=[[1] * len(response) for response in responses],
        pack_length=pack_length,
        pad_token_id=0,
        vllm_logprobs=[[float(d)] * len(response) for d, response in enumerate(responses)],
        min_num_batches=dp_world_size,
    )
    # Same advantage lookup and mask conversion as DataPreparationActor.
    lookup_advantages = np.arange(-1, len(doc_lens), dtype=np.float32)
    lookup_advantages[0] = 0.0
    packed.advantages = [torch.tensor(lookup_advantages[mask]) for mask in packed.response_masks]
    packed.response_masks = [mask.bool() for mask in packed.response_masks]
    return packed


class TestPrepareCollatedDataLeftoverPacks(unittest.TestCase):
    def _collated_docs(self, collated: list[data_types.CollatedBatchData], doc_lens: list[tuple[int, int]]):
        """Return {document index: packed length} after checking that every field of each document is intact."""
        docs = {}
        for worker in collated:
            for micro_batch in range(len(worker.query_responses)):
                tokens = worker.query_responses[micro_batch][0]
                doc_ids, counts = torch.unique_consecutive(worker.attention_masks[micro_batch][0], return_counts=True)
                self.assertEqual(doc_ids.tolist(), list(range(1, len(doc_ids) + 1)))
                start = 0
                for length in counts.tolist():
                    end = start + length
                    d = int(tokens[start]) - 1
                    query_len, response_len = doc_lens[d]
                    self.assertNotIn(d, docs)
                    self.assertEqual(length, query_len + response_len)
                    self.assertTrue(torch.all(tokens[start:end] == d + 1))
                    position_ids = worker.position_ids[micro_batch][0][start:end]
                    self.assertEqual(position_ids.tolist(), list(range(length)))
                    response_mask = worker.response_masks[micro_batch][0][start:end]
                    self.assertEqual(response_mask.tolist(), [False] * query_len + [True] * response_len)
                    advantages = worker.advantages[micro_batch][0][start:end]
                    self.assertTrue(torch.all(advantages[response_mask] == d))
                    self.assertTrue(torch.all(advantages[~response_mask] == 0))
                    logprobs = worker.vllm_logprobs[micro_batch][0][start:end]
                    self.assertTrue(torch.all(logprobs[response_mask] == d))
                    self.assertTrue(torch.all(torch.isnan(logprobs[~response_mask])))
                    docs[d] = length
                    start = end
        return docs

    def test_leftover_documents_are_moved_into_shorter_packs(self):
        # Packs of 10, 7, 7, 3 and 3 tokens for 3 ranks: the last two packs fit into the first three.
        packed, doc_lens = _make_packs(
            [[(1, 5), (1, 3)], [(1, 4), (1, 1)], [(1, 3), (1, 2)], [(1, 1), (1, 0)], [(1, 2)]]
        )

        collated = data_loader.prepare_collated_data_for_workers(packed, 3, 1, pad_token_id=0, pin_memory=False)

        self.assertEqual([len(worker.query_responses) for worker in collated], [1, 1, 1])
        self.assertEqual([worker.query_responses[0].shape[1] for worker in collated], [10, 10, 10])
        docs = self._collated_docs(collated, doc_lens)
        self.assertEqual(sorted(docs), list(range(len(doc_lens))))

    def test_leftover_document_that_fits_nowhere_is_dropped(self):
        # Packs of 6, 5 and 5 (4 + 1) tokens for 2 ranks: only the 1-token document fits.
        packed, doc_lens = _make_packs([[(1, 5)], [(1, 4)], [(1, 3), (1, 0)]])

        with self.assertLogs(data_loader.logger, level="WARNING") as logs:
            collated = data_loader.prepare_collated_data_for_workers(packed, 2, 1, pad_token_id=0, pin_memory=False)

        self.assertIn("Dropping 1 responses", logs.output[0])
        self.assertEqual([worker.query_responses[0].shape[1] for worker in collated], [6, 6])
        self.assertEqual(sorted(self._collated_docs(collated, doc_lens)), [0, 1, 3])

    @parameterized.parameterized.expand([(2,), (3,), (4,), (8,)])
    def test_ranks_get_equal_packs_of_whole_documents(self, dp_world_size):
        recovered = 0
        for seed in range(10):
            rng = np.random.default_rng(seed)
            doc_lens = [(int(q), int(r)) for q, r in zip(rng.integers(1, 64, 96), rng.integers(0, 448, 96))]
            packed = _pack_sequences_with_doc_ids(doc_lens, pack_length=1024, dp_world_size=dp_world_size)
            num_packs = len(packed.query_responses) // dp_world_size
            kept_docs = sum(
                len(torch.unique_consecutive(m)) for m in packed.attention_masks[: num_packs * dp_world_size]
            )
            max_len = max(len(pack) for pack in packed.query_responses[: num_packs * dp_world_size])

            collated = data_loader.prepare_collated_data_for_workers(
                packed, dp_world_size, 1, pad_token_id=0, pin_memory=False
            )

            self.assertEqual([len(worker.query_responses) for worker in collated], [num_packs] * dp_world_size)
            self.assertLessEqual(max(qr.shape[1] for worker in collated for qr in worker.query_responses), max_len)
            num_docs = len(self._collated_docs(collated, doc_lens))
            self.assertGreaterEqual(num_docs, kept_docs)
            recovered += num_docs > kept_docs
        self.assertGreater(recovered, 0)


if __name__ == "__main__":
    unittest.main()
