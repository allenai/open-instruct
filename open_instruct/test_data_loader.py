import tempfile
import unittest

import numpy as np
import parameterized
import torch
from datasets import Dataset

from open_instruct import data_loader, data_types
from open_instruct.model_utils import Batch
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


class TestMaskTruncatedCompletions(unittest.TestCase):
    def _make_batch(self) -> Batch:
        return Batch(
            queries=[[11], [11], [22], [22]],
            ground_truths=[[1], [1], [2], [2]],
            datasets=["train", "train", "train", "train"],
            raw_queries=["q0a", "q0b", "q1a", "q1b"],
            decoded_responses=["r0a", "r0b", "r1a", "r1b"],
            indices=[10, 10, 11, 11],
            scores=[0.1, 0.2, 0.3, 0.4],
            model_steps=[0, 0, 0, 0],
        )

    def _make_result(self, finish_reasons: list[str]) -> data_types.GenerationResult:
        return data_types.GenerationResult(
            responses=[[100 + i, 200 + i] for i in range(len(finish_reasons))],
            finish_reasons=finish_reasons,
            masks=[[1, 1] for _ in finish_reasons],
            request_info=data_types.RequestInfo(
                num_calls=[], timeouts=[], tool_errors=[], tool_outputs=[], tool_runtimes=[], tool_calleds=[]
            ),
            index=0,
            prompt_id="0_0",
            logprobs=[[0.1, 0.2] for _ in finish_reasons],
        )

    def test_mask_truncated_completions_keeps_batch_and_arrays_aligned(self):
        batch = self._make_batch()
        result = self._make_result(["stop", "length", "stop", "tool_calls"])
        scores = np.array(batch.scores, dtype=np.float32)
        advantages = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        filtered_batch, filtered_scores, filtered_advantages = data_loader.maybe_mask_truncated_completions(
            result, batch, scores, advantages, enabled=True
        )

        self.assertEqual(filtered_batch.indices, [10, 11])
        self.assertEqual(filtered_batch.scores, [0.1, 0.3])
        self.assertEqual(filtered_batch.decoded_responses, ["r0a", "r1a"])
        self.assertEqual(result.finish_reasons, ["stop", "stop"])
        self.assertEqual(result.responses, [[100, 200], [102, 202]])
        self.assertEqual(result.logprobs, [[0.1, 0.2], [0.1, 0.2]])
        np.testing.assert_allclose(filtered_scores, np.array([0.1, 0.3], dtype=np.float32))
        np.testing.assert_allclose(filtered_advantages, np.array([1.0, 3.0], dtype=np.float32))

    def test_mask_truncated_completions_handles_all_truncated(self):
        batch = self._make_batch()
        result = self._make_result(["length", "tool_calls", "length", "tool_calls"])
        scores = np.array(batch.scores, dtype=np.float32)
        advantages = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        filtered_batch, filtered_scores, filtered_advantages = data_loader.maybe_mask_truncated_completions(
            result, batch, scores, advantages, enabled=True
        )

        self.assertEqual(filtered_batch.indices, [])
        self.assertEqual(filtered_batch.scores, [])
        self.assertEqual(result.responses, [])
        self.assertEqual(result.finish_reasons, [])
        self.assertEqual(result.masks, [])
        self.assertEqual(result.logprobs, [])
        self.assertEqual(filtered_scores.size, 0)
        self.assertEqual(filtered_advantages.size, 0)


if __name__ == "__main__":
    unittest.main()
