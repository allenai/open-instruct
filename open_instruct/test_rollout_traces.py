import unittest

import numpy as np

from open_instruct import model_utils, rl_utils
from open_instruct.data_types import GenerationResult, RequestInfo


class TestRolloutRecords(unittest.TestCase):
    def _make_result(self) -> GenerationResult:
        return GenerationResult(
            responses=[[10, 11], [12, 13]],
            finish_reasons=["stop", "length"],
            masks=[[1, 1], [1, 1]],
            request_info=RequestInfo(
                num_calls=[0, 1],
                timeouts=[0, 0],
                tool_errors=["", ""],
                tool_outputs=["", "ok"],
                tool_runtimes=[0.0, 0.1],
                tool_calleds=[False, True],
                tool_call_stats=[[], []],
                rollout_states=[{}, {"done": False}],
            ),
            index=3,
            prompt_id="prompt_3",
            logprobs=[[0.1, 0.2], [0.3, 0.4]],
            model_step=7,
        )

    def _make_batch(self) -> model_utils.Batch:
        return model_utils.Batch(
            queries=[[1, 2, 3], [1, 2, 3]],
            ground_truths=[[4], [4]],
            datasets=["math", "math"],
            raw_queries=["user: solve 2+2", "user: solve 2+2"],
            decoded_responses=None,
            indices=[3, 3],
            scores=[10.0, 0.0],
            source_row_ids=[11, 11],
            source_datasets=["demo", "demo"],
            model_steps=[7, 7],
        )

    def test_build_rollout_records_full_format(self):
        records = rl_utils.build_rollout_records(
            self._make_batch(),
            self._make_result(),
            np.array([5.0, -5.0]),
            step=9,
            num_samples_per_prompt=2,
            record_format="full",
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["step"], 9)
        self.assertEqual(records[0]["prompt_idx"], 0)
        self.assertEqual(records[0]["source_row_id"], 11)
        self.assertEqual(records[0]["source_dataset"], "demo")
        self.assertEqual(records[0]["response_tokens"], [10, 11])
        self.assertEqual(records[1]["finish_reason"], "length")
        self.assertEqual(records[1]["request_info"]["tool_outputs"], "ok")

    def test_build_rollout_records_scores_only_format(self):
        records = rl_utils.build_rollout_records(
            self._make_batch(),
            self._make_result(),
            np.array([5.0, -5.0]),
            step=9,
            num_samples_per_prompt=2,
            record_format="scores_only",
        )

        self.assertEqual(
            records,
            [
                {"dataset": "math", "reward": 10.0, "source_row_id": 11, "source_dataset": "demo"},
                {"dataset": "math", "reward": 0.0, "source_row_id": 11, "source_dataset": "demo"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
