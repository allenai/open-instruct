import unittest
from unittest.mock import patch

from open_instruct.grpo_utils import validate_allocated_gpus


class TestValidateAllocatedGpus(unittest.TestCase):
    def test_accepts_matching_gpu_count(self):
        with patch("open_instruct.grpo_utils.ray.cluster_resources", return_value={"GPU": 16.0}):
            validate_allocated_gpus((4, 4), 8, 1, False)

    def test_accounts_for_tensor_parallel_size(self):
        with patch("open_instruct.grpo_utils.ray.cluster_resources", return_value={"GPU": 24.0}):
            validate_allocated_gpus((4, 4), 8, 2, False)

    def test_single_gpu_mode_does_not_require_extra_vllm_gpus(self):
        with patch("open_instruct.grpo_utils.ray.cluster_resources", return_value={"GPU": 1.0}):
            validate_allocated_gpus((1,), 1, 1, True)

    def test_single_gpu_mode_requires_exactly_one_learner_and_one_engine(self):
        with self.assertRaisesRegex(
            ValueError,
            r"single_gpu_mode requires exactly one learner and one vLLM engine.*"
            r"sum\(num_learners_per_node=\[4, 4\]\)=8.*vllm_num_engines=8",
        ):
            validate_allocated_gpus((4, 4), 8, 1, True)

    def test_rejects_mismatched_gpu_count(self):
        with (
            patch("open_instruct.grpo_utils.ray.cluster_resources", return_value={"GPU": 15.0}),
            self.assertRaisesRegex(
                ValueError,
                r"Ray reports 15\.0 allocated GPUs.*num_learners_per_node=\[4, 4\].*"
                r"vllm_num_engines \(8\) \* vllm_tensor_parallel_size \(1\).*"
                r"single_gpu_mode \(False\) = 16",
            ),
        ):
            validate_allocated_gpus((4, 4), 8, 1, False)
