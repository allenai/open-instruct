import unittest
from types import SimpleNamespace

from open_instruct import grpo_fast_resource_plan


class TestGrpoFastResourcePlanning(unittest.TestCase):
    def _make_args(self, *, num_learners_per_node, single_gpu_mode=False, num_nodes=1):
        return SimpleNamespace(
            num_learners_per_node=num_learners_per_node, single_gpu_mode=single_gpu_mode, num_nodes=num_nodes
        )

    def _make_vllm_config(self, *, vllm_num_engines, vllm_tensor_parallel_size):
        return SimpleNamespace(vllm_num_engines=vllm_num_engines, vllm_tensor_parallel_size=vllm_tensor_parallel_size)

    def test_build_resource_plan_for_single_node_tulu_topology(self):
        args = self._make_args(num_learners_per_node=[6])
        vllm_config = self._make_vllm_config(vllm_num_engines=2, vllm_tensor_parallel_size=1)

        plan = grpo_fast_resource_plan.build_grpo_fast_resource_plan(args, vllm_config)

        self.assertEqual(plan.learner_pg_strategy, "STRICT_SPREAD")
        self.assertEqual(plan.learner_pg_bundles, [{"GPU": 6, "CPU": 60}])
        self.assertEqual(plan.learner_pg_total_gpus, 6.0)
        self.assertEqual(plan.learner_pg_total_cpus, 60.0)
        self.assertEqual(plan.separate_vllm_total_gpus, 2.0)
        self.assertEqual(plan.separate_vllm_total_cpus, 2.0)
        self.assertEqual(plan.min_total_cluster_gpus, 8.0)
        self.assertEqual(plan.min_total_cluster_cpus, 64.0)

    def test_build_resource_plan_omits_separate_vllm_totals_in_single_gpu_mode(self):
        args = self._make_args(num_learners_per_node=[1], single_gpu_mode=True)
        vllm_config = self._make_vllm_config(vllm_num_engines=1, vllm_tensor_parallel_size=1)

        plan = grpo_fast_resource_plan.build_grpo_fast_resource_plan(args, vllm_config)

        self.assertEqual(plan.separate_vllm_total_gpus, 0.0)
        self.assertEqual(plan.separate_vllm_total_cpus, 0.0)
        self.assertEqual(plan.min_total_cluster_gpus, 1.0)
        self.assertEqual(plan.min_total_cluster_cpus, 12.0)

    def test_resource_shortfalls_report_learner_pg_cpu_gap_first(self):
        args = self._make_args(num_learners_per_node=[6])
        vllm_config = self._make_vllm_config(vllm_num_engines=2, vllm_tensor_parallel_size=1)
        plan = grpo_fast_resource_plan.build_grpo_fast_resource_plan(args, vllm_config)

        shortfalls = grpo_fast_resource_plan.get_grpo_fast_resource_shortfalls(plan, {"GPU": 8, "CPU": 32})

        self.assertEqual(len(shortfalls), 1)
        self.assertIn("learner placement group requires CPU=60", shortfalls[0])

    def test_resource_shortfalls_report_full_topology_gap_after_pg_fits(self):
        args = self._make_args(num_learners_per_node=[6])
        vllm_config = self._make_vllm_config(vllm_num_engines=2, vllm_tensor_parallel_size=1)
        plan = grpo_fast_resource_plan.build_grpo_fast_resource_plan(args, vllm_config)

        shortfalls = grpo_fast_resource_plan.get_grpo_fast_resource_shortfalls(plan, {"GPU": 8, "CPU": 60})

        self.assertEqual(len(shortfalls), 1)
        self.assertIn("full topology requires at least CPU=64", shortfalls[0])
