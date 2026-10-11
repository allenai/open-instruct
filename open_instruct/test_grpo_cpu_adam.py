import unittest
from unittest.mock import MagicMock, patch

import torch

from open_instruct import grpo_utils


class TestCpuAdamConfigValidation(unittest.TestCase):
    def test_default_config_uses_neither_cpu_adam_nor_offload(self):
        args = grpo_utils.GRPOExperimentConfig()
        self.assertFalse(args.use_cpu_adam)
        self.assertFalse(args.deepspeed_offload_optimizer)

    def test_cpu_adam_with_optimizer_offload_is_allowed(self):
        args = grpo_utils.GRPOExperimentConfig(use_cpu_adam=True, deepspeed_stage=2, deepspeed_offload_optimizer=True)
        self.assertTrue(args.use_cpu_adam)

    def test_cpu_adam_without_optimizer_offload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "deepspeed_offload_optimizer"):
            grpo_utils.GRPOExperimentConfig(use_cpu_adam=True, deepspeed_stage=2, deepspeed_offload_optimizer=False)

    def test_cpu_adam_without_deepspeed_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "deepspeed_stage"):
            grpo_utils.GRPOExperimentConfig(use_cpu_adam=True, deepspeed_stage=0, deepspeed_offload_optimizer=True)

    def test_param_offload_requires_stage_3(self):
        with self.assertRaisesRegex(ValueError, "deepspeed_offload_param"):
            grpo_utils.GRPOExperimentConfig(deepspeed_stage=2, deepspeed_offload_param=True)

    def test_param_offload_with_stage_3_is_allowed(self):
        args = grpo_utils.GRPOExperimentConfig(deepspeed_stage=3, deepspeed_offload_param=True)
        self.assertTrue(args.deepspeed_offload_param)


class TestCreateOptimizer(unittest.TestCase):
    def setUp(self):
        self.params = [torch.nn.Parameter(torch.zeros(2))]

    def test_default_is_adamw_with_configured_weight_decay(self):
        args = grpo_utils.GRPOExperimentConfig(learning_rate=1e-3, weight_decay=0.1, fused_optimizer=False)

        optimizer = grpo_utils.create_optimizer(self.params, args)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["lr"], 1e-3)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.1)

    def test_default_weight_decay_is_zero_not_adamw_default(self):
        # Passing ``weight_decay=args.weight_decay`` replaces torch's AdamW default (0.01) with the config's
        # default (0.0) for parameters that are not in an explicit weight-decay group.
        args = grpo_utils.GRPOExperimentConfig(fused_optimizer=False)

        optimizer = grpo_utils.create_optimizer(self.params, args)

        self.assertEqual(args.weight_decay, 0.0)
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0)

    def test_cpu_adam_receives_learning_rate_and_weight_decay(self):
        args = grpo_utils.GRPOExperimentConfig(
            use_cpu_adam=True,
            deepspeed_stage=2,
            deepspeed_offload_optimizer=True,
            learning_rate=2e-5,
            weight_decay=0.05,
        )
        cpu_adam = MagicMock()

        with patch.object(grpo_utils, "DeepSpeedCPUAdam", cpu_adam):
            optimizer = grpo_utils.create_optimizer(self.params, args)

        cpu_adam.assert_called_once_with(self.params, lr=2e-5, weight_decay=0.05)
        self.assertIs(optimizer, cpu_adam.return_value)

    def test_cpu_adam_without_deepspeed_raises(self):
        args = grpo_utils.GRPOExperimentConfig(use_cpu_adam=True, deepspeed_stage=2, deepspeed_offload_optimizer=True)

        with patch.object(grpo_utils, "DeepSpeedCPUAdam", None), self.assertRaisesRegex(RuntimeError, "DeepSpeed"):
            grpo_utils.create_optimizer(self.params, args)


if __name__ == "__main__":
    unittest.main()
