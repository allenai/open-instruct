"""Tests for grpo_utils helpers whose import chain requires vllm.

Collection of this module is skipped when vllm is unavailable (e.g. on macOS
dev machines) via ``collect_ignore`` in the repository-root ``conftest.py``.
"""

import unittest

import parameterized

from open_instruct import grpo_utils


class CheckOlmoCoreCompatibleConfigTest(unittest.TestCase):
    def test_default_config_passes(self):
        args = grpo_utils.GRPOExperimentConfig()
        grpo_utils.check_olmo_core_compatible_config(args)  # must not raise

    @parameterized.parameterized.expand(
        [
            ("deepspeed_stage", {"deepspeed_stage": 2}),
            ("deepspeed_zpg", {"deepspeed_zpg": 1}),
            ("deepspeed_offload_param", {"deepspeed_offload_param": True}),
            ("deepspeed_offload_optimizer", {"deepspeed_offload_optimizer": True}),
            ("deepspeed_checkpoint_load_universal", {"deepspeed_checkpoint_load_universal": True}),
            # sequence_parallel_size > 1 requires deepspeed_stage == 3 at construction
            # time (GRPOExperimentConfig.__post_init__), so set both; the guard must
            # still name sequence_parallel_size in its error.
            ("sequence_parallel_size", {"sequence_parallel_size": 4, "deepspeed_stage": 3}),
            # consumed only by grpo_fast.py's weight sync; the OLMo-core actor
            # always broadcasts with the default.
            ("gather_whole_model", {"gather_whole_model": False}),
        ]
    )
    def test_deepspeed_only_flag_raises(self, flag, overrides):
        with self.assertRaisesRegex(ValueError, flag):
            args = grpo_utils.GRPOExperimentConfig(**overrides)
            grpo_utils.check_olmo_core_compatible_config(args)

    def test_guard_defaults_match_config(self):
        for name, default in grpo_utils._DEEPSPEED_ONLY_FLAG_DEFAULTS.items():
            self.assertEqual(default, getattr(grpo_utils.GRPOExperimentConfig(), name))


if __name__ == "__main__":
    unittest.main()
