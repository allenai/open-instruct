"""Unit tests for cache-validation and checkpoint-detection helpers."""

import json
import os
import tempfile
import unittest
from unittest import mock

import torch
from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.olmo3 import build_olmo3_moe_config_from_hf_config
from parameterized import parameterized
from transformers import AutoConfig

from open_instruct import dataset_transformation, olmo_core_finetune, olmo_core_utils


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w"):
        pass


class NumpyDirIsPopulatedTest(unittest.TestCase):
    def test_empty_dir_is_not_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp))

    def test_token_ids_only_is_not_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "token_ids_part_0000.npy"))
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp))

    def test_missing_metadata_is_not_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "token_ids_part_0000.npy"))
            _touch(os.path.join(tmp, "labels_mask_part_0000.npy"))
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp))

    def test_complete_single_chunk_is_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "token_ids_part_0000.npy"))
            _touch(os.path.join(tmp, "labels_mask_part_0000.npy"))
            _touch(os.path.join(tmp, "token_ids_part_0000.csv.gz"))
            self.assertTrue(olmo_core_finetune._numpy_dir_is_populated(tmp))

    def test_partial_second_chunk_is_not_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for i in (0, 1):
                _touch(os.path.join(tmp, f"token_ids_part_{i:04d}.npy"))
            _touch(os.path.join(tmp, "labels_mask_part_0000.npy"))
            _touch(os.path.join(tmp, "token_ids_part_0000.csv.gz"))
            self.assertFalse(olmo_core_finetune._numpy_dir_is_populated(tmp))


class NativeCheckpointInitializationTest(unittest.TestCase):
    def test_loads_only_model_weights(self) -> None:
        trainer = mock.MagicMock()

        olmo_core_finetune._load_initial_native_checkpoint(trainer, "/checkpoints/pretrained")

        trainer.load_checkpoint.assert_called_once_with(
            "/checkpoints/pretrained", load_trainer_state=False, load_optim_state=False
        )


class IsHfCheckpointTest(unittest.TestCase):
    def test_local_dir_with_config_json_is_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "config.json"))
            self.assertTrue(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_local_dir_without_config_json_is_olmo_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "model.pt"))
            self.assertFalse(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_native_checkpoint_with_root_config_json_is_not_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "config.json"))
            _touch(os.path.join(tmp, "model_and_optim", ".metadata"))
            self.assertFalse(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_relative_local_olmo_core_dir_is_olmo_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                os.makedirs("ckpt")
                _touch(os.path.join("ckpt", "model.pt"))
                self.assertFalse(olmo_core_utils.is_hf_checkpoint("ckpt"))
            finally:
                os.chdir(cwd)

    @parameterized.expand([("allenai/Olmo-3-1025-7B",), ("allenai/OLMo-2-1124-7B",), ("Qwen/Qwen3-0.6B",)])
    def test_nonexistent_hub_id_is_hf(self, path: str) -> None:
        self.assertFalse(os.path.exists(path))
        self.assertTrue(olmo_core_utils.is_hf_checkpoint(path))

    def test_hf_marker_in_absolute_path(self) -> None:
        # Path doesn't exist on disk, but contains '-hf'.
        self.assertTrue(olmo_core_utils.is_hf_checkpoint("/weka/checkpoints/some-model-hf/step1"))


class SFTTerminalEosTest(unittest.TestCase):
    def test_replaces_truncated_token_and_matching_label(self) -> None:
        tokenizer = mock.MagicMock()
        tokenizer.eos_token_id = 99
        tokenized = (torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long), torch.tensor([[-100, 2]]))
        row = {"messages": [{"role": "assistant", "content": "answer"}]}

        with mock.patch.object(
            dataset_transformation, "_tokenize_tulu_sft_with_assistant_labels", return_value=tokenized
        ):
            out = dataset_transformation.sft_tulu_tokenize_and_truncate_v1(
                row, tokenizer, max_seq_length=2, ensure_terminal_eos_after_truncation=True
            )

        self.assertEqual([1, 99], out[dataset_transformation.INPUT_IDS_KEY].tolist())
        self.assertEqual([-100, 99], out[dataset_transformation.LABELS_KEY].tolist())


class Olmo3MoeModelConfigTest(unittest.TestCase):
    def test_setup_model_builds_olmo_ddp_with_requested_ep_settings(self) -> None:
        hf_config = Olmo3MoeConfig(
            vocab_size=64,
            hidden_size=32,
            attention_hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            max_position_embeddings=32,
            use_head_qk_norm=True,
            dense_layers_indices=[0],
            dense_mlp_intermediate_size=24,
            layer_types=["full_attention", "full_attention"],
            use_peri_ln=True,
        )
        args = olmo_core_utils.ModelConfig(
            model_name_or_path="test/olmo3moe",
            attn_implementation=AttentionBackendName.torch,
            moe_expert_parallel_degree=2,
            moe_expert_parallel_path=ExpertParallelPath.sync_1d.value,
            moe_expert_parallel_capacity_factor=2.0,
            moe_router_aux_loss_weight=0.015,
            moe_router_z_loss_weight=0.0001,
        )

        with mock.patch.object(AutoConfig, "from_pretrained", return_value=hf_config):
            model, model_config = olmo_core_utils.setup_model(args, init_device="meta")

        self.assertEqual("OLMoDDPModel", type(model).__name__)
        self.assertTrue(model_config.recompute_each_block)
        self.assertTrue(all(block.use_peri_norm for block in model_config.resolved_block_configs))
        routed_blocks = [block for block in model_config.resolved_block_configs if block.routed_experts is not None]
        self.assertEqual(1, len(routed_blocks))
        self.assertEqual(2.0, routed_blocks[0].ep.capacity_factor)
        self.assertEqual(0.015, routed_blocks[0].routed_experts_router.lb_loss_weight)
        self.assertEqual(0.0001, routed_blocks[0].routed_experts_router.z_loss_weight)

    @parameterized.expand([(AttentionType.default,), (AttentionType.fused_v2,)])
    def test_setup_model_derives_architecture_from_native_checkpoint(self, attention_type: AttentionType) -> None:
        hf_config = Olmo3MoeConfig(
            vocab_size=64,
            hidden_size=32,
            attention_hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            max_position_embeddings=128,
            use_head_qk_norm=True,
            dense_layers_indices=[0],
            dense_mlp_intermediate_size=24,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=16,
            use_peri_ln=True,
        )
        native_config = build_olmo3_moe_config_from_hf_config(
            hf_config, attention_backend=AttentionBackendName.torch, attention_type=attention_type
        )
        tc = mock.MagicMock()
        tc.tokenizer.pad_token_id = 1
        tc.tokenizer.bos_token_id = None
        tc.tokenizer.eos_token_id = 2

        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "model_and_optim", ".metadata"))
            with open(os.path.join(tmp, "config.json"), "w") as config_file:
                json.dump(
                    {
                        "model": native_config.as_dict(
                            include_class_name=True, include_registered_name=True, json_safe=True
                        ),
                        "dataset": {"max_target_sequence_length": 128},
                    },
                    config_file,
                )
            args = olmo_core_utils.ModelConfig(model_name_or_path=tmp, attn_implementation=AttentionBackendName.torch)
            model, derived_config = olmo_core_utils.setup_model(args, tc, init_device="meta")

        self.assertEqual("OLMoDDPModel", type(model).__name__)
        self.assertEqual(64, derived_config.vocab_size)
        self.assertEqual(16, derived_config.resolved_block_configs[0].sequence_mixer.head_dim)
        self.assertEqual(attention_type, derived_config.resolved_block_configs[0].sequence_mixer.name)
        self.assertTrue(derived_config.resolved_block_configs[0].use_peri_norm)
        self.assertIsNone(derived_config.resolved_block_configs[0].routed_experts)
        self.assertEqual(4, derived_config.resolved_block_configs[1].routed_experts.num_experts)

    def test_non_synchronized_ep_path_requires_multiple_ranks(self) -> None:
        with self.assertRaisesRegex(ValueError, "degree > 1"):
            olmo_core_utils.ModelConfig(
                model_name_or_path="test/olmo3moe",
                attn_implementation=AttentionBackendName.torch,
                moe_expert_parallel_path=ExpertParallelPath.rowwise_nvshmem.value,
            )


if __name__ == "__main__":
    unittest.main()
