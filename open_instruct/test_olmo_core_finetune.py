"""Unit tests for cache-validation and checkpoint-detection helpers."""

import os
import tempfile
import unittest

from olmo_core.nn.hf import convert as olmo_hf_convert
from parameterized import parameterized

from open_instruct import olmo_core_finetune, olmo_core_utils


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


class IsHfCheckpointTest(unittest.TestCase):
    def test_local_dir_with_config_json_is_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "config.json"))
            self.assertTrue(olmo_core_utils.is_hf_checkpoint(tmp))

    def test_local_dir_without_config_json_is_olmo_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "model.pt"))
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


class DeepSeekR1DistillQwenConfigTest(unittest.TestCase):
    """Offline (no download) checks for the qwen2-architecture preset and its HF conversion patch.

    See open_instruct/olmo_core_utils.py for why this preset and the "qwen2" mapping entries exist:
    olmo-core ships no Qwen2/DeepSeek-R1-Distill config or conversion support out of the box.
    """

    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    def test_preset_matches_hf_config_json(self) -> None:
        config = olmo_core_utils.get_transformer_config(self.MODEL_ID, vocab_size=151936, attn_backend="torch")
        self.assertEqual(config.d_model, 1536)
        self.assertEqual(config.n_layers, 28)
        self.assertEqual(config.vocab_size, 151936)
        self.assertFalse(config.tie_word_embeddings)
        sequence_mixer = config.block.sequence_mixer
        self.assertEqual(sequence_mixer.n_heads, 12)
        self.assertEqual(sequence_mixer.n_kv_heads, 2)
        self.assertEqual(sequence_mixer.head_dim, 128)
        # Qwen2 uses QKV attention bias, unlike the Qwen3 presets olmo-core ships.
        self.assertTrue(sequence_mixer.bias)
        # Qwen2 has no per-head QK-norm (that's a Qwen3-only architectural feature).
        self.assertIsNone(sequence_mixer.qk_norm)

    def test_qwen2_hf_conversion_mapping_is_registered(self) -> None:
        self.assertIn("qwen2", olmo_hf_convert.MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS)
        weight_mapping = olmo_hf_convert.MODEL_TYPE_SPECIFIC_HF_TO_OLMO_CORE_WEIGHT_MAPPINGS["qwen2"]
        for proj in ("q", "k", "v"):
            key = f"model.layers.{olmo_hf_convert.LAYER}.self_attn.{proj}_proj.bias"
            self.assertIn(key, weight_mapping)
            self.assertTrue(weight_mapping[key].endswith(f"attention.w_{proj}.bias"))

    def test_qwen2_style_model_types_includes_qwen2(self) -> None:
        self.assertIn("qwen2", olmo_core_utils.QWEN2_STYLE_HF_MODEL_TYPES)


if __name__ == "__main__":
    unittest.main()
