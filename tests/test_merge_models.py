"""Unit tests for open_instruct.merge_models."""

import tempfile
import unittest
from pathlib import Path

import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, set_seed

from open_instruct.merge_models import get_safetensor_files, merge_models

MODEL_NAME = "EleutherAI/pythia-14m"


def _create_dummy_model(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Create a minimal model directory with one safetensors file and a config."""
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text('{"model_type": "test"}')
    (path / "tokenizer_config.json").write_text('{"tokenizer_class": "test"}')


def _save_model_with_seed(seed: int, tmp: Path, name: str) -> Path:
    """Load pythia-14m, reinitialize weights with a seed, and save as safetensors.

    Saves using safetensors directly (not save_pretrained) to avoid deepspeed import issues.
    """
    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    for param in model.parameters():
        torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
    path = tmp / name
    path.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(path / "model.safetensors"))
    config = AutoConfig.from_pretrained(MODEL_NAME)
    (path / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
    return path


class TestGetSafetensorFiles(unittest.TestCase):
    def test_finds_safetensor_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model"
            model_path.mkdir()
            (model_path / "model-00001.safetensors").write_bytes(b"")
            (model_path / "model-00002.safetensors").write_bytes(b"")
            (model_path / "config.json").write_text("{}")

            files = get_safetensor_files(model_path)
            self.assertEqual(files, ["model-00001.safetensors", "model-00002.safetensors"])

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(get_safetensor_files(Path(tmp)), [])


class TestMergeModels(unittest.TestCase):
    def _make_model(self, tmp: Path, name: str, weight_val: float) -> Path:
        model_path = tmp / name
        tensors = {
            "layer.weight": torch.full((4, 4), weight_val, dtype=torch.bfloat16),
            "layer.bias": torch.full((4,), weight_val, dtype=torch.bfloat16),
        }
        _create_dummy_model(model_path, tensors)
        return model_path

    def test_two_model_equal_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = self._make_model(tmp, "m1", 2.0)
            m2 = self._make_model(tmp, "m2", 4.0)
            output = tmp / "merged"

            merge_models([m1, m2], output)

            # Equal weights -> average of 2.0 and 4.0 = 3.0
            with safe_open(str(output / "model.safetensors"), framework="pt") as f:
                w = f.get_tensor("layer.weight")
                self.assertTrue(torch.allclose(w, torch.full((4, 4), 3.0, dtype=torch.bfloat16)))

    def test_three_model_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = self._make_model(tmp, "m1", 3.0)
            m2 = self._make_model(tmp, "m2", 6.0)
            m3 = self._make_model(tmp, "m3", 9.0)
            output = tmp / "merged"

            merge_models([m1, m2, m3], output)

            with safe_open(str(output / "model.safetensors"), framework="pt") as f:
                w = f.get_tensor("layer.weight")
                # Equal weights -> average of 3, 6, 9 = 6.0
                self.assertTrue(torch.allclose(w, torch.full((4, 4), 6.0, dtype=torch.bfloat16)))

    def test_custom_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = self._make_model(tmp, "m1", 0.0)
            m2 = self._make_model(tmp, "m2", 10.0)
            output = tmp / "merged"

            # Weight 3:1 -> 0.0 * 0.75 + 10.0 * 0.25 = 2.5
            merge_models([m1, m2], output, model_weights=[3.0, 1.0])

            with safe_open(str(output / "model.safetensors"), framework="pt") as f:
                w = f.get_tensor("layer.weight")
                self.assertTrue(torch.allclose(w, torch.full((4, 4), 2.5, dtype=torch.bfloat16)))

    def test_copies_config_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = self._make_model(tmp, "m1", 1.0)
            m2 = self._make_model(tmp, "m2", 1.0)
            output = tmp / "merged"

            merge_models([m1, m2], output)

            self.assertTrue((output / "config.json").exists())
            self.assertTrue((output / "tokenizer_config.json").exists())

    def test_mismatched_weights_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = self._make_model(tmp, "m1", 1.0)
            m2 = self._make_model(tmp, "m2", 1.0)
            output = tmp / "merged"

            with self.assertRaises(ValueError, msg="Number of weights"):
                merge_models([m1, m2], output, model_weights=[1.0])

    def test_zero_weights_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = self._make_model(tmp, "m1", 1.0)
            m2 = self._make_model(tmp, "m2", 1.0)
            output = tmp / "merged"

            with self.assertRaises(ValueError, msg="sum to zero"):
                merge_models([m1, m2], output, model_weights=[0.0, 0.0])

    def test_mismatched_files_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            m1 = self._make_model(tmp, "m1", 1.0)
            # m2 has a different safetensors filename
            m2 = tmp / "m2"
            m2.mkdir()
            save_file({"x": torch.zeros(4)}, str(m2 / "other.safetensors"))
            (m2 / "config.json").write_text("{}")
            output = tmp / "merged"

            with self.assertRaises(ValueError, msg="different safetensor files"):
                merge_models([m1, m2], output)


class TestMergeModelsPythia14m(unittest.TestCase):
    """Integration tests using EleutherAI/pythia-14m with seeded weight init."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.m1_path = _save_model_with_seed(seed=42, tmp=tmp, name="m1")
        cls.m2_path = _save_model_with_seed(seed=123, tmp=tmp, name="m2")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _load_first_tensor(self, model_path: Path) -> tuple[str, torch.Tensor]:
        """Load the first tensor from the first safetensors file in a model dir."""
        sf_files = get_safetensor_files(model_path)
        with safe_open(str(model_path / sf_files[0]), framework="pt") as f:
            key = list(f.keys())[0]
            return key, f.get_tensor(key)

    def test_models_are_different(self):
        """Verify seeded init actually produces different weights."""
        _, t1 = self._load_first_tensor(self.m1_path)
        _, t2 = self._load_first_tensor(self.m2_path)
        self.assertFalse(torch.allclose(t1, t2))

    def test_equal_merge_is_average(self):
        with tempfile.TemporaryDirectory() as out:
            output = Path(out) / "merged"
            merge_models([self.m1_path, self.m2_path], output)

            key, t1 = self._load_first_tensor(self.m1_path)
            _, t2 = self._load_first_tensor(self.m2_path)
            expected = (t1.to(torch.bfloat16) + t2.to(torch.bfloat16)) / 2.0

            with safe_open(str(output / get_safetensor_files(output)[0]), framework="pt") as f:
                merged = f.get_tensor(key)
            self.assertTrue(torch.allclose(merged, expected))

    def test_weighted_merge(self):
        with tempfile.TemporaryDirectory() as out:
            output = Path(out) / "merged"
            # 3:1 weighting -> 0.75 * m1 + 0.25 * m2
            merge_models([self.m1_path, self.m2_path], output, model_weights=[3.0, 1.0])

            key, t1 = self._load_first_tensor(self.m1_path)
            _, t2 = self._load_first_tensor(self.m2_path)
            expected = t1.to(torch.bfloat16) * 0.75 + t2.to(torch.bfloat16) * 0.25

            with safe_open(str(output / get_safetensor_files(output)[0]), framework="pt") as f:
                merged = f.get_tensor(key)
            self.assertTrue(torch.allclose(merged, expected))

    def test_merge_copies_config(self):
        with tempfile.TemporaryDirectory() as out:
            output = Path(out) / "merged"
            merge_models([self.m1_path, self.m2_path], output)

            self.assertTrue((output / "config.json").exists())

    def test_merged_model_loads(self):
        """Verify the merged output can be loaded as a HuggingFace model."""
        with tempfile.TemporaryDirectory() as out:
            output = Path(out) / "merged"
            merge_models([self.m1_path, self.m2_path], output)

            model = AutoModelForCausalLM.from_pretrained(output)
            self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
