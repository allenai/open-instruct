import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from open_instruct.model_utils import uses_olmo3_generation_config


class TestUsesOlmo3GenerationConfig(unittest.TestCase):
    def test_olmo_template_name(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "User: {{ content }}"
        self.assertTrue(uses_olmo3_generation_config("olmo123", tokenizer))

    def test_hybrid_model_type_with_tokenizer_default(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        model = SimpleNamespace(config=SimpleNamespace(model_type="olmo_hybrid"))
        self.assertTrue(uses_olmo3_generation_config("tokenizer_default", tokenizer, model))

    def test_resolved_template_contains_im_end(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "<|im_start|>assistant\n{{ content }}<|im_end|>"
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama", _name_or_path="/weka/HYBRID_INSTRUCT_SFT"))
        self.assertTrue(uses_olmo3_generation_config("tokenizer_default", tokenizer, model))

    def test_non_olmo_template_without_im_end(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "{{ messages }}"
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
        self.assertFalse(uses_olmo3_generation_config("tulu", tokenizer, model))
        self.assertFalse(uses_olmo3_generation_config("tokenizer_default", tokenizer, model))


if __name__ == "__main__":
    unittest.main()
