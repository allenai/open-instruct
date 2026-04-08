import importlib.util
import pathlib
import unittest


def _load_script_module():
    script_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "tokenizers" / "test_chat_templates.py"
    spec = importlib.util.spec_from_file_location("test_chat_templates_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


chat_template_script = _load_script_module()


class TestChatTemplateDebugScript(unittest.TestCase):
    def test_resolve_examples_all_preserves_declared_order(self):
        resolved = chat_template_script.resolve_examples(["multi_reasoning", "all", "basic_chat"])
        self.assertEqual(resolved, list(chat_template_script.EXAMPLES.keys()))

    def test_extract_conversations_supports_preference_rows(self):
        row = {
            "chosen": [{"role": "user", "content": "hi"}],
            "rejected": [{"role": "user", "content": "bye"}],
        }
        conversations = chat_template_script.extract_conversations(row, "demo/dpo", 3)
        self.assertEqual([conv["name"] for conv in conversations], ["dataset row 3 [chosen]", "dataset row 3 [rejected]"])

    def test_resolve_template_variants_includes_builtin_first(self):
        variants = chat_template_script.resolve_template_variants(None, [])
        self.assertEqual(variants, [("built-in", None)])

    def test_diff_rendered_outputs_reports_changed_lines(self):
        diff = chat_template_script.diff_rendered_outputs("base", "hello\nworld\n", "alt", "hello\nthere\n")
        self.assertIn("--- base", diff)
        self.assertIn("+++ alt", diff)
        self.assertIn("-world", diff)
        self.assertIn("+there", diff)


if __name__ == "__main__":
    unittest.main()
