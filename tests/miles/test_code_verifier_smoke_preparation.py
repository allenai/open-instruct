import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/miles/prepare_code_verifier_smoke.py"
SPEC = importlib.util.spec_from_file_location("code_verifier_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"] + "\nassistant:"

    def encode(self, text, **kwargs):
        return text.split()


def test_preparation_preserves_all_tests_without_reference_solution():
    tests = [f"assert f({i}) == {i}" for i in range(50)]
    source = {
        "id": "source-id",
        "prompt": "user: Implement f.",
        "solution": "REFERENCE_SOLUTION_MUST_NOT_ENTER_TRAINING",
        "ground_truth": [json.dumps(tests)],
    }
    row = smoke.prepare_row(source, 19, Tokenizer(), 2048)
    assert row["metadata"]["verifiers"] == [{"name": "code", "target": tests, "weight": 1.0}]
    assert row["metadata"]["source_row"] == 19
    assert "REFERENCE_SOLUTION" not in json.dumps(row)
    assert row["input"] == "Implement f.\nassistant:"


def test_preparation_keeps_stdio_target():
    tests = [{"input": "1\n", "output": "2\n"}]
    source = {"id": "stdio", "prompt": "user: Add one.", "ground_truth": [json.dumps(tests)]}
    row = smoke.prepare_row(source, 2, Tokenizer(), 2048)
    assert row["metadata"]["verifiers"][0] == {"name": "code_stdio", "target": tests, "weight": 1.0}


@pytest.mark.parametrize("labels", [[], ["[]"], ["[]", "[]"]])
def test_preparation_rejects_missing_or_ambiguous_tests(labels):
    with pytest.raises(ValueError):
        smoke.prepare_row({"id": "bad", "prompt": "user: Question", "ground_truth": labels}, 0, Tokenizer(), 20)
