"""Check prepared benchmark inputs without downloading models or datasets."""

import copy
from pathlib import Path

import pytest
from scripts.miles import prepare_boxed_math as prep

from open_instruct.miles.configuration.run_spec import RunSpec


class Tokenizer:
    chat_template = "test model template"

    def apply_chat_template(self, messages, **kwargs):
        return "|".join(message["role"] + ":" + message["content"] for message in messages) + "|assistant:"

    def encode(self, text, **kwargs):
        return list(text.encode())


def test_solution_is_removed_and_native_template_is_used():
    row = {
        "messages": [
            {"role": "system", "content": "Solve carefully."},
            {"role": "user", "content": "What is 6 times 7?"},
            {"role": "assistant", "content": "SECRET SOLUTION"},
        ],
        "ground_truth": ["42"],
    }
    original = copy.deepcopy(row)
    rows, excluded = prep.prepare_rows("gsm8k", [row], Tokenizer())
    assert row == original
    assert not excluded
    assert rows[0]["input"].startswith("system:Solve carefully.|user:What is 6 times 7?")
    assert rows[0]["input"].endswith(prep.INSTRUCTION + "|assistant:")
    assert "SECRET" not in rows[0]["input"]
    assert rows[0]["label"] == "42"
    assert rows[0]["metadata"]["verifiers"] == [{"name": "gsm8k", "target": "42", "weight": 1.0}]
    assert rows[0]["metadata"]["rendering_template_sha256"] == prep.sha(Tokenizer.chat_template.encode())


def test_dolci_overlap_and_instruction_are_not_duplicated():
    rows = [
        {"prompt": "held  out", "ground_truth": "7"},
        {"prompt": "retained\n\n" + prep.INSTRUCTION, "ground_truth": "\\frac{1}{2}"},
    ]
    result, excluded = prep.prepare_rows("dolci", rows, Tokenizer(), held_out=["held out"])
    assert excluded == [0]
    assert len(result) == 1
    assert result[0]["input"].count(prep.INSTRUCTION) == 1
    assert result[0]["label"] == "\\frac{1}{2}"
    assert result[0]["metadata"]["verifiers"][0]["name"] == "math"


def test_invalid_or_overlong_prompts_fail():
    with pytest.raises(ValueError, match="prompt tokens"):
        prep.prepare_rows("gsm8k", [{"question": "Q", "ground_truth": "1"}], Tokenizer(), max_prompt_tokens=1)
    with pytest.raises(ValueError, match="scalar target"):
        prep.prepare_rows("dolci", [{"prompt": "Q", "ground_truth": ["1", "2"]}], Tokenizer())
    with pytest.raises(ValueError, match="No training rows"):
        prep.prepare_rows("dolci", [{"prompt": "Q", "ground_truth": "1"}], Tokenizer(), held_out=["Q"])


def test_half_node_recipes_preserve_topology_and_separate_aime_years():
    root = Path(__file__).resolve().parents[1] / "configs/miles/examples/tiny"
    gsm8k = RunSpec.load(root / "half-node-gsm8k.toml")
    dolci = RunSpec.load(root / "half-node-dolci-math.toml")
    for run in (gsm8k, dolci):
        config = run.compile()
        assert config.arguments()
        assert run.sections["trainer"]["gpus"] == 1
        assert run.sections["inference"]["gpus"] == 3
        assert config.miles["lr"] == 1e-6
        assert config.miles["use_tis"]
        assert not config.core.reward_final_answer_only
        assert "tasks" not in run.data
        assert [point["update"] for point in run.plan()["evaluation"]["milestones"]] == list(range(0, 1001, 100))
    assert [task["task"] for task in dolci.evaluation["tasks"]] == ["aime_2024:pass_at_32", "aime_2025:pass_at_32"]
    assert all(task["generation"]["num_samples"] == 32 for task in dolci.evaluation["tasks"])
