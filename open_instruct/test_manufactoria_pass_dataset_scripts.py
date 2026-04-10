import argparse
import importlib.util
from pathlib import Path
from unittest.mock import Mock

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_module(relative_path: str, module_name: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


manufactoria_pass_module = load_module(
    "scripts/data/rlvr/manufactoria_pass_at_k_dataset.py", "manufactoria_pass_at_k_dataset"
)


def _build_args(**overrides):
    defaults = dict(
        no_difficulty=False,
        pass_score_threshold=1.0,
        score_threads=4,
        num_samples=32,
        use_existing_completions=True,
        manufactoria_scoring_mode="pass_rate",
        model="Qwen/Qwen3-4B-Instruct-2507",
        chat_template="from_model",
        temperature=1.0,
        top_p=1.0,
        response_length=8192,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_extract_per_test_pass_vector_uses_test_results_metadata():
    metadata = {
        "manufactoria_test_results": [
            {"test_index": 2, "passed": True},
            {"test_index": 0, "passed": False},
            {"test_index": 1, "passed": True},
        ]
    }

    assert manufactoria_pass_module._extract_per_test_pass_vector(metadata, 3) == [0.0, 1.0, 1.0]


def test_build_output_row_recomputes_full_and_per_test_metrics():
    verifier = Mock()
    verifier.side_effect = [
        Mock(
            score=1.0,
            metadata={
                "manufactoria_test_results": [{"test_index": 0, "passed": True}, {"test_index": 1, "passed": True}]
            },
        ),
        Mock(
            score=0.5,
            metadata={
                "manufactoria_test_results": [{"test_index": 0, "passed": False}, {"test_index": 1, "passed": True}]
            },
        ),
        Mock(
            score=0.5,
            metadata={
                "manufactoria_test_results": [{"test_index": 0, "passed": True}, {"test_index": 1, "passed": False}]
            },
        ),
    ]
    sample = {"messages": [{"role": "user", "content": "solve"}], "ground_truth": [[{"input": "a"}, {"input": "b"}]]}

    row = manufactoria_pass_module.build_output_row(
        sample=sample, completions=["c1", "c2", "c3"], verifier=verifier, args=_build_args(num_samples=99)
    )

    assert row["Full pass count"] == 1
    assert row["Full pass rate"] == "1/3"
    assert row["Per-test pass count"] == [2, 2]
    assert row["Per-test pass rate"] == ["2/3", "2/3"]
    assert row["difficulty"] == [1, 2]
    assert row["num_samples"] == 3
    assert "generator_model" not in row


def test_existing_completions_are_loaded_per_row():
    ds = Dataset.from_list([{"completions": ["a", "b"]}, {"completions": ["c"]}])

    assert [list(sample["completions"]) for sample in ds] == [["a", "b"], ["c"]]
