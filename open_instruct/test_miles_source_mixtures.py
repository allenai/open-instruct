"""CPU preparation, configuration, and exact-rule grading for source mixtures."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct import test_miles_run_data, test_miles_run_spec
from open_instruct.miles.configuration.config import CoreConfig
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.datasets import run_data, synlogic
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import rewards, sudoku

environment = test_miles_run_data.environment


def data_config():
    return {
        "seed": 17,
        "sources": [
            {
                "name": "arithmetic",
                "fraction": 0.9,
                "tasks": [{"task": "multiplication", "train_count": 10, "eval_count": 2}],
            },
            {"name": "puzzles", "fraction": 0.1, "tasks": [{"task": "synlogic-sudoku", "train_count": 1}]},
        ],
    }


def solution():
    return [[(3 * r + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]


def puzzle():
    answer = solution()
    givens = copy.deepcopy(answer)
    givens[0][0] = "X"
    return {
        "data_source": "sudoku",
        "extra_info": {
            "index": "puzzle-id",
            "game_data_str": json.dumps(
                {"difficulty": 3, "metadata": {"original_sudoku": givens, "complete_sudoku": answer}}
            ),
        },
    }


@pytest.fixture
def source_rows(monkeypatch):
    rows = synlogic.sudoku_rows([{"data_source": "other"}, puzzle()])
    monkeypatch.setattr(run_data, "_source_rows", lambda name: rows)
    return rows


def test_separate_sources_provenance_registry_resume_and_compilation(environment, source_rows, tmp_path):
    data = data_config()
    result = test_miles_run_data.prepare(environment, data)
    manifest = json.loads(Path(result["prompt_mixture"]).read_text())
    assert [source["rows"] for source in manifest["sources"]] == [10, 1]
    assert [source["fraction"] for source in manifest["sources"]] == [0.9, 0.1]
    assert len(result["train_sources"]) == 2
    assert not (environment[1] / "train.jsonl").exists()
    row = json.loads(Path(result["train_sources"][1]).read_text())
    assert row["metadata"]["original_row"] == 1
    assert row["metadata"]["source_revision"] == synlogic.REVISION
    assert row["metadata"]["source_config"] == "hard"
    assert row["label"] == source_rows[0]["ground_truth"]
    assert set(json.loads(Path(result["reward_config"]).read_text())) == {"multiplication", "r1_format", "sudoku"}
    assert test_miles_run_data.prepare(environment, data) == result
    doc = test_miles_run_spec.document(tmp_path) | {"data": data}
    spec = RunSpec.from_dict(doc, config_path=tmp_path / "run.toml")
    planned, compiled = spec.compile(), spec.compile(result)
    assert planned.core.prompt_provider == compiled.core.prompt_provider == "mixture"
    assert compiled.core.prompt_mixture == result["prompt_mixture"]
    assert compiled.miles["prompt_data"] == result["train_sources"][0]
    assert compiled.miles["eval_prompt_data"] == result["eval_prompt_data"]
    Path(result["train_sources"][1]).write_text("changed")
    with pytest.raises(InputError, match="Prepared artifact changed"):
        test_miles_run_data.prepare(environment, data)


@pytest.mark.parametrize("fractions", [[0.8, 0.1], [1, 1], [True, 0], [float("nan"), 0.1], [-0.1, 1.1], [0, 1]])
def test_invalid_fractions_fail_at_plan_time(tmp_path, fractions):
    data = data_config()
    for source, fraction in zip(data["sources"], fractions, strict=True):
        source["fraction"] = fraction
    with pytest.raises(InputError, match="fraction"):
        test_miles_run_spec.spec(tmp_path, data=data)


@pytest.mark.parametrize("change", [{"name": "arithmetic"}, {"name": "../bad"}, {"seed": 2}, {"sources": []}])
def test_invalid_source_contract(tmp_path, change):
    data = data_config()
    data["sources"][1].update(change)
    with pytest.raises(InputError):
        test_miles_run_spec.spec(tmp_path, data=data)


def test_relative_paths_no_eval_and_provider_conflict(tmp_path):
    data = {
        "sources": [{"name": "dolci", "fraction": 1, "prompt_data": "dolci.jsonl", "reward_config": "verifiers.json"}]
    }
    spec = test_miles_run_spec.spec(tmp_path, data=data)
    assert spec.data["sources"][0]["prompt_data"] == str(tmp_path / "dolci.jsonl")
    assert "eval_prompt_data" not in spec.compile().miles
    with pytest.raises(InputError, match="Conflicting settings"):
        test_miles_run_spec.spec(tmp_path, data=data, core={"prompt_provider": "remind"}).compile()


def test_cross_source_heldout_overlap_is_rejected(environment, source_rows, tmp_path):
    # Existing rows can be supplied directly, but a second source cannot leak a held-out prompt.
    result = test_miles_run_data.prepare(environment, data_config())
    data = {
        "sources": [
            {
                "name": "a",
                "fraction": 0.5,
                "prompt_data": result["train_sources"][0],
                "reward_config": result["reward_config"],
                "eval_prompt_data": ["heldout", result["train_sources"][1]],
            },
            {
                "name": "b",
                "fraction": 0.5,
                "prompt_data": result["train_sources"][1],
                "reward_config": result["reward_config"],
            },
        ]
    }
    with pytest.raises(InputError, match="overlap"):
        run_data.prepare_data(data, environment[0], tmp_path / "overlap", max_prompt_length=2000, seed=17)
    data["sources"][0].pop("eval_prompt_data")
    registry = json.loads(Path(result["reward_config"]).read_text())
    registry["sudoku"]["factory"] = "different.Verifier"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(registry))
    data["sources"][1]["reward_config"] = str(changed)
    with pytest.raises(InputError, match="conflicting reward registry"):
        run_data.prepare_data(data, environment[0], tmp_path / "conflict", max_prompt_length=2000, seed=17)


def grade(answer, givens=None):
    return asyncio.run(sudoku.SudokuVerifier().async_call([], answer, givens or [[0] * 9 for _ in range(9)])).score


def test_sudoku_checks_constraints_instead_of_matching_reference():
    answer = solution()
    text = f"Reasoning first. <answer>{json.dumps(answer)}</answer>"
    assert grade(text) == 1
    assert grade(text, answer) == 1
    swapped = [answer[1], answer[0], *answer[2:]]
    assert grade(f"<answer>{json.dumps(swapped)}</answer>") == 1  # Another valid solution for blank givens.
    assert grade(text, swapped) == 0  # But it violates these givens.
    bad = copy.deepcopy(answer)
    bad[0] = bad[1]
    assert grade(f"<answer>{json.dumps(bad)}</answer>") == 0
    # Latin square with valid rows/columns but invalid boxes.
    latin = [[(r + c) % 9 + 1 for c in range(9)] for r in range(9)]
    assert grade(f"<answer>{json.dumps(latin)}</answer>") == 0


@pytest.mark.parametrize("answer", ["", "[[1]]", "__import__('os').system('false')", "[true]", "[1.0]", "[" * 2000])
def test_sudoku_malformed_answers_are_zero(answer):
    assert grade(f"<answer>{answer}</answer>") == 0


def test_sudoku_answer_format_and_target_errors():
    text = f"<answer>{json.dumps(solution())}</answer>"
    assert grade(text + text) == 0
    assert grade(text[:-9]) == 0
    assert grade(json.dumps(solution())) == 0
    for value in (True, 1.0, 10, 0):
        bad = solution()
        bad[0][0] = value
        assert grade(f"<answer>{json.dumps(bad)}</answer>") == 0
    with pytest.raises(ValueError, match="target"):
        asyncio.run(sudoku.SudokuVerifier().async_call([], text, [[1]]))


def test_source_adapter_rejects_broken_reference():
    row = puzzle()
    game = json.loads(row["extra_info"]["game_data_str"])
    game["metadata"]["complete_sudoku"][0][0] = 2
    row["extra_info"]["game_data_str"] = json.dumps(game)
    with pytest.raises(InputError, match="invalid supplied Sudoku"):
        synlogic.sudoku_rows([row])


def test_sudoku_runs_through_registered_reward_bridge(environment, source_rows):
    data = {"tasks": [{"task": "synlogic-sudoku", "train_count": 1}]}
    result = test_miles_run_data.prepare(environment, data)
    row = json.loads(Path(result["prompt_data"]).read_text())
    args = SimpleNamespace(olmo_core=CoreConfig(reward_config=result["reward_config"]))
    sample = SimpleNamespace(
        prompt=row["input"],
        response=f"<answer>{json.dumps(solution())}</answer>",
        tokens=[],
        response_length=0,
        metadata=row["metadata"],
    )
    assert asyncio.run(rewards.registered_reward(args, sample)) == 1
    assert sample.metadata["verifier_diagnostics"]["sudoku"]["status"] == "completed"
    sample.response = "I cannot solve this."
    assert asyncio.run(rewards.registered_reward(args, sample)) == 0
