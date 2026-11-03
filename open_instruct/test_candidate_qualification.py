"""Qualification must detect adversarial rewards and preserve checks/config paths."""

import asyncio
import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.data.rl_readiness import local_coverage

from open_instruct.miles.rewards import candidate_verifiers as cv
from open_instruct.miles.rewards import rewards


def test_relative_coverage_catalog_is_resolved_before_writing_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "gpu-prompts.jsonl").write_text('{"input":"question"}\n')
    (catalog / "gpu-verifiers.json").write_text("{}")
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "config.json").write_text("{}")
    observed = []
    monkeypatch.setattr(local_coverage.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(local_coverage.ray, "shutdown", lambda: None)
    monkeypatch.setattr(local_coverage.runpy, "run_module", lambda *args, **kwargs: observed.append(sys.argv.copy()))
    monkeypatch.setattr(sys, "argv", [])
    monkeypatch.setenv("SGLANG_EXTERNAL_MODEL_PACKAGE", "test")
    local_coverage.main(SimpleNamespace(catalog=Path("catalog"), fixture=fixture, output=tmp_path / "different/run"))
    spec = json.loads(Path(observed[0][2]).read_text())
    assert spec["miles"]["prompt_data"] == str(catalog / "gpu-prompts.jsonl")
    assert spec["core"]["reward_config"] == str(catalog / "gpu-verifiers.json")


def test_qualification_check_survives_python_optimization():
    script = "from scripts.data.rl_readiness import checks; checks.require(False, 'missing rollout')"
    result = subprocess.run([sys.executable, "-O", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert "ValueError: missing rollout" in result.stderr


@pytest.fixture
def qualification_module():
    pytest.importorskip("miles")
    pytest.importorskip("rdkit")
    return importlib.import_module("scripts.data.rl_readiness.qualify")


@pytest.mark.parametrize("task", ["base_conversion", "leg_counting", "simple_equations"])
def test_qualification_rejects_old_substring_scorer(qualification_module, task, monkeypatch):
    library = pytest.importorskip("reasoning_gym")
    target = library.create_dataset(task, seed=17, size=1)[0]
    spec = {"name": "rg", "target": target}
    entry = {
        "row": {"input": target["question"], "label": target, "metadata": {"verifiers": [spec]}},
        "gold": target["answer"],
        "negative": "unrelated",
        "registry": {"rg": {"factory": "open_instruct.miles.rewards.reasoning_gym.ReasoningGymVerifier"}},
    }
    verifier = qualification_module.reasoning_gym.ReasoningGymVerifier()
    monkeypatch.setattr(rewards, "_registry", lambda path: {"rg": verifier})
    args = SimpleNamespace(
        reward_key=None,
        olmo_core=SimpleNamespace(reward_config="unused", reward_zero_truncated=True, reward_final_answer_only=False),
    )
    receipt = asyncio.run(qualification_module.controls(copy.deepcopy(entry), args))
    assert receipt["gold_containing_wrong"] == [0.0, 0.0, 0.0]

    class OldVerifier:
        async def async_call(self, tokens, prediction, label, **kwargs):
            return cv.Result(library.get_score_answer_fn(task)(answer=prediction, entry=label))

    monkeypatch.setattr(rewards, "_registry", lambda path: {"rg": OldVerifier()})
    with pytest.raises(ValueError, match="Gold-containing wrong answers earned reward"):
        asyncio.run(qualification_module.controls(copy.deepcopy(entry), args))


def test_qualification_rejects_timeout_fallback(qualification_module, monkeypatch):
    class TimeoutVerifier:
        async def async_call(self, tokens, prediction, label, **kwargs):
            if prediction == "gold":
                return cv.Result(1.0)
            return cv.Result(0.0, diagnostics={"kind": "format", "status": "timeout", "error": "regex timed out"})

    monkeypatch.setattr(rewards, "_registry", lambda path: {"format": TimeoutVerifier()})
    entry = {
        "row": {"input": "question", "label": {}, "metadata": {"verifiers": [{"name": "format", "target": {}}]}},
        "gold": "gold",
        "negative": "wrong",
        "registry": {"format": {"factory": "open_instruct.miles.rewards.candidate_verifiers.CandidateVerifier"}},
    }
    args = SimpleNamespace(
        reward_key=None,
        olmo_core=SimpleNamespace(reward_config="unused", reward_zero_truncated=True, reward_final_answer_only=False),
    )
    with pytest.raises(ValueError, match="fallback zero cannot qualify"):
        asyncio.run(qualification_module.controls(entry, args))
