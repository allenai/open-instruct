"""Reward cleanup preserves transport policies and lightweight preparation imports."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from open_instruct.miles.rewards import code_rewards, general_judge


def test_reward_clients_keep_distinct_retry_policies():
    code = code_rewards._get_session()
    judge = general_judge._get_session()
    assert code is code_rewards._get_session()
    assert judge is general_judge._get_session()
    assert code is not judge
    assert code.get_adapter("https://").max_retries.total == 3
    assert judge.get_adapter("https://").max_retries.total == 0


def test_reward_settings_preserve_argument_precedence_and_environment_fallback(monkeypatch):
    monkeypatch.setenv("OI_MILES_CODE_MAX_EXECUTION_TIME", "2.5")
    monkeypatch.setenv("OI_MILES_CODE_APPLY_PERF_PENALTY", "yes")
    monkeypatch.setenv("OI_MILES_JUDGE_API_BASE", "http://judge")
    monkeypatch.setenv("OI_MILES_JUDGE_MAX_TOKENS", "128")
    args = SimpleNamespace(code_max_execution_time=1.5, code_apply_perf_penalty=False, llm_judge_max_tokens=64)
    code = code_rewards.code_verifier_config(args)
    assert code.max_execution_time == 1.5 and code.apply_perf_penalty is False
    assert general_judge.general_judge_config(args).max_tokens == 64
    code = code_rewards.code_verifier_config(SimpleNamespace())
    assert code.max_execution_time == 2.5 and code.apply_perf_penalty is True
    assert general_judge.general_judge_config(SimpleNamespace()).max_tokens == 128


@pytest.mark.parametrize("name", ["general-quality", "general-quality_ref", "general-web_instruct_general_verifier"])
def test_shared_judge_templates_retain_qualified_hashes(name):
    prompt, digest = general_judge.build_judge_prompt(name, query="Query", prediction="Answer", target="Target")
    assert "Query" in prompt and "Answer" in prompt
    assert len(digest) == 64  # build_judge_prompt checks the existing qualification hash.


def test_preparation_and_service_helpers_do_not_import_grading_or_training_stack():
    code = """
import importlib.abc
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('miles', 'torch', 'transformers', 'open_instruct.ground_truth_utils')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Unexpected planning dependency: ' + fullname)

sys.meta_path.insert(0, BlockHeavyImports())
from open_instruct.miles.datasets import run_data
from open_instruct.miles.rewards import service, task_verifiers
run_data.validate_data({'tasks': [{'task': 'multiplication', 'train_count': 1}]})
assert run_data.MultiplicationVerifier is task_verifiers.MultiplicationVerifier
assert run_data.ManifestIFVerifier is task_verifiers.ManifestIFVerifier
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30)
