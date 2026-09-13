import copy
import hashlib
import json
from pathlib import Path

import pytest
from scripts.miles import launch_readiness_cpu, readiness_cpu


def fixture():
    target = {"name": "math", "target": "42", "weight": 0.5}
    row = {
        "input": "question",
        "label": "42",
        "metadata": {
            "verifiers": [target],
            "run_prompt_token_ids_sha256": hashlib.sha256((json.dumps([1, 2]) + "\n").encode()).hexdigest(),
        },
    }
    sample = {
        "prompt": "question",
        "label": "42",
        "tokens": [1, 2, 3],
        "response_length": 1,
        "weight_versions": [2],
        "reward": 0.5,
        "metadata": {
            "verifiers": [copy.deepcopy(target)],
            "reward_components": [{"name": "math", "score": 1.0, "weight": 0.5}],
        },
    }
    return sample, row


def test_retained_sample_checks_identity_reward_and_policy():
    sample, row = fixture()
    assert readiness_cpu.verify_sample(sample, row, version=3, max_lag=1, response_cap=10) == 2
    for field, value, message in (
        ("tokens", [9, 2, 3], "token identity"),
        ("reward", 1.0, "accounting"),
        ("weight_versions", [1], "behavior policy"),
        ("weight_versions", [2, 3], "behavior policy"),
        ("label", "43", "Prompt/label"),
    ):
        changed = copy.deepcopy(sample)
        changed[field] = value
        with pytest.raises(ValueError, match=message):
            readiness_cpu.verify_sample(changed, row, version=3, max_lag=1, response_cap=10)


def test_reward_weight_is_checked_independently_of_total():
    sample, row = fixture()
    sample["metadata"]["reward_components"][0]["weight"] = 1.0
    sample["reward"] = 1.0
    with pytest.raises(ValueError, match="weight changed"):
        readiness_cpu.verify_sample(sample, row, version=3, max_lag=1, response_cap=10)


def test_cpu_jobs_always_use_saturn_and_embed_exact_source():
    spec = launch_readiness_cpu.specification(
        "image", "audit", [Path("/weka/oe-training-default/a b")], b"print('hello')"
    )
    task = spec["tasks"][0]
    assert task["constraints"] == {"cluster": ["ai2/saturn"]}
    assert task["resources"]["gpuCount"] == 0
    assert "set -euo pipefail" in task["arguments"][0]
    assert "'/weka/oe-training-default/a b'" in task["arguments"][0]
    assert hashlib.sha256(b"print('hello')").hexdigest() in spec["description"]


def test_long_preparation_launcher_does_not_pass_internal_mode_to_script():
    spec = launch_readiness_cpu.specification("image", "prepare-long", [Path("/model"), Path("/fixture")], b"pass")
    assert spec["tasks"][0]["arguments"][0].endswith("python /output/readiness_cpu.py /model /fixture")
