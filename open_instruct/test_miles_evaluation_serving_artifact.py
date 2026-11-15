"""Independent evaluator policy transport, isolation, and resume provenance."""

import base64
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import evaluation_runner as runner
from open_instruct.miles.evaluation import serving_artifact

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path):
    lock = ROOT / "runtime/miles/runtime.lock.json"
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
                "model_config_sha256": hashlib.sha256(b"{}").hexdigest(),
            }
        )
    )
    return {
        "kda_policy": {
            "path": str(policy),
            "sha256": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "core_compat": "off",
        },
        "gpus": 1,
        "provider_kwargs": {"kv_cache_dtype": "nvfp4", "attention_backend": "trtllm_mha"},
    }


def freeze(config):
    config["serving_artifact"] = serving_artifact.build(config["kda_policy"])
    return {"evaluation": config, "tasks": [{"task": "aime", "generation": {}, "scoring": {}}], "checkpoint": "/hf"}


def test_transport_is_exact_and_importable_without_training_package(config, monkeypatch):
    receipt = freeze(config)
    monkeypatch.setenv("PYTHONPATH", "/untrusted/training")
    monkeypatch.setenv("OI_MILES_KDA_POLICY", "/untrusted/policy")
    policy_env, env = runner.gpu_environments(receipt)
    assert policy_env["PYTHONPATH"] == env["PYTHONPATH"]
    assert env["PYTHONPATH"] != "/untrusted/training"
    assert env["OLMO_SGLANG_CORE_COMPAT"] == "off"
    assert env["SGLANG_RETURN_ORIGINAL_LOGPROB"] == "1"
    assert env["OLMO_EVAL_PROVIDER_INIT_TIMEOUT_SECONDS"] == "2400"
    root = Path(env["PYTHONPATH"])
    assert not (root / "open_instruct").exists()
    assert (root / "oi_miles_serving/kda_dispatch.py").read_bytes() == (
        ROOT / "open_instruct/miles/execution/kda_dispatch.py"
    ).read_bytes()
    # The original external policy need not exist when a receipt is resumed.
    Path(config["kda_policy"]["path"]).unlink()
    args = runner.command(receipt, "/out")
    assert args[:3] == [sys.executable, "-m", "oi_miles_serving.evaluator_startup_budget"]
    assert "provider.kwargs.model_registration=oi_miles_serving.kda_dispatch:register" in args
    assert "provider.kwargs.kv_cache_dtype=nvfp4" in args
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            "from oi_miles_serving import kda_dispatch as k; import sys; assert 'open_instruct' not in sys.modules; assert k.RUNTIME_LOCK.is_file()",
        ],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


@pytest.mark.parametrize("change", ["bytes", "path", "runtime"])
def test_tampered_bundle_is_rejected_before_execution(config, change):
    receipt = freeze(config)
    files = config["serving_artifact"]["files"]
    if change == "bytes":
        files["oi_miles_serving/kda_dispatch.py"]["base64"] = base64.b64encode(b"bad").decode()
    elif change == "path":
        files["../../escape"] = files.pop("oi_miles_serving/kda_dispatch.py")
    else:
        raw = b"{}"
        files["runtime.lock.json"] = {
            "base64": base64.b64encode(raw).decode(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    with pytest.raises(ValueError):
        runner.command(receipt, "/out")
    with pytest.raises(ValueError):
        runner.gpu_environments(receipt)


def test_policy_must_match_config_and_missing_bundle_fails(config):
    with pytest.raises(ValueError, match="missing"):
        runner.validate_serving_artifact(config)
    freeze(config)
    config["kda_policy"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="policy/artifact"):
        runner.validate_serving_artifact(config)
    with pytest.raises(ValueError, match="configured hash"):
        serving_artifact.build(config["kda_policy"])


def test_resume_identity_freezes_source_and_policy(config):
    receipt = {**freeze(config), "update": 100, "group_id": "aime"}
    previous = runner.resume_identity(receipt)
    changed = copy.deepcopy(receipt)
    changed["evaluation"]["serving_artifact"]["files"]["oi_miles_serving/kda_dispatch.py"]["sha256"] = "0" * 64
    assert runner.resume_identity(changed) != previous
    receipt["experiment_id"] = "replacement"
    assert runner.resume_identity(receipt) == previous


def test_no_policy_removes_inherited_serving_environment(monkeypatch):
    for key in ("PYTHONPATH", "OI_MILES_KDA_POLICY", "OI_MILES_KDA_RUNTIME_LOCK"):
        monkeypatch.setenv(key, "/training")
    receipt = {"evaluation": {"gpus": 1}, "tasks": [{}]}
    for env in runner.gpu_environments(receipt):
        assert not {"PYTHONPATH", "OI_MILES_KDA_POLICY", "OI_MILES_KDA_RUNTIME_LOCK"}.intersection(env)


def test_snapshot_configuration_must_match_frozen_export_policy(config, tmp_path):
    receipt = freeze(config)
    receipt["checkpoint"] = str(tmp_path)
    (tmp_path / "config.json").write_text('{"changed_shape": true}')
    with pytest.raises(ValueError, match="snapshot configuration"):
        runner.run_evaluation(receipt)


def test_native_policy_option_is_parsed_and_rejects_external_tasks(config):
    document = RunSpec.load(ROOT / "configs/miles/examples/tiny/three-node.toml").to_dict()
    document["evaluation"]["tasks"] = [document["evaluation"]["tasks"][0]]
    document["evaluation"]["kda_policy"] = config["kda_policy"]
    parsed = RunSpec.from_dict(document)
    assert parsed.evaluation["kda_policy"] == config["kda_policy"]
    document["evaluation"]["kda_policy"]["core_compat"] = "on"
    with pytest.raises(InputError, match="core_compat"):
        RunSpec.from_dict(document)
    document["evaluation"]["kda_policy"]["core_compat"] = "off"
    document["evaluation"]["tasks"] = [
        {"task": "scicode", "kind": "external", "arguments": {"split": "validation", "h5py_host_path": "/data"}}
    ]
    with pytest.raises(InputError, match="native evaluator tasks only"):
        RunSpec.from_dict(document)
