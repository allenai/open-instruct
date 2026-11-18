"""The checkpoint override must not bypass refresh topology or cache checks."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import constraints, refresh_deployment
from open_instruct.test_miles_policy_refresh import configured


def deployment():
    return {
        "sglang": [
            {
                "name": "default",
                "model_path": "/serialized/fp8",
                "update_weights": True,
                "server_groups": [{"worker_type": "regular", "num_gpus": 3, "num_gpus_per_engine": 1}],
            }
        ]
    }


def config(tmp_path, raw):
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(raw))
    base = configured()
    return dataclasses.replace(base, miles=base.miles | {"sglang_config": str(path), "rollout_num_gpus": 3})


def test_refresh_accepts_only_checkpoint_change(tmp_path):
    run = config(tmp_path, deployment())
    run.validate()
    assert run.core.publication_mode == "refresh"
    assert '"publication_mode": "refresh"' in run.arguments()[-1]


def test_native_validation_preserves_rollout_gpu_count(tmp_path):
    run = config(tmp_path, deployment())
    fields = constraints.native_values(SimpleNamespace(**run.miles), set())
    refresh_deployment.validate(fields)


@pytest.mark.parametrize("change", ["frozen", "tp", "cache", "gpus", "extra_model", "placeholder", "unknown"])
def test_refresh_rejects_overrides(tmp_path, change):
    raw = deployment()
    model = raw["sglang"][0]
    group = model["server_groups"][0]
    if change == "frozen":
        model["update_weights"] = False
    elif change == "tp":
        group["num_gpus_per_engine"] = 2
    elif change == "cache":
        group["overrides"] = {"disable_radix_cache": False}
    elif change == "gpus":
        group["num_gpus"] = 4
    elif change == "extra_model":
        raw["sglang"].append(model.copy())
    elif change == "placeholder":
        group["worker_type"] = "placeholder"
    else:
        model["unknown"] = True
    with pytest.raises(ValueError, match="refresh.*single-turn"):
        config(tmp_path, raw).validate()
