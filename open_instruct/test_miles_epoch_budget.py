"""Epoch budgets must reach native MILES without a default rollout override."""

import copy

import pytest

from open_instruct.miles.configuration import run_spec


def test_num_epoch_uses_native_budget_instead_of_the_default_rollout_limit(tmp_path):
    document = {
        "schema_version": 1,
        "name": "epochs",
        "model": {"source": "model"},
        "output": {"root": str(tmp_path)},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 8}]},
        "training": {"num_epoch": 2},
    }
    spec = run_spec.RunSpec.from_dict(document, config_path=tmp_path / "run.toml")
    compiled = spec.compile()
    assert compiled.miles["num_epoch"] == 2 and "num_rollout" not in compiled.miles
    assert "--num-epoch" in compiled.arguments() and "--num-rollout" not in compiled.arguments()
    assert compiled.core.save_final_checkpoint
    assert "save_interval" not in compiled.miles
    spec.plan()  # Includes warnings/checkpoint planning before dataset size is known.
    for section in ("training", "miles"):
        conflicting = copy.deepcopy(document)
        conflicting.setdefault(section, {})["num_rollouts" if section == "training" else "num_rollout"] = 7
        with pytest.raises(ValueError, match="Choose num_epoch or num_rollouts"):
            run_spec.RunSpec.from_dict(conflicting).compile()
    document["training"]["save_checkpoints"] = False
    assert not run_spec.RunSpec.from_dict(document).compile().core.save_final_checkpoint
    for count in (0, -1, True):
        document["training"]["num_epoch"] = count
        with pytest.raises(ValueError, match="num_epoch"):
            run_spec.RunSpec.from_dict(document).compile()
