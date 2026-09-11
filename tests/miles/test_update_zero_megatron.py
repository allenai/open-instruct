"""Preserve the original checkpoint and serving recipe for update-zero diagnostics."""

import copy
import importlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.miles import update_zero_megatron as helper

MANIFEST = Path(__file__).parents[2] / "scripts/miles/diagnostics/update-zero-megatron.json"


def prepared(manifest):
    return helper.prepare(
        manifest,
        driver="/tmp/zero-probe/update_zero_driver.py",
        output="/weka/diagnostic/megatron",
        probe_dir="/tmp/zero-probe",
    )


def test_exact_original_flags_only_change_for_diagnostic_scope():
    manifest = json.loads(MANIFEST.read_text())
    original = copy.deepcopy(manifest)
    result = prepared(manifest)
    command = result["command"]
    assert command[3] == "/tmp/zero-probe/update_zero_driver.py"
    assert command[command.index("--num-rollout") + 1] == "0"
    assert "--use-wandb" not in command
    assert "--check-weight-update-equal" in command
    assert manifest == original
    expected = list(manifest["argv"])
    expected[3] = command[3]
    for flag in result["changed_flags"]:
        if flag == "--use-wandb":
            expected.remove(flag)
        else:
            helper.replace_value(expected, flag, command[command.index(flag) + 1])
    expected.extend(result["added_arguments"])
    assert command == expected
    assert result["runtime_env"]["worker_process_setup_hook"] == manifest["runtime_env"]["worker_process_setup_hook"]
    assert result["runtime_env"]["env_vars"]["PYTHONPATH"].endswith(manifest["runtime_env"]["env_vars"]["PYTHONPATH"])


@pytest.mark.parametrize("field", ["image", "olmo_miles_source", "worker_process_setup_hook"])
def test_changed_frozen_identity_rejected(field):
    manifest = json.loads(MANIFEST.read_text())
    if field == "worker_process_setup_hook":
        manifest["runtime_env"][field] = "wrong"
    else:
        manifest[field] = "wrong"
    with pytest.raises(ValueError):
        prepared(manifest)


@pytest.mark.parametrize("output", [helper.ORIGINAL_OUTPUT, helper.ORIGINAL_OUTPUT + "/nested", "relative"])
def test_original_output_cannot_be_reused(output):
    with pytest.raises(ValueError):
        helper.prepare(
            json.loads(MANIFEST.read_text()),
            driver="/tmp/zero-probe/update_zero_driver.py",
            output=output,
            probe_dir="/tmp/zero-probe",
        )


def test_missing_full_weight_check_is_rejected():
    manifest = json.loads(MANIFEST.read_text())
    manifest["argv"].remove("--check-weight-update-equal")
    with pytest.raises(ValueError):
        prepared(manifest)


def test_original_source_bootstrap_is_pinned_and_valid_bash():
    result = prepared(json.loads(MANIFEST.read_text()))
    script = result["bootstrap_script"]
    assert f"export OLMO_MILES_EXPECTED_REVISION={helper.SOURCE}" in script
    assert "https://github.com/allenai/olmo-miles.git" in script
    assert "olmo_miles_fetch /src/olmo-miles" in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_zero_update_scheduler_uses_original_positive_horizon():
    command = prepared(json.loads(MANIFEST.read_text()))["command"]
    assert command[command.index("--num-rollout") + 1] == "0"
    assert command[command.index("--lr-decay-iters") + 1] == "100"


def test_actual_megatron_scheduler_constructs_without_optimizer_updates():
    if importlib.util.find_spec("miles") is None:
        pytest.skip("Pinned MILES/Megatron runtime required")
    model = importlib.import_module("miles.backends.megatron_utils.model")
    command = prepared(json.loads(MANIFEST.read_text()))["command"]
    args = SimpleNamespace(
        num_rollout=0,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        lr_decay_iters=None,
        lr_wsd_decay_iters=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_warmup_init=0.0,
        lr=1e-6,
        min_lr=0.0,
        lr_decay_style="constant",
        start_weight_decay=0.0,
        end_weight_decay=0.0,
        weight_decay_incr_style="constant",
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="exponential",
    )
    optimizer = SimpleNamespace(param_groups=[{}])
    with pytest.raises(AssertionError):
        model.get_optimizer_param_scheduler(args, optimizer)
    args.lr_decay_iters = int(command[command.index("--lr-decay-iters") + 1])
    scheduler = model.get_optimizer_param_scheduler(args, optimizer)
    assert args.train_iters == 0
    assert scheduler.lr_decay_steps == 1600
    assert scheduler.num_steps == 0
    assert optimizer.param_groups[0]["lr"] == 1e-6
    assert optimizer.param_groups[0]["weight_decay"] == 0.0
