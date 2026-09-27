"""CPU-only configuration guards for experimental forced-exit training."""

import dataclasses
from pathlib import Path

import pytest
import tomllib

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec


def configuration():
    return RunConfig(
        CoreConfig(
            reward_zero_truncated=True,
            reward_final_answer_only=True,
            forced_exit_positions=5,
            forced_exit_trials=3,
            scoring_pass_required=True,
            router_aux_loss_weight=0,
            router_z_loss_weight=0,
        ),
        dict(
            hf_checkpoint="model",
            global_batch_size=16,
            rollout_batch_size=1,
            n_samples_per_prompt=16,
            rollout_function_path="open_instruct.miles.rollout.forced_exits.ForcedExitRollout",
            loss_type="custom_loss",
            custom_loss_function_path="miles.backends.core_utils.stopping.policy_loss",
        ),
    )


def test_valid_stopping_config():
    configuration().validate()


@pytest.mark.parametrize(
    "key,value",
    [
        ("n_samples_per_prompt", 1),
        ("fully_async", True),
        ("calculate_per_token_loss", True),
        ("use_rollout_logprobs", True),
        ("rewards_normalization", False),
        ("normalize_advantages", True),
        ("custom_reward_post_process_path", "exclude"),
        ("loss_type", "policy_loss"),
        ("partial_rollout", True),
        ("rollout_function_path", "other"),
    ],
)
def test_reject_unsafe_stopping_options(key, value):
    config = configuration()
    with pytest.raises(ValueError):
        dataclasses.replace(config, miles=config.miles | {key: value}).validate()


@pytest.mark.parametrize(
    "key,value",
    [
        ("reward_zero_truncated", False),
        ("reward_final_answer_only", False),
        ("scoring_pass_required", False),
        ("publication_mode", "refresh"),
        ("router_aux_loss_weight", 0.01),
        ("forced_exit_parents", 17),
    ],
)
def test_reject_unsafe_core_stopping_options(key, value):
    config = configuration()
    with pytest.raises(ValueError):
        dataclasses.replace(config, core=dataclasses.replace(config.core, **{key: value})).validate()


def test_disabled_stopping_preserves_default_training_options():
    baseline = RunConfig(CoreConfig(), {"hf_checkpoint": "model", "n_samples_per_prompt": 4, "global_batch_size": 8})
    baseline.validate()
    options = baseline.resolved_miles()
    assert baseline.core.forced_exit_positions == 0
    assert baseline.core.forced_exit_probe_interval == 0
    assert baseline.core.reward_zero_truncated is False
    assert baseline.core.reward_final_answer_only is False
    assert "rollout_function_path" not in options
    assert "custom_loss_function_path" not in options
    assert "loss_type" not in options


def test_disable_guidance_keeps_reward_policy_when_requested():
    enabled = configuration()
    options = {
        k: v
        for k, v in enabled.miles.items()
        if k not in {"rollout_function_path", "custom_loss_function_path", "loss_type"}
    }
    disabled = dataclasses.replace(
        enabled, core=dataclasses.replace(enabled.core, forced_exit_positions=0), miles=options
    )
    disabled.validate()
    assert disabled.core.reward_zero_truncated and disabled.core.reward_final_answer_only


@pytest.mark.parametrize("hook", ["rollout_function_path", "custom_loss_function_path"])
def test_disabled_stopping_rejects_leftover_hooks(hook):
    enabled = configuration()
    disabled = RunConfig(CoreConfig(), {"hf_checkpoint": "model", hook: enabled.miles[hook]})
    with pytest.raises(ValueError, match="remove the forced-exit"):
        disabled.validate()


def test_documented_enable_overlay_compiles(tmp_path):
    guide = Path(__file__).parents[1] / "docs/miles/forced-exit-stopping.md"
    overlay = tomllib.loads(guide.read_text().split("```toml\n")[1].split("```")[0])
    spec = RunSpec.from_dict(
        {
            "schema_version": 1,
            "name": "stopping-doc-check",
            "model": {"source": "model", "format": "hf"},
            "output": {"root": str(tmp_path)},
            "data": {"tasks": [{"task": "gsm8k", "train_count": 8}]},
            "training": {"filter_zero_std_groups": False},
            "inference": {"samples_per_prompt": 4, "max_response_length": 8192, "max_context_length": 10240},
            **overlay,
        }
    )
    compiled = spec.compile()
    compiled.validate()
    assert compiled.core.forced_exit_positions == 5
    assert compiled.miles["custom_loss_function_path"].endswith("stopping.policy_loss")
