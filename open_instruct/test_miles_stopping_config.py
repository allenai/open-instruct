"""CPU-only configuration guards for experimental forced-exit training."""

import dataclasses

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig


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
            custom_loss_function_path="open_instruct.miles.training.stopping.policy_loss",
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
