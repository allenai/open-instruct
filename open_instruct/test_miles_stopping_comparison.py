"""CPU checks for the experiment's estimator, tie rule and acquisition controller."""

import dataclasses

import pytest

from open_instruct.miles.rollout import stopping_comparison as comparison
from open_instruct.test_miles_stopping_config import configuration


def test_negative_accuracy_difference_is_not_overridden_by_bonus():
    result = comparison.comparison([1, 1, 1, 0], [1] * 4, [10] * 4, [100] * 4, tie_bonus=0.01)
    assert result["advantage"] == -0.25
    assert result["tie_bonus"] == 0


@pytest.mark.parametrize(
    "rewards,saved,expected", [([0, 0], True, 0), ([1, 0], True, 0.01), ([1, 1], True, 0.01), ([1, 0], False, 0)]
)
def test_exact_tie_requires_accuracy_and_savings(rewards, saved, expected):
    result = comparison.comparison(rewards, rewards, [10, 10], [20, 20] if saved else [5, 5], tie_bonus=0.01)
    assert result["advantage"] == expected
    assert result["accuracy_advantage"] == 0


def test_partial_rescue_and_answer_sharpening_are_separate():
    result = comparison.comparison([1, 0, 0, 0], [0] * 4, [10] * 4, [100] * 4)
    assert result["advantage"] == 0.25
    assert comparison.centered_rewards([1, 0, 0, 0]) == [0.75, -0.25, -0.25, -0.25]
    assert comparison.centered_rewards([1, 1]) == [0, 0]


def test_taper_floor_and_repair():
    assert [comparison.tapered_rate(s, 0.125, 32, 1 / 16) for s in (0, 32, 64, 128, 999)] == [
        0.125,
        0.0625,
        0.03125,
        0.0078125,
        0.0078125,
    ]
    assert comparison.advance_rate(0.03125, 64, initial=0.125, interval=32, floor_fraction=1 / 16, risk=True) == 0.0625
    assert not comparison.risk_detected([-0.5], minimum_parents=4, margin=0.02)
    assert comparison.risk_detected([-0.2] * 4, minimum_parents=4, margin=0.02)
    assert not comparison.risk_detected([-0.8, 0.5, -0.5, 0.5], minimum_parents=4, margin=0.02)


def test_parent_weighting_is_capped_and_keeps_short_parents():
    weights = comparison.parent_weights([10, 1000, 10000], 0.25, 1000)
    assert sum(weights) == pytest.approx(1)
    assert weights[1] == weights[2]
    assert weights[0] > 0.25 / 3


def test_comparative_config_is_explicit_and_rejects_filtering():
    config = configuration()
    core = dataclasses.replace(
        config.core,
        forced_exit_mode="comparative",
        forced_exit_positions=2,
        forced_exit_guidance="first_token",
        filter_zero_std_groups=False,
    )
    config = dataclasses.replace(
        config,
        core=core,
        miles=config.miles
        | {"rollout_function_path": "open_instruct.miles.rollout.comparative_exits.ComparativeExitRollout"},
    )
    config.validate()
    with pytest.raises(ValueError, match="zero-std"):
        dataclasses.replace(config, core=dataclasses.replace(core, filter_zero_std_groups=True)).validate()


@pytest.mark.parametrize(
    "changes",
    [
        {"forced_exit_trials": 1},
        {"forced_exit_initial_trials": 1},
        {"forced_exit_positions": 0},
        {"forced_exit_guidance": "full_tag"},
        {"forced_exit_parent_probability": 0},
        {"forced_exit_risk_min_parents": 1},
        {"forced_exit_tie_bonus": -0.1},
    ],
)
def test_comparative_invalid_config(changes):
    options = dict(forced_exit_mode="comparative", forced_exit_positions=2, forced_exit_guidance="first_token")
    with pytest.raises(ValueError):
        base = configuration()
        core = dataclasses.replace(base.core, **(options | changes))
        dataclasses.replace(base, core=core).validate()
