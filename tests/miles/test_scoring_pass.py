"""The standalone scoring pass runs exactly when the recipe needs it, with a sampled check otherwise."""

import pytest
import torch

from open_instruct.miles import contract
from open_instruct.miles.config import CoreConfig, RunConfig, scoring_check_due, scoring_pass, stochastic_fields


def _options(**overrides):
    options = {"global_batch_size": 16, "rollout_batch_size": 4, "n_samples_per_prompt": 4}
    options.update(overrides)
    return options


@pytest.mark.parametrize(
    ("core", "options", "standalone", "reason", "steps"),
    [
        (CoreConfig(), _options(), False, "training forward", 1),
        (CoreConfig(), _options(use_rollout_logprobs=True), False, "rollout log-probabilities", 1),
        (CoreConfig(), _options(global_batch_size=8), True, "2 optimizer steps", 2),
        (CoreConfig(), _options(rollout_batch_size=16), True, "4 optimizer steps", 4),
        (CoreConfig(), _options(kl_coef=0.01), True, "kl_coef", 1),
        (CoreConfig(), {"global_batch_size": 16}, True, "unknown collection size", None),
        (CoreConfig(scoring_pass_required=True), _options(), True, "scoring_pass_required", 1),
    ],
)
def test_static_decision_follows_the_recipe(core, options, standalone, reason, steps):
    decision = scoring_pass(core, options)
    assert decision.standalone is standalone
    assert reason in decision.reason
    assert decision.optimizer_steps_per_collection == steps


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scoring_check_interval", -1, "scoring_check_interval"),
        ("scoring_check_interval", 2.0, "scoring_check_interval"),
        ("scoring_check_tolerance", float("nan"), "finite"),
        ("scoring_check_tolerance", -0.5, "finite number"),
        ("scoring_check_tolerance", True, "finite"),
        ("scoring_pass_required", "true", "boolean"),
    ],
)
def test_invalid_scoring_controls_rejected(field, value, message):
    with pytest.raises(ValueError, match=message):
        CoreConfig(**{field: value})


def test_defaults_check_first_then_every_fifty_updates():
    core = CoreConfig()
    assert (core.scoring_pass_required, core.scoring_check_interval, core.scoring_check_tolerance) == (False, 50, 1e-3)


def test_plan_reports_the_decision_and_check_controls():
    plan = RunConfig(
        CoreConfig(scoring_check_interval=20, scoring_check_tolerance=1e-4),
        {"hf_checkpoint": "/hf", "global_batch_size": 4, "rollout_batch_size": 1, "n_samples_per_prompt": 4},
    ).plan()
    assert plan["scoring_pass"] == {
        "standalone": False,
        "reason": plan["scoring_pass"]["reason"],
        "optimizer_steps_per_collection": 1,
        "check_interval": 20,
        "check_tolerance": 1e-4,
    }
    assert "training forward" in plan["scoring_pass"]["reason"]


def test_check_cadence_covers_first_update_then_interval():
    core = CoreConfig(scoring_check_interval=3)
    assert scoring_check_due(core, checks_done=0, completed_steps=0)
    for step, due in ((1, False), (2, False), (3, True), (4, False), (6, True)):
        assert scoring_check_due(core, checks_done=1, completed_steps=step) is due
    assert scoring_check_due(core, checks_done=0, completed_steps=137), "a fresh process (resume) checks first"


def test_interval_zero_keeps_only_the_first_check():
    core = CoreConfig(scoring_check_interval=0)
    assert scoring_check_due(core, checks_done=0, completed_steps=0)
    assert not scoring_check_due(core, checks_done=1, completed_steps=50)


def test_dropout_fields_are_detected_by_value():
    assert stochastic_fields({"attention_dropout": 0.1, "hidden_size": 8, "embd_pdrop": 0.0}) == ["attention_dropout"]
    assert stochastic_fields({"attention_dropout": 0.0, "hidden_dropout": 0}) == []


def _scores(*rows):
    return [torch.tensor(row, dtype=torch.float32) for row in rows]


def test_scoring_check_reduces_active_tokens_only():
    masks = [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 1])]
    standalone = _scores([-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5])
    training = _scores([-1.0, -9.0, -3.002], [-0.5, -0.5, -0.5])
    sums, maximum = contract.scoring_check(standalone, training, masks)
    assert sums.tolist() == pytest.approx([0.002, 5, 1], abs=1e-6)
    assert float(maximum) == pytest.approx(0.002, abs=1e-6)
    report = contract.validate_scoring_check(sums, maximum, tolerance=1e-3)
    assert report["active_tokens"] == 5 and report["tokens_above_edge"] == 1
    assert report["mean_abs"] == pytest.approx(0.0004, rel=1e-4)
    with pytest.raises(ValueError, match="exceeds"):
        contract.validate_scoring_check(sums, maximum, tolerance=1e-4)


@pytest.mark.parametrize(
    ("training", "message"),
    [
        (_scores([-1.0, -2.0], [-0.5, -0.5, -0.5]), "shapes differ"),
        (_scores([-1.0, float("nan"), -3.0], [-0.5, -0.5, -0.5]), "Non-finite"),
        (_scores([-1.0, -2.0, -3.0]), "one standalone and one training score"),
    ],
)
def test_scoring_check_rejects_malformed_inputs(training, message):
    masks = [torch.tensor([1, 1, 1]), torch.tensor([1, 1, 1])]
    standalone = _scores([-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5])
    with pytest.raises(ValueError, match=message):
        contract.scoring_check(standalone, training, masks)


def test_scoring_check_requires_active_tokens():
    sums, maximum = contract.scoring_check(_scores([-1.0]), _scores([-1.0]), [torch.tensor([0])])
    with pytest.raises(ValueError, match="No active tokens"):
        contract.validate_scoring_check(sums, maximum, tolerance=1e-3)
