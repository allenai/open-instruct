"""Score centering keeps its native weighting separate from PPO TIS."""

import dataclasses

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.test_miles_policy_refresh import configured
from open_instruct.test_miles_run_spec import document


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_async_score_centering_compiles_without_legacy_tis(tmp_path, mode):
    data = document(tmp_path) | {
        "async": {"fully_async": True},
        "trainer": {"fp32_lm_head": True},
        "miles": {"loss_type": "score_centering", "score_centering_is": mode, "rollout_top_logprobs_num": 20},
    }
    config = RunSpec.from_dict(data, config_path=tmp_path / "run.toml").compile()
    assert config.miles["use_tis"] is False
    assert config.resolved_miles()["sglang_enable_fp32_lm_head"] is True
    assert "sglang_enable_fp32_lm_head" not in config.miles
    assert "--use-tis" not in config.arguments()
    assert "--score-centering-is" in config.arguments()


def test_refresh_accepts_score_centering_tis():
    config = configured()
    dataclasses.replace(
        config,
        core=dataclasses.replace(config.core, fp32_lm_head=True),
        miles=config.miles
        | {
            "use_tis": False,
            "loss_type": "score_centering",
            "score_centering_is": "tis",
            "rollout_top_logprobs_num": 20,
        },
    ).validate()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"use_tis": True}, "use_tis"),
        ({"sglang_enable_fp32_lm_head": False}, "Score centering requires"),
        ({"rollout_top_logprobs_num": 0}, "rollout_top_logprobs_num"),
        ({"score_centering_tis_clip": 0}, "score_centering_tis_clip"),
        ({"score_centering_mis_low": 6}, "score_centering_mis_high"),
        ({"advantage_estimator": "reinforce_plus_plus"}, "grpo"),
        ({"custom_convert_samples_to_train_data_path": "custom.converter"}, "custom_convert"),
        ({"score_centering_is": "none"}, "refresh requires"),
    ],
)
def test_refresh_rejects_invalid_score_centering(change, match):
    config = configured()
    options = config.miles | {
        "use_tis": False,
        "loss_type": "score_centering",
        "score_centering_is": "tis",
        "rollout_top_logprobs_num": 20,
    }
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(
            config, core=dataclasses.replace(config.core, fp32_lm_head=True), miles=options | change
        ).validate()


def test_explicit_off_policy_correction_selects_native_score_centering_tis(tmp_path):
    data = document(tmp_path) | {
        "async": {"fully_async": True, "off_policy_correction": "tis"},
        "trainer": {"fp32_lm_head": True},
        "miles": {"loss_type": "score_centering", "rollout_top_logprobs_num": 20},
    }
    config = RunSpec.from_dict(data, config_path=tmp_path / "run.toml").compile()
    assert config.miles["score_centering_is"] == "tis"
    assert config.miles["use_tis"] is False


@pytest.mark.parametrize("trainer", [{}, {"fp32_lm_head": False}])
def test_score_centering_requires_explicit_core_fp32_head(tmp_path, trainer):
    data = document(tmp_path) | {
        "trainer": trainer,
        "miles": {"loss_type": "score_centering", "rollout_top_logprobs_num": 20, "sglang_enable_fp32_lm_head": True},
    }
    with pytest.raises(ValueError, match="requires core.fp32_lm_head=true"):
        RunSpec.from_dict(data, config_path=tmp_path / "run.toml").compile()
