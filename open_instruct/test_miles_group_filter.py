"""The public filtering switch must reach the native rollout path in every mode."""

import dataclasses
from pathlib import Path

import pytest

from open_instruct.miles.configuration.config import (
    EXCLUDE_TRUNCATED,
    EXCLUDE_UNGRADED,
    FINISHED_STD_FILTER,
    GRADED_STD_FILTER,
    ZERO_STD_FILTER,
    CoreConfig,
    RunConfig,
)
from open_instruct.miles.configuration.run_spec import RunSpec


def configured(mode="barrier", enabled=True, **miles):
    return RunConfig(
        CoreConfig(publication_mode=mode, max_policy_lag=2, filter_zero_std_groups=enabled),
        dict(
            hf_checkpoint="/fixture/hf",
            global_batch_size=4,
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            fully_async=True,
            use_miles_router=True,
            use_tis=True,
            sglang_cuda_graph_backend_decode="disabled",
            sglang_cuda_graph_backend_prefill="disabled",
        )
        | miles,
    )


@pytest.mark.parametrize("mode", ["barrier", "engine_drain", "refresh"])
@pytest.mark.parametrize("enabled", [True, False])
def test_native_arguments_and_plan_agree(mode, enabled):
    config = configured(mode, enabled)
    argv = config.arguments()
    plan = config.plan()
    assert plan["core"]["filter_zero_std_groups"] is enabled
    # Ungraded responses are excluded from their group by default, so the paired
    # graded-response filter replaces the native zero-std filter.
    assert argv[argv.index("--custom-reward-post-process-path") + 1] == EXCLUDE_UNGRADED
    assert plan["miles"]["custom_reward_post_process_path"] == EXCLUDE_UNGRADED
    if enabled:
        assert argv[argv.index("--dynamic-sampling-filter-path") + 1] == GRADED_STD_FILTER
        assert plan["miles"]["dynamic_sampling_filter_path"] == GRADED_STD_FILTER
    else:
        assert "--dynamic-sampling-filter-path" not in argv
        assert "dynamic_sampling_filter_path" not in plan["miles"]
    # Resolution must not mutate the caller's options or interfere with an opt-out.
    assert "dynamic_sampling_filter_path" not in config.miles


def test_raw_and_structured_defaults_and_researcher_override(tmp_path):
    assert CoreConfig().filter_zero_std_groups is True
    payload = {
        "schema_version": 1,
        "name": "filtering",
        "model": {"source": "/fixture/hf"},
        "output": {"root": str(tmp_path / "run")},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 8, "eval_count": 4}]},
    }
    run = RunSpec.from_dict(payload, config_path=tmp_path / "run.toml")
    assert run.compile().plan()["miles"]["dynamic_sampling_filter_path"] == GRADED_STD_FILTER
    assert run.compile().plan()["miles"]["custom_reward_post_process_path"] == EXCLUDE_UNGRADED
    omitting = RunSpec.from_dict(
        payload | {"training": {"exclude_ungraded": False}}, config_path=tmp_path / "run.toml"
    )
    assert omitting.compile().plan()["miles"]["dynamic_sampling_filter_path"] == ZERO_STD_FILTER
    assert "custom_reward_post_process_path" not in omitting.compile().plan()["miles"]
    assert run.compile().miles["n_samples_per_eval_prompt"] == 1
    disabled = RunSpec.from_dict(
        payload | {"training": {"filter_zero_std_groups": False}}, config_path=tmp_path / "run.toml"
    )
    assert "--dynamic-sampling-filter-path" not in disabled.compile().arguments()
    with pytest.raises(ValueError, match="[Cc]onflict"):
        RunSpec.from_dict(
            payload | {"training": {"filter_zero_std_groups": False}, "core": {"filter_zero_std_groups": True}},
            config_path=tmp_path / "run.toml",
        )


def test_single_response_requires_explicit_opt_out():
    with pytest.raises(ValueError, match="samples_per_prompt > 1"):
        configured(n_samples_per_prompt=1, rollout_batch_size=4).validate()
    configured(enabled=False, n_samples_per_prompt=1, rollout_batch_size=4).validate()


@pytest.mark.parametrize("mode", ["barrier", "engine_drain", "refresh"])
@pytest.mark.parametrize("enabled", [True, False])
def test_truncation_exclusion_selects_matching_group_filter(mode, enabled):
    config = configured(mode, enabled, custom_reward_post_process_path=EXCLUDE_TRUNCATED)
    config = dataclasses.replace(config, core=dataclasses.replace(config.core, reward_zero_truncated=False))
    plan = config.plan()
    assert plan["miles"].get("dynamic_sampling_filter_path") == (FINISHED_STD_FILTER if enabled else None)
    if enabled:
        argv = config.arguments()
        assert argv[argv.index("--dynamic-sampling-filter-path") + 1] == FINISHED_STD_FILTER


def test_ungraded_exclusion_opt_out_and_runtime_limits():
    omitting = RunConfig(dataclasses.replace(configured().core, exclude_ungraded=False), configured().miles)
    assert "--custom-reward-post-process-path" not in omitting.arguments()
    assert omitting.plan()["miles"]["dynamic_sampling_filter_path"] == ZERO_STD_FILTER
    # The exclusion hooks support single-segment rollouts only; partial rollouts
    # keep the group-omission behaviour instead of an injected default.
    assert "custom_reward_post_process_path" not in configured(partial_rollout=True).plan()["miles"]
    custom = configured(enabled=False, custom_reward_post_process_path="custom.fn")
    custom = RunConfig(dataclasses.replace(custom.core, reward_zero_truncated=False), custom.miles)
    assert custom.plan()["miles"]["custom_reward_post_process_path"] == "custom.fn"
    assert "dynamic_sampling_filter_path" not in custom.plan()["miles"]


def test_zero_truncation_rewards_allow_ungraded_but_not_truncation_exclusion():
    configured(custom_reward_post_process_path=EXCLUDE_UNGRADED).validate()
    with pytest.raises(ValueError, match="Zero-truncation"):
        configured(custom_reward_post_process_path=EXCLUDE_TRUNCATED).validate()
    overlong = configured(custom_reward_post_process_path=EXCLUDE_TRUNCATED)
    RunConfig(dataclasses.replace(overlong.core, reward_zero_truncated=False), overlong.miles).validate()


def test_native_filter_conflicts_are_not_silently_overridden():
    configured(dynamic_sampling_filter_path=ZERO_STD_FILTER).validate()
    configured(dynamic_sampling_filter_path=GRADED_STD_FILTER).validate()
    with pytest.raises(ValueError, match="conflicts"):
        configured(enabled=False, dynamic_sampling_filter_path=GRADED_STD_FILTER).validate()
    with pytest.raises(ValueError, match="conflicts"):
        configured(enabled=False, dynamic_sampling_filter_path=ZERO_STD_FILTER).validate()
    with pytest.raises(ValueError, match="custom filter"):
        configured(dynamic_sampling_filter_path="custom.filter").validate()
    configured(enabled=False, dynamic_sampling_filter_path="custom.filter").validate()
    for mode in ("refresh", "engine_drain"):
        with pytest.raises(ValueError, match="only the built-in"):
            configured(mode, enabled=False, dynamic_sampling_filter_path="custom.filter").validate()


@pytest.mark.parametrize("value", ["true", 1, None])
def test_switch_requires_a_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        dataclasses.replace(CoreConfig(), filter_zero_std_groups=value)


@pytest.mark.parametrize("name", ["dev", "node", "three-node"])
def test_maintained_examples_explain_their_filtering_choice(name):
    path = Path(__file__).resolve().parents[1] / "configs/miles/examples" / f"{name}.toml"
    run = RunSpec.load(path)
    enabled = name in ("node", "three-node")
    assert run.compile().core.filter_zero_std_groups is enabled
    assert ("--dynamic-sampling-filter-path" in run.compile().arguments()) is enabled
