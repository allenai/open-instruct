"""Maintained recipes and omitted objective settings share the demo baseline."""

from pathlib import Path

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec


@pytest.mark.parametrize("name,batch,lag,buffer", [("node", 512, 6, 1), ("three-node", 1024, 12, 4)])
def test_learning_baseline_and_worker_controls(name, batch, lag, buffer):
    config = RunSpec.load(f"configs/miles/examples/{name}.toml").compile()
    options = config.resolved_miles()
    assert options["n_samples_per_prompt"] == 8
    assert options["global_batch_size"] == batch
    assert config.core.max_policy_lag == lag
    assert options["async_data_buffer_capacity_factor"] == buffer
    assert config.core.reward_zero_truncated is True
    assert options["calculate_per_token_loss"] is False
    assert "--calculate-per-token-loss" not in config.arguments()
    assert options["use_distributed_post"] is True
    assert options["http_post_actors_per_node"] == 16
    assert options["http_post_num_cpus"] == 1
    assert options["rollout_executor_num_cpus"] == 2
    assert options["train_env_vars"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_omitted_settings_cannot_revert_the_objective(tmp_path):
    config = RunSpec.from_dict(
        dict(
            schema_version=1,
            name="baseline",
            model=dict(source="/fixture/hf"),
            output=dict(root=str(tmp_path)),
            data=dict(tasks=[dict(task="gsm8k", train_count=8, eval_count=4)]),
        ),
        config_path=tmp_path / "baseline.toml",
    ).compile()
    assert config.core.reward_zero_truncated is True
    assert config.miles["calculate_per_token_loss"] is False
    assert CoreConfig().reward_zero_truncated is True
    assert RunConfig(CoreConfig(), {}).resolved_miles()["calculate_per_token_loss"] is False


def test_explicit_experiments_remain_possible():
    config = RunSpec.load(
        Path("configs/miles/examples/node.toml"),
        ["core.reward_zero_truncated=false", "miles.calculate_per_token_loss=true"],
    ).compile()
    assert config.core.reward_zero_truncated is False
    assert "--calculate-per-token-loss" in config.arguments()
