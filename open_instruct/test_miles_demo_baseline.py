"""Maintained token-mean recipes preserve compatible omitted library defaults."""

from pathlib import Path

import pytest
import tomllib

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec


@pytest.mark.parametrize("name,batch,lag,buffer", [("node", 2048, 12, 1), ("three-node", 2048, 12, 4)])
def test_learning_baseline_and_worker_controls(name, batch, lag, buffer):
    config = RunSpec.load(f"configs/miles/examples/tiny/{name}.toml").compile()
    options = config.resolved_miles()
    assert options["n_samples_per_prompt"] == 8
    assert options["global_batch_size"] == batch
    assert options["rollout_batch_size"] == 256
    assert options["lr"] == 6e-6
    assert options["lr_decay_style"] == "constant"
    assert options["lr_warmup_iters"] == 0
    assert config.core.max_policy_lag == lag
    assert options["async_data_buffer_capacity_factor"] == buffer
    assert config.core.reward_zero_truncated is True
    assert options["calculate_per_token_loss"] is True
    assert "--calculate-per-token-loss" in config.arguments()
    assert options["use_distributed_post"] is True
    assert options["http_post_actors_per_node"] == 16
    assert options["http_post_num_cpus"] == 1
    assert options["rollout_executor_num_cpus"] == 2
    assert options["train_env_vars"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


@pytest.mark.parametrize(
    "section,topology_fields",
    [
        ("training", set()),
        ("core", {"checkpoint_keep_every"}),
        ("inference", {"gpus"}),
        ("trainer", {"gpus", "expert_parallel_size"}),
    ],
)
def test_learning_demos_share_training_parameters(section, topology_fields):
    # Dataset, judge and evaluation workloads intentionally differ. Changing GPU
    # topology must not silently change the learning recipe. Artifact milestone
    # cadence follows each template's evaluation/retention needs.
    runs = [
        tomllib.loads(Path(f"configs/miles/examples/tiny/{name}.toml").read_text()) for name in ("node", "three-node")
    ]
    parameters = [{k: v for k, v in run[section].items() if k not in topology_fields} for run in runs]
    assert parameters[0] == parameters[1]


def test_omitted_library_settings_preserve_response_mean_compatibility(tmp_path):
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
        Path("configs/miles/examples/tiny/node.toml"),
        ["core.reward_zero_truncated=false", "miles.calculate_per_token_loss=false"],
    ).compile()
    assert config.core.reward_zero_truncated is False
    assert config.miles["calculate_per_token_loss"] is False
    assert "--calculate-per-token-loss" not in config.arguments()


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).resolve().parents[1] / "configs/miles/examples/small").glob("*.toml")),
    ids=lambda path: path.stem,
)
def test_small_recipes_use_tiny_learning_baseline(path):
    # Match the final Tiny recipe in #2271, including the Small mechanics run.
    # Hardware, serving pools, run length and qualification controls are separate.
    config = RunSpec.load(path).compile()
    options = config.resolved_miles()
    assert options["rollout_batch_size"] == 256
    assert options["n_samples_per_prompt"] == 8
    assert options["global_batch_size"] == 2048
    assert options["lr"] == 6e-6
    assert options["lr_decay_style"] == "constant"
    assert options["lr_warmup_iters"] == 0
    assert options["calculate_per_token_loss"] is True
    assert "--calculate-per-token-loss" in config.arguments()
    assert options["grpo_std_normalization"] is False
    # The seven- and eight-node recipes limit routing-replay payloads in the Ray object store.
    assert options["async_data_buffer_capacity_factor"] == (1.5 if path.stem in ("seven-node", "eight-node") else 4.0)
    assert config.plan()["scoring_pass"]["optimizer_steps_per_collection"] == 1
