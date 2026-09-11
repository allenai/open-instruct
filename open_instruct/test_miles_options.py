"""Public configuration regressions without importing the GPU runtime."""

import json
from pathlib import Path

import pytest

from open_instruct.miles import __main__, options
from open_instruct.miles.config import CoreConfig, RunConfig


def config(**kwargs):
    return RunConfig(CoreConfig(), {"hf_checkpoint": "model", "global_batch_size": 8, **kwargs})


def test_boolean_flags_and_inverse_names():
    argv = config(use_wandb=False, colocate=False, grpo_std_normalization=False).arguments()
    assert "--no-use-wandb" not in argv
    assert "--colocate" not in argv
    assert "--disable-grpo-std-normalization" in argv
    assert config(disable_grpo_std_normalization=True).arguments() == config(grpo_std_normalization=False).arguments()
    assert "--disable-rollout-global-dataset" in config(rollout_global_dataset=False).arguments()
    with pytest.raises(ValueError, match="Multiple spellings"):
        config(grpo_std_normalization=False, disable_grpo_std_normalization=True).arguments()


def test_structured_json_and_nargs_are_distinct():
    argv = config(
        train_env_vars={"NCCL_DEBUG": "WARN"},
        eval_prompt_data=["gsm8k", "/data/eval.jsonl"],
        sglang_json_model_override_args={"max_position_embeddings": 8192},
    ).arguments()
    assert json.loads(argv[argv.index("--train-env-vars") + 1]) == {"NCCL_DEBUG": "WARN"}
    assert argv[argv.index("--eval-prompt-data") + 1 : argv.index("--eval-prompt-data") + 3] == [
        "gsm8k",
        "/data/eval.jsonl",
    ]
    assert json.loads(argv[argv.index("--sglang-json-model-override-args") + 1])["max_position_embeddings"] == 8192


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_wandb": "false"},
        {"lr": float("nan")},
        {"lr": "1e-6"},
        {"rollout_temperature": True},
        {"eval_prompt_data": []},
        {"rollout_top_p": {}},
        {"rollout_batch_siz": 4},
        {"sglang_cuda_graph_backend_decode": "nonsense"},
        {"update_weight_transfer_mode": "disk-delta"},
        {"async_save": True},
        {"no_save_optim": True},
        {"reset_optimizer_states": True},
        {"override_lr_scheduler": True},
        {"use_checkpoint_lr_scheduler": False},
        {"compute_advantages_and_returns": False},
        {"skip_actor_forward_only": True},
        {"keep_old_actor": True},
        {"dp_replicate_size": 2},
        {"deterministic_mode": True},
        {"lora_train_only": True},
        {"lora_rank": 8},
        {"save_hf": "/data/hf/{rollout_id}"},
        {"update_weights_interval": 2},
        {"debug_skip_weight_update": True},
        {"gradient_checkpointing": False},
        {"max_tokens_per_gpu": 256},
        {"optimizer": "sgd"},
        {"fp16": True},
        {"debug_disable_optimizer": True},
        {"data_source_path": "some.module.Class"},
        {"max_weight_staleness": 2},
    ],
)
def test_invalid_or_ignored_options_fail_early(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs).arguments()


def test_overrides_and_shape(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        '[miles]\nhf_checkpoint="model"\nglobal_batch_size=8\nrollout_batch_size=2\nn_samples_per_prompt=4\n'
    )
    loaded = RunConfig.load(
        path,
        [
            "miles.lr=1e-6",
            "miles.use_wandb=false",
            'miles.wandb_project="trial"',
            "core.max_policy_lag=1",
            "miles.rollout_batch_size=4",
        ],
    )
    plan = loaded.plan()
    assert plan["shape"]["optimizer_steps_per_collection"] == 2
    assert plan["shape"]["samples_per_collection"] == 16
    assert plan["miles"]["lr"] == 1e-6
    assert plan["runtime_validated"] is False
    assert "--no-use-wandb" not in plan["argv"]
    with pytest.raises(ValueError, match="quote strings"):
        RunConfig.load(path, ["miles.wandb_project=unquoted"])
    with pytest.raises(ValueError, match="Overrides must"):
        RunConfig.load(path, ["lr=1e-6"])


def test_snapshot_tracks_runtime_pins():
    root = Path(__file__).parents[1]
    schema = json.loads((root / "open_instruct/miles/options.json").read_text())
    lock = json.loads((root / "runtime/miles/runtime.lock.json").read_text())
    assert schema["sources"] == lock["sources"]


def test_option_strings_are_preserved():
    assert options.encode_options({"wandb_run_name": "--test"}) == ["--wandb-run-name=--test"]
    with pytest.raises(ValueError, match="did you mean rollout_batch_size"):
        options.encode_options({"rollout_batch_siz": 3})


def test_all_starting_profiles_compile():
    root = Path(__file__).parents[1]
    paths = sorted((root / "configs/miles/profiles").glob("*.toml"))
    assert len(paths) >= 3
    for path in paths:
        assert RunConfig.load(path).plan()["argv"]


def test_public_plan_cli(tmp_path, monkeypatch, capsys):
    path = tmp_path / "run.toml"
    path.write_text('[miles]\nhf_checkpoint="model"\nglobal_batch_size=8\n')
    monkeypatch.setattr("sys.argv", ["miles", "plan", str(path), "--set", "miles.use_wandb=false"])
    __main__.main()
    assert json.loads(capsys.readouterr().out)["miles"]["use_wandb"] is False


def test_row_specialization_is_explicit_and_validated():
    assert CoreConfig().row_specialization == "static"
    value = RunConfig(CoreConfig(row_specialization="dynamic"), {"hf_checkpoint": "model", "global_batch_size": 8})
    assert value.plan()["core"]["row_specialization"] == "dynamic"
    with pytest.raises(ValueError, match="row_specialization"):
        CoreConfig(row_specialization="auto")


def test_supported_resume_and_snapshot_options_compile():
    argv = config(
        no_save_optim=False,
        reset_optimizer_states=False,
        override_lr_scheduler=False,
        use_checkpoint_lr_scheduler=True,
        compute_advantages_and_returns=True,
        skip_actor_forward_only=False,
        dp_replicate_size=1,
        deterministic_mode=False,
        lora_train_only=False,
        lora_rank=0,
        eval_hf_dir="/data/snapshots",
    ).arguments()
    assert "--eval-hf-dir" in argv
    assert "--no-save-optim" not in argv
    assert "--disable-compute-advantages-and-returns" not in argv
