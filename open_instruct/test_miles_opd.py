"""CPU checks for OPD configuration and allocation boundaries."""

import asyncio
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles import launch, opd_config, opd_launch, opd_prepare, opd_runtime, rewards, specs
from open_instruct.miles.errors import InputError

CONFIG = Path(__file__).resolve().parents[1] / "configs/miles/opd/qwen35-4b-tiny.toml"
PREPARED = {
    "model": "/assets/Qwen3.5-4B-851bf6e806ef",
    "teacher": "/assets/Qwen3.5-9B-c20223623576",
    "data": {"prompt_data": "/assets/data/train.jsonl", "eval_prompt_data": ["gsm8k", "/assets/data/eval.jsonl"]},
}


def native(spec):
    arguments = opd_runtime.native_arguments(spec, PREPARED, "/assets/ckpt", "http://127.0.0.1:1/generate", [])
    values = {}
    flags = set()
    index = 0
    while index < len(arguments):
        key = arguments[index]
        if index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            values.setdefault(key, arguments[index + 1])
            index += 2
        else:
            flags.add(key)
            index += 1
    return values, flags


def test_opd_dispatch_and_gpu_isolation():
    spec = specs.load(CONFIG)
    assert isinstance(spec, opd_config.OPDRunSpec)
    assert spec.allocation()["roles"] == {"trainer": [0, 1], "student": [2], "teacher": [3]}
    task = launch.specification("test-image", spec)["tasks"][0]
    assert task["resources"]["gpuCount"] == 4
    assert task["context"]["autoResume"] is False
    assert task["constraints"] == {"cluster": ["ai2/holmes"]}
    assert {"name": "WANDB_MODE", "value": "offline"} in task["envVars"]
    assert specs.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    assert spec.document["model"]["revision"] == opd_config.REVISIONS["Qwen/Qwen3.5-4B"]
    assert spec.document["model"]["architecture"] == "qwen3.5-4B"


def test_cpu_preparation_requires_saturn():
    with pytest.raises(InputError, match="ai2/saturn"):
        specs.load(CONFIG, ['training.phase="prepare"'])
    spec = specs.load(CONFIG, ['training.phase="prepare"', 'launch.cluster="ai2/saturn"'])
    task = launch.specification("test-image", spec)["tasks"][0]
    assert task["resources"]["gpuCount"] == 0
    assert task["constraints"] == {"cluster": ["ai2/saturn"]}


@pytest.mark.parametrize(
    "override",
    [
        'trainer.backend="olmo-core"',
        "launch.auto_resume=true",
        "distillation.log_prob_top_k=10",
        "distillation.task_reward_weight=0.5",
        'model.source="Qwen/Qwen3.5-1.7B"',  # no pinned revision and no architecture profile
        'model.architecture="qwen2.5-7B"',
        'teacher.revision="main"',
        "trainer.gpus=3",
        "inference.gpus=3 inference.tensor_parallel_size=2",
        "launch.gpus_per_replica=8",
        "trainer.gpus=8",
        "training.eval_interval=-1",
        'tracking.wandb_mode="online"',
        'tracking.wandb_mode="verbose"',
    ],
)
def test_unsupported_paths_fail_before_launch(override):
    with pytest.raises(InputError):
        specs.load(CONFIG, override.split(" "))


def test_mounts_and_credentials_checked_before_build():
    document = specs.load(CONFIG).to_dict()
    document["launch"]["weka_mounts"] = []
    with pytest.raises(InputError, match="WEKA"):
        launch.specification("test-image", specs.from_dict(document))
    document = copy.deepcopy(specs.load(CONFIG).to_dict())
    document["launch"]["env"]["HF_TOKEN"] = "placeholder"
    with pytest.raises(InputError, match="secrets"):
        launch.specification("test-image", specs.from_dict(document))


def test_opd_task_data_needs_one_registered_task_with_evaluation():
    document = specs.load(CONFIG).to_dict()
    document["data"]["tasks"] = [{"task": "math", "train_count": 16, "eval_count": 8}]
    assert specs.from_dict(document).document["data"]["tasks"][0]["task"] == "math"
    for tasks in (
        [{"task": "gsm8k", "train_count": 16}],
        [{"task": "gsm8k", "train_count": 16, "eval_count": 8}, {"task": "math", "train_count": 16}],
    ):
        document["data"]["tasks"] = tasks
        with pytest.raises(InputError, match="eval_count"):
            specs.from_dict(document)
    document["data"]["tasks"] = [{"task": "dapo", "train_count": 16, "eval_count": 8}]
    with pytest.raises(InputError, match="data.tasks\\[0\\].task"):
        specs.from_dict(document)


def test_opd_accepts_prerendered_prompts_with_registry(tmp_path):
    document = specs.load(CONFIG).to_dict()
    document["data"] = {
        "prompt_data": "data/train.jsonl",
        "eval_prompt_data": ["dapo_math_holdout", "data/dapo_math_holdout.jsonl", "math_500", "data/math_500.jsonl"],
        "reward_config": "data/verifiers.json",
    }
    spec = specs.from_dict(document, config_path=tmp_path / "run.toml")
    assert spec.document["data"]["prompt_data"] == str(tmp_path / "data/train.jsonl")
    assert spec.document["data"]["eval_prompt_data"][1] == str(tmp_path / "data/dapo_math_holdout.jsonl")
    for missing in ("eval_prompt_data", "reward_config"):
        broken = copy.deepcopy(document)
        del broken["data"][missing]
        with pytest.raises(InputError, match="prompt_data"):
            specs.from_dict(broken, config_path=tmp_path / "run.toml")


def test_registered_verifiers_score_held_out_samples(tmp_path):
    registry = tmp_path / "verifiers.json"
    registry.write_text(
        json.dumps(
            {
                "gsm8k": {"factory": "open_instruct.ground_truth_utils.GSM8KVerifier"},
                "math": {"factory": "open_instruct.ground_truth_utils.MathVerifier"},
            }
        )
    )

    async def run():
        results = []
        for name, response, target in (
            ("gsm8k", "The answer is 7. #### 7", "7"),
            ("math", "So the value is \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
            ("math", "So the value is \\boxed{3}.", "\\frac{1}{2}"),
        ):
            sample = SimpleNamespace(
                tokens=[1, 2, 3],
                response_length=2,
                response=response,
                prompt="q",
                metadata={"verifiers": [{"name": name, "target": target, "weight": 1.0}], "query": "q"},
            )
            results.append(await rewards.score(sample, str(registry)))
        return results

    assert asyncio.run(run()) == [1.0, 1.0, 0.0]


def test_opd_honors_shared_memory_setting():
    spec = specs.load(CONFIG, ['launch.shared_memory="48 GiB"'])
    task = launch.specification("test-image", spec)["tasks"][0]
    assert task["resources"]["sharedMemory"] == "48 GiB"


def test_local_checkpoints_resolve_against_the_run_file(tmp_path):
    teacher = tmp_path / "teacher-step_100"
    teacher.mkdir()
    document = specs.load(CONFIG).to_dict()
    document["teacher"] = {"source": "./teacher-step_100"}
    document["model"] = {"source": str(tmp_path / "student"), "architecture": "qwen3.5-4B"}
    spec = specs.from_dict(document, config_path=tmp_path / "run.toml")
    assert spec.document["teacher"]["source"] == str(teacher.resolve())
    assert spec.document["teacher"]["revision"] == ""
    assert spec.document["model"]["source"] == str((tmp_path / "student").resolve())
    document["teacher"]["revision"] = opd_config.REVISIONS["Qwen/Qwen3.5-9B"]
    with pytest.raises(InputError, match="must not specify revision"):
        specs.from_dict(document, config_path=tmp_path / "run.toml")
    document["teacher"]["revision"] = ""
    del document["model"]["architecture"]
    with pytest.raises(InputError, match="model.architecture"):
        specs.from_dict(document, config_path=tmp_path / "run.toml")


def test_default_native_arguments_match_the_exercised_prototype():
    values, flags = native(specs.load(CONFIG))
    assert values["--save-interval"] == "1"
    assert values["--eval-interval"] == "2"
    assert values["--n-samples-per-eval-prompt"] == "1"
    assert values["--eval-temperature"] == "0.0"
    assert values["--wandb-project"] == "open-instruct-opd"
    assert values["--wandb-mode"] == "offline"
    assert values["--actor-num-gpus-per-node"] == "2"
    assert values["--num-gpus-per-node"] == "3"
    assert values["--rollout-num-gpus"] == "1"
    assert values["--tensor-model-parallel-size"] == "2"
    assert "--use-wandb" in flags
    assert "--use-rollout-logprobs" not in flags
    assert "--wandb-team" not in values


def test_relaxed_training_and_topology_settings_reach_native_arguments():
    spec = specs.load(
        CONFIG,
        [
            "training.num_rollouts=100",
            "training.save_interval=10",
            "training.eval_interval=20",
            "trainer.gpus=4",
            "inference.gpus=2",
            "inference.tensor_parallel_size=2",
            "teacher.gpus=2",
            "inference.max_running_requests=64",
            "inference.eval_temperature=0.6",
            "inference.eval_top_p=0.8",
            "inference.eval_max_response_length=1024",
            "inference.eval_samples_per_prompt=4",
            "training.optimizer_steps_per_rollout=4",
            'optimizer.lr_decay_style="cosine"',
            "optimizer.lr_warmup_iters=10",
            "optimizer.min_lr=1e-7",
            "optimizer.weight_decay=0.01",
            "optimizer.adam_beta2=0.999",
            'training.loss_aggregation="token"',
            "distillation.use_rollout_logprobs=true",
            'tracking.wandb_mode="disabled"',
            'model.source="Qwen/Qwen3.5-9B"',
        ],
    )
    assert spec.allocation() == {
        "replicas": 1,
        "gpus_per_replica": 8,
        "ray_gpus": 6,
        "roles": {"trainer": [0, 1, 2, 3], "student": [4, 5], "teacher": [6, 7]},
    }
    assert spec.document["model"]["architecture"] == "qwen3.5-9B"
    assert launch.specification("test-image", spec)["tasks"][0]["resources"]["gpuCount"] == 8
    assert any("topology" in warning for warning in spec.plan()["warnings"])
    values, flags = native(spec)
    assert values["--save-interval"] == "10"
    assert values["--eval-interval"] == "20"
    assert values["--num-rollout"] == "100"
    assert values["--eval-temperature"] == "0.6"
    assert values["--eval-top-p"] == "0.8"
    assert values["--eval-max-response-len"] == "1024"
    assert values["--rollout-top-p"] == "1.0"
    assert values["--n-samples-per-eval-prompt"] == "4"
    # 4 prompts x 2 samples per rollout split into 4 optimizer steps of 2 samples.
    assert values["--global-batch-size"] == "2"
    assert values["--lr-decay-style"] == "cosine"
    assert values["--lr-warmup-iters"] == "10"
    assert values["--min-lr"] == "1e-07"
    assert values["--lr-decay-iters"] == "400"
    assert values["--weight-decay"] == "0.01"
    assert values["--adam-beta1"] == "0.9"
    assert values["--adam-beta2"] == "0.999"
    assert "--calculate-per-token-loss" in flags
    assert values["--sglang-max-running-requests"] == "64"
    assert values["--sglang-max-total-tokens"] == str(2048 * 64)
    assert values["--actor-num-gpus-per-node"] == "4"
    assert values["--num-gpus-per-node"] == "6"
    assert values["--rollout-num-gpus"] == "2"
    assert values["--rollout-num-gpus-per-engine"] == "2"
    assert "--use-rollout-logprobs" in flags
    assert "--use-wandb" not in flags
    assert "--wandb-mode" not in values


def test_default_schedule_is_one_constant_lr_step_per_rollout():
    values, flags = native(specs.load(CONFIG))
    assert values["--global-batch-size"] == "8"
    assert values["--lr-decay-style"] == "constant"
    assert values["--min-lr"] == "0.0"
    assert values["--rollout-top-p"] == "1.0"
    assert values["--eval-max-response-len"] == values["--rollout-max-response-len"]
    assert "--lr-decay-iters" not in values
    assert values["--weight-decay"] == "0.0"
    assert values["--adam-beta2"] == "0.98"
    assert "--calculate-per-token-loss" not in flags


def test_qwen3_replication_models_resolve_to_upstream_profiles():
    spec = specs.load(CONFIG, ['model.source="Qwen/Qwen3-4B-Base"', 'teacher.source="Qwen/Qwen3-8B"'])
    assert spec.document["model"]["architecture"] == "qwen3-4B"
    assert spec.document["model"]["revision"] == opd_config.REVISIONS["Qwen/Qwen3-4B-Base"]
    assert spec.document["teacher"]["revision"] == opd_config.REVISIONS["Qwen/Qwen3-8B"]


@pytest.mark.parametrize(
    "override, message",
    [
        ("training.optimizer_steps_per_rollout=3", "must divide"),
        ("inference.eval_top_p=1.5", "inference.eval_top_p"),
        ("inference.eval_max_response_length=4096", "exceed eval_max_response_length"),
        ('optimizer.lr_decay_style="step"', "optimizer.lr_decay_style"),
        ("optimizer.min_lr=1e-3", "min_lr must not exceed"),
        ("optimizer.weight_decay=-0.1", "optimizer.weight_decay"),
        ("optimizer.adam_beta2=1.0", "optimizer.adam_beta2"),
        ('training.loss_aggregation="mean"', "training.loss_aggregation"),
        (["inference.top_p=0.9", "distillation.use_rollout_logprobs=true"], "inference.top_p must be 1.0"),
    ],
)
def test_schedule_and_sampling_knobs_are_validated(override, message):
    with pytest.raises(InputError, match=message):
        specs.load(CONFIG, override if isinstance(override, list) else [override])


def test_online_tracking_requires_a_wandb_secret():
    document = specs.load(CONFIG).to_dict()
    document["tracking"] = {"wandb_mode": "online", "wandb_project": "opd", "wandb_entity": "allenai-team1"}
    document["launch"]["secrets"] = {"WANDB_API_KEY": "kevinfarhat_WANDB_API_KEY"}
    spec = specs.from_dict(document)
    values, _ = native(spec)
    assert values["--wandb-mode"] == "online"
    assert values["--wandb-team"] == "allenai-team1"
    assert values["--wandb-project"] == "opd"
    task = launch.specification("test-image", spec)["tasks"][0]
    assert {"name": "WANDB_MODE", "value": "online"} in task["envVars"]
    assert {"name": "WANDB_API_KEY", "secret": "kevinfarhat_WANDB_API_KEY"} in task["envVars"]


def test_repository_profiles_render_through_the_miles_loader():
    miles = Path(os.environ.get("MILES_SOURCE", "")) / "miles/utils/external_utils/model_args_utils.py"
    if not miles.is_file():
        pytest.skip("Set MILES_SOURCE to a Miles checkout to render architecture profiles")
    spec = importlib.util.spec_from_file_location("model_args_utils", miles)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_args_utils"] = module
    spec.loader.exec_module(module)
    for profile in opd_config.PROFILES:
        directory = opd_runtime.PROFILES if (opd_runtime.PROFILES / f"{profile}.py").exists() else None
        rendered = module.load_model_args(profile, model_script_dir=directory or miles.parents[3] / "scripts/models")
        if profile.startswith("qwen3.5"):
            assert "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec" in rendered
        else:
            assert "--vocab-size 151936" in rendered
    two_b = module.load_model_args("qwen3.5-2B", model_script_dir=opd_runtime.PROFILES).split()
    assert two_b[two_b.index("--num-layers") + 1] == "24"
    assert "--untie-embeddings-and-output-weights" not in two_b
    assert specs.load(CONFIG, ['model.source="Qwen/Qwen3.5-2B"']).document["model"]["architecture"] == "qwen3.5-2B"


def _repository_profile(profile):
    path = opd_runtime.PROFILES / f"{profile}.py"
    spec = importlib.util.spec_from_file_location(f"profile_{profile.replace('.', '_').replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_args().split()


@pytest.mark.parametrize(
    "profile, layers, hidden, ffn, heads",
    [("qwen3-1.7B", "28", "2048", "6144", "16"), ("qwen3-4B", "36", "2560", "9728", "32")],
)
def test_qwen3_profiles_pin_the_unpadded_vocabulary(profile, layers, hidden, ffn, heads):
    # mbridge scatters the 151936-row HF embedding across TP ranks unpadded; Megatron would
    # otherwise pad to 152064 under TP2 and the conversion fails on a size mismatch.
    args = _repository_profile(profile)
    values = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("--")}
    assert values["--vocab-size"] == values["--padded-vocab-size"] == "151936"
    assert int(values["--padded-vocab-size"]) % 128 == 0
    assert (values["--num-layers"], values["--hidden-size"], values["--ffn-hidden-size"]) == (layers, hidden, ffn)
    assert values["--num-attention-heads"] == heads and values["--num-query-groups"] == "8"
    assert values["--rotary-base"] == "1000000" and values["--kv-channels"] == "128"
    assert "--untie-embeddings-and-output-weights" not in args  # Qwen3-Base 1.7B/4B tie embeddings
    assert "--qk-layernorm" in args and "--swiglu" in args


def test_code_overlay_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MILES_CODE_OVERLAY", raising=False)
    spec = specs.load(CONFIG)
    assert opd_launch.code_overlay() == ""
    assert "oi-overlay" not in launch.specification("test-image", spec)["tasks"][0]["arguments"][0]


def test_code_overlay_fetches_head_before_training(monkeypatch):
    monkeypatch.setenv("MILES_CODE_OVERLAY", "1")
    spec = specs.load(CONFIG)
    command = launch.specification("test-image", spec)["tasks"][0]["arguments"][0]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert "fetch -q --depth 1 https://github.com/" in command
    assert revision in command
    for directory in opd_launch.OVERLAY_DIRS:
        assert f"cp -r /tmp/oi-overlay/{directory}/. /opt/core-rl/{directory}/" in command
    assert command.index("cp -r /tmp/oi-overlay/open_instruct/.") < command.index(
        "python -m open_instruct.miles train"
    )


def test_teacher_eos_remap_reads_the_prepared_generation_config(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [151645, 151643]}))
    remap = opd_prepare.teacher_eos_remap(tmp_path)
    assert remap == {151643: 151645}
    assert opd_prepare.eos_remap_environment(remap) == "151643:151645"
    assert opd_prepare.eos_remap_environment({4: 9, 3: 9}) == "3:9,4:9"
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 151645}))
    with pytest.raises(InputError, match="own stop id"):
        opd_prepare.teacher_eos_remap(tmp_path)


def test_align_eos_with_teacher_default_and_validation():
    spec = specs.load(CONFIG)
    assert spec.document["model"]["align_eos_with_teacher"] is False
    with pytest.raises(InputError, match="align_eos_with_teacher"):
        specs.load(CONFIG, ['model.align_eos_with_teacher="yes"'])
    assert (
        specs.load(CONFIG, ["model.align_eos_with_teacher=true"]).document["model"]["align_eos_with_teacher"] is True
    )


def test_eos_token_ids_puts_teacher_eos_first_and_keeps_learner_eos():
    assert opd_prepare.eos_token_ids({"eos_token_id": 151643}, 151645) == [151645, 151643]
    assert opd_prepare.eos_token_ids({"eos_token_id": [151645, 151643]}, 151645) == [151645, 151643]
    assert opd_prepare.eos_token_ids({}, 7) == [7]


def test_eopd_selects_the_custom_loss_and_forwards_its_settings():
    values, flags = native(specs.load(CONFIG))
    assert "--loss-type" not in values and "--custom-loss-function-path" not in values
    assert not opd_runtime.eopd_settings(specs.load(CONFIG)).enabled
    spec = specs.load(
        CONFIG, ["distillation.use_rollout_logprobs=true", "distillation.eopd=true", "distillation.eopd_tau=0.7"]
    )
    values, flags = native(spec)
    assert values["--loss-type"] == "custom_loss"
    assert values["--custom-loss-function-path"] == "open_instruct.miles.eopd_loss.policy_loss"
    assert values["--opd-log-prob-top-k"] == "0"  # the upstream top-k reverse-KL path stays off
    assert "--use-opd" in flags and "--use-rollout-logprobs" in flags
    assert opd_runtime.eopd_settings(spec).environment() == {
        "OI_OPD_EOPD_TOP_K": "16",
        "OI_OPD_EOPD_ALPHA": "1.0",
        "OI_OPD_EOPD_TAU": "0.7",
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (["distillation.eopd_top_k=0"], "positive integer"),
        (["distillation.eopd_alpha=0"], "> 0"),
        (["distillation.eopd_tau=-1"], ">= 0"),
        (['distillation.eopd="yes"'], "distillation.eopd"),
    ],
)
def test_eopd_knobs_are_validated(override, message):
    with pytest.raises(InputError, match=message):
        specs.load(CONFIG, override)


def test_eopd_tiny_smoke_config_mirrors_the_opd_tiny_run():
    folder = CONFIG.parent
    opd = specs.load(folder / "eopd-opd-qwen3-tiny.toml").to_dict()
    eopd = specs.load(folder / "eopd-eopd-qwen3-tiny.toml").to_dict()
    assert eopd["distillation"] == opd["distillation"] | {"eopd": True}
    for section in ("model", "teacher", "training", "trainer", "inference", "optimizer", "data"):
        assert eopd[section] == opd[section], section
    assert eopd["output"]["root"] != opd["output"]["root"]
