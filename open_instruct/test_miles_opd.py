"""CPU checks for OPD configuration and allocation boundaries."""

import copy
from pathlib import Path

import pytest

from open_instruct.miles import launch, opd_config, opd_runtime, specs
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
        "training.resume=true",
        "launch.auto_resume=true",
        "distillation.log_prob_top_k=10",
        "distillation.task_reward_weight=0.5",
        'model.source="Qwen/Qwen3.5-2B"',  # no pinned revision and no architecture profile
        'model.architecture="qwen3-4B"',
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


def test_opd_rejects_data_without_gsm8k_evaluation():
    document = specs.load(CONFIG).to_dict()
    for tasks in ([{"task": "math", "train_count": 16, "eval_count": 8}], [{"task": "gsm8k", "train_count": 16}]):
        document["data"]["tasks"] = tasks
        with pytest.raises(InputError, match="GSM8K"):
            specs.from_dict(document)


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
            "inference.eval_samples_per_prompt=4",
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
    assert values["--n-samples-per-eval-prompt"] == "4"
    assert values["--sglang-max-running-requests"] == "64"
    assert values["--sglang-max-total-tokens"] == str(2048 * 64)
    assert values["--actor-num-gpus-per-node"] == "4"
    assert values["--num-gpus-per-node"] == "6"
    assert values["--rollout-num-gpus"] == "2"
    assert values["--rollout-num-gpus-per-engine"] == "2"
    assert "--use-rollout-logprobs" in flags
    assert "--use-wandb" not in flags
    assert "--wandb-mode" not in values


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
