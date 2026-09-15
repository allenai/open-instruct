"""CPU checks for OPD configuration and allocation boundaries."""

import asyncio
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles import launch, opd_config, opd_runtime, rewards, specs
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
        'model.source="Qwen/Qwen3.5-1.7B"',  # no pinned revision and no architecture profile
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
        assert "--spec miles_plugins.models.qwen3_5 get_qwen3_5_spec" in rendered
    two_b = module.load_model_args("qwen3.5-2B", model_script_dir=opd_runtime.PROFILES).split()
    assert two_b[two_b.index("--num-layers") + 1] == "24"
    assert "--untie-embeddings-and-output-weights" not in two_b
    assert specs.load(CONFIG, ['model.source="Qwen/Qwen3.5-2B"']).document["model"]["architecture"] == "qwen3.5-2B"
