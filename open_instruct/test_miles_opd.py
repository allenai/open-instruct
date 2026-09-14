"""CPU checks for OPD configuration and allocation boundaries."""

import copy
from pathlib import Path

import pytest

from open_instruct.miles import launch, opd_config, specs
from open_instruct.miles.errors import InputError

CONFIG = Path(__file__).resolve().parents[1] / "configs/miles/opd/qwen35-4b-tiny.toml"


def test_opd_dispatch_and_gpu_isolation():
    spec = specs.load(CONFIG)
    assert isinstance(spec, opd_config.OPDRunSpec)
    assert spec.allocation()["roles"] == {"trainer": [0, 1], "student": [2], "teacher": [3]}
    task = launch.specification("test-image", spec)["tasks"][0]
    assert task["resources"]["gpuCount"] == 4
    assert task["context"]["autoResume"] is False
    assert task["constraints"] == {"cluster": ["ai2/holmes"]}
    assert specs.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


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
        "teacher.gpus=2",
        "training.resume=true",
        "training.save_interval=2",
        "launch.auto_resume=true",
        "distillation.log_prob_top_k=10",
        'model.source="Qwen/Qwen3.5-2B"',
    ],
)
def test_unsupported_paths_fail_before_launch(override):
    with pytest.raises(InputError):
        specs.load(CONFIG, [override])


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
