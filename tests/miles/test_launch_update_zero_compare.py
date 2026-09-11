"""Retained WEKA trace analysis always stays on CPU Saturn."""

import pytest
from scripts.miles import launch_update_zero_compare as launch


def test_cpu_saturn_and_separate_retry_roots():
    core = launch.ROOT / "update-zero-20260911-v2/core"
    mega = launch.ROOT / "update-zero-20260911-v1/megatron"
    document = launch.specification(launch.IMAGE, core, mega)
    task = document["tasks"][0]
    assert task["constraints"]["cluster"] == ["ai2/saturn"]
    assert task["resources"]["cpuCount"] == 8 and task["resources"].get("gpuCount", 0) == 0
    assert task["context"]["minRuntime"] == "20m" and task["timeout"] == "30m"
    assert str(core) in task["arguments"][0] and str(mega) in task["arguments"][0]
    assert "--output /output/comparison.json" in task["arguments"][0]


def test_rejects_changed_image_or_unrelated_paths():
    core = launch.ROOT / "update-zero-v2/core"
    mega = launch.ROOT / "update-zero-v1/megatron"
    with pytest.raises(ValueError):
        launch.specification("other-image", core, mega)
    with pytest.raises(ValueError):
        launch.specification(launch.IMAGE, "/tmp/unrelated/core", mega)
    with pytest.raises(ValueError):
        launch.specification(launch.IMAGE, core, launch.ROOT / "../outside/megatron")
