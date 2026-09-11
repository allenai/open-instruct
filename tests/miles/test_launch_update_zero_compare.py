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


def test_limited_hf_mode_is_explicit_in_remote_command():
    task = launch.specification(
        launch.IMAGE, launch.ROOT / "update-zero-v2/core", launch.ROOT / "update-zero-v1/megatron", hf_only=True
    )["tasks"][0]
    assert "--hf-only" in task["arguments"][0]


def test_evidence_only_cli_keeps_failure_scope_explicit():
    task = launch.specification(
        launch.IMAGE, launch.ROOT / "update-zero-v2/core", launch.ROOT / "update-zero-v3/megatron", evidence_only=True
    )["tasks"][0]
    assert "--evidence-only" in task["arguments"][0]
    with pytest.raises(ValueError):
        launch.specification(
            launch.IMAGE,
            launch.ROOT / "update-zero-v2/core",
            launch.ROOT / "update-zero-v3/megatron",
            hf_only=True,
            evidence_only=True,
        )


def test_hf_twins_allow_explicit_same_backend_roots_only_in_hf_mode():
    left = launch.ROOT / "hfmatched-a/core"
    right = launch.ROOT / "hfmatched-b/core"
    task = launch.specification(launch.IMAGE, left, right, hf_only=True)["tasks"][0]
    assert str(left) in task["arguments"][0] and str(right) in task["arguments"][0]
    with pytest.raises(ValueError):
        launch.specification(launch.IMAGE, left, right)


def test_hf_reference_suite_uses_one_cpu_job_and_three_labeled_outputs():
    left, right, reference = (launch.ROOT / name / "core" for name in ("pina-a", "pina-b", "original-a"))
    result = launch.specification(launch.IMAGE, left, right, hf_only=True, reference_root=reference)
    assert len(result["tasks"]) == 1
    command = result["tasks"][0]["arguments"][0]
    assert command.count("python /tmp/compare_update_zero.py") == 3
    for name in ("pinned-twins", "reference-vs-left", "reference-vs-right"):
        assert f"--output /output/{name}.json" in command
    with pytest.raises(ValueError):
        launch.specification(launch.IMAGE, left, launch.ROOT / "mega/megatron", reference_root=reference)
    with pytest.raises(ValueError):
        launch.specification(launch.IMAGE, left, right, hf_only=True, reference_root="/tmp/unrelated/core")
