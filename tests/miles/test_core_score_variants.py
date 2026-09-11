"""Fail-closed source/input/numerical checks for the isolated score experiment."""

import importlib.util
import json
from pathlib import Path

import pytest

LAUNCHER_PATH = Path(__file__).parents[2] / "scripts/miles/launch_core_score_variants.py"
LAUNCHER_SPEC = importlib.util.spec_from_file_location("score_variant_launcher", LAUNCHER_PATH)
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(launcher)


def test_launch_isolates_sources_caches_and_processes():
    task = launcher.specification(launcher.IMAGE)["tasks"][0]
    command = task["arguments"][0]
    assert "cp -a /opt/core-rl/sources/olmo-core /tmp/score-parent/core" in command
    assert "cp -a /opt/core-rl/sources/olmo-core /tmp/score-candidate/core" in command
    assert "for arm in parent candidate" in command
    assert "--fuzz=0" in command
    assert task["resources"]["gpuCount"] == 2
    assert task["timeout"] == "90m"
    assert task["context"]["minRuntime"] == "30m"
    assert task["constraints"]["cluster"] == ["ai2/holmes"]


@pytest.mark.parametrize("image,output", [("latest", launcher.OUTPUT), (launcher.IMAGE, "/tmp/overwrite")])
def test_invalid_image_or_output_rejected(image, output):
    with pytest.raises(ValueError):
        launcher.specification(image, output=output)


def fixture_reports(root):
    torch = pytest.importorskip("torch")
    manifest = {"parent_sha256": "parent", "candidate_sha256": "candidate"}
    for arm in ("parent", "candidate"):
        (root / arm).mkdir()
        for rank in (0, 1):
            document = {
                "valid": True,
                "recipe_argv": ["same-recipe"],
                "sources": {"kernel_sha256": arm, "other_core_python_sha256": "same", "oi_modules": {"actor": "same"}},
                "inputs": [{"sha256": "identical", "partition": [rank]}],
            }
            (root / arm / f"rank{rank}.json").write_text(json.dumps(document))
            for rollout in range(5, 10):
                torch.save([torch.tensor([-1.0, -2.0])], root / arm / f"scores-{rollout}-{rank}.pt")
    path = Path(__file__).parents[2] / "scripts/miles/profile_core_score_variants.py"
    spec = importlib.util.spec_from_file_location("score_variants_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return torch, module, manifest


def test_exact_scores_pass_and_changed_scores_retain_residual(tmp_path):
    torch, worker, manifest = fixture_reports(tmp_path)
    assert worker.compare_runs(tmp_path, manifest)["valid"]
    torch.save([torch.tensor([-1.0, -2.1])], tmp_path / "candidate/scores-7-1.pt")
    report = worker.compare_runs(tmp_path, manifest)
    assert not report["valid"]
    assert len(report["comparisons"]) == 10
    assert sum(not row["valid"] for row in report["comparisons"]) == 1
    assert report["comparisons"][7]["max_abs"] > 0.09


@pytest.mark.parametrize("field", ["other_core_python_sha256", "oi_modules", "kernel_sha256", "inputs", "valid"])
def test_non_kernel_or_input_differences_rejected(tmp_path, field):
    _, worker, manifest = fixture_reports(tmp_path)
    path = tmp_path / "candidate/rank0.json"
    document = json.loads(path.read_text())
    if field == "inputs":
        document[field] = [{"sha256": "changed"}]
    elif field == "valid":
        document[field] = False
    else:
        document["sources"][field] = "changed"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        worker.compare_runs(tmp_path, manifest)
