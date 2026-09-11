"""Retained scheduling audits are CPU-only Saturn jobs with fresh evidence."""

import pytest
from scripts.miles import launch_async_trial as launch

ROOT = "/weka/oe-training-default/robertb/open-instruct/miles-scheduling/01M274MRNFSFF34QRYRGAASH52/async"


def test_cpu_reaudit_placement_and_retained_input():
    task = launch.audit_specification("image", ROOT)["tasks"][0]
    assert task["resources"] == {"cpuCount": 8, "memory": "32 GiB"}
    assert task["constraints"]["cluster"] == ["ai2/saturn"]
    assert task["context"]["minRuntime"] == "20m" and task["timeout"] == "30m"
    assert ROOT in task["arguments"][0]
    assert "--audit-only /output/audit.json" in task["arguments"][0]


@pytest.mark.parametrize("path", ["/tmp/run", ROOT + "/extra", ROOT + "/../../other", ROOT[:-5] + "sync"])
def test_reaudit_rejects_ambiguous_or_mismatched_retained_path(path):
    with pytest.raises(ValueError):
        launch.audit_specification("image", path)
