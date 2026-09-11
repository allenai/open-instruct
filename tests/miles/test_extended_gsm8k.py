"""Check extension isolation, cluster placement, and bounded training protocol."""

import pytest
from scripts.miles import extended_gsm8k, launch_extended_gsm8k


def test_extension_cannot_reuse_original_output(tmp_path):
    with pytest.raises(ValueError, match="distinct campaign"):
        extended_gsm8k.prepare(tmp_path, tmp_path)


@pytest.mark.parametrize("stage", ["prepare", "core", "audit"])
def test_extension_stage_placement_and_horizon(stage):
    document = launch_extended_gsm8k.specification("test-image", stage)
    task = document["tasks"][0]
    command = task["arguments"][0]
    assert str(extended_gsm8k.ROOT) in command
    assert "--updates 500 --eval-interval 20" in command
    assert task["context"]["priority"] == "urgent"
    assert task["constraints"]["cluster"] == ["ai2/holmes" if stage == "core" else "ai2/saturn"]
    if stage == "core":
        assert task["resources"]["gpuCount"] == 3
        assert task["timeout"] == "18h"
        assert task["context"]["minRuntime"] == "12h"
        assert "--save-interval 100" in command
    else:
        assert task["resources"]["gpuCount"] == 0
