"""The researcher CLI must remain usable without site-packages or GPU dependencies."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", ["plan", "validate"])
@pytest.mark.parametrize("tier", ["dev", "small", "medium", "large"])
def test_cli_without_runtime_dependencies(command, tier):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "open_instruct.miles", command, f"configs/miles/examples/{tier}.toml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    if command == "plan":
        assert json.loads(result.stdout)
    else:
        assert "validated" in result.stdout
