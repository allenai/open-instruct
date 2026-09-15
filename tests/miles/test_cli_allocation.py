"""The validate command must reject layouts that the launcher cannot allocate."""

import json
import sys
from pathlib import Path

import pytest

from open_instruct.miles import __main__ as cli
from open_instruct.miles.run_spec import RunSpec


def test_validate_rejects_nonshared_multinode_output(tmp_path, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[2]
    payload = RunSpec.load(root / "configs/miles/examples/medium.toml").to_dict()
    payload["output"]["root"] = str(tmp_path / "not-shared")
    config = tmp_path / "run.json"
    config.write_text(json.dumps(payload))
    # Trainer settings alone accept this; the physical allocation does not.
    RunSpec.load(config).compile().arguments()
    monkeypatch.setattr(sys, "argv", ["miles", "validate", str(config)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "Multi-node rendezvous requires output.root on shared WEKA" in capsys.readouterr().err
