"""The validate command must reject layouts that the launcher cannot allocate."""

import json
import sys
from pathlib import Path

from open_instruct.miles import __main__ as cli
from open_instruct.miles.configuration.run_spec import RunSpec


def test_validate_multinode_does_not_assume_a_storage_provider(tmp_path, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    payload = RunSpec.load(root / "configs/miles/examples/medium.toml").to_dict()
    payload["output"]["root"] = "/shared/training/run"
    payload["launch"]["weka_mounts"] = []
    config = tmp_path / "run.json"
    config.write_text(json.dumps(payload))
    # Planning validates GPU allocation; Beaker submission checks shared mounts.
    RunSpec.load(config).compile().arguments()
    monkeypatch.setattr(sys, "argv", ["miles", "validate", str(config)])
    cli.main()
    assert "Run schema, topology and MILES/Core options validated" in capsys.readouterr().out
