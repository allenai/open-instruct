"""One W&B run per output root: create, reuse, override, atomic write, mismatch warning."""

import json
import logging
import re
from types import SimpleNamespace

import pytest

from open_instruct.miles.execution import wandb_identity

FIELDS = {"model": "/weka/model", "data": {"recipe": "r"}, "wandb_project": "p", "wandb_group": "g"}


def _spec(model="/weka/model"):
    return SimpleNamespace(model={"source": model}, data={"recipe": "r"})


def test_first_start_creates_legal_id_and_fingerprint(tmp_path):
    run_id = wandb_identity.resolve(tmp_path, FIELDS)
    # W&B run ids: no "/", "\\", "#", "?", "%" or ":"; uuid4 hex is 32 lowercase hex chars.
    assert re.fullmatch(r"[0-9a-f]{32}", run_id)
    assert (tmp_path / "wandb_run_id").read_text().strip() == run_id
    stored = json.loads((tmp_path / "wandb_run_id.json").read_text())
    assert stored["id"] == run_id
    assert stored["config_hash"] == wandb_identity.config_hash(FIELDS)
    assert stored["created_at"] > 0


def test_restart_reuses_stored_id(tmp_path, caplog):
    first = wandb_identity.resolve(tmp_path, FIELDS)
    with caplog.at_level(logging.WARNING):
        assert wandb_identity.resolve(tmp_path, dict(FIELDS)) == first
    assert "MISMATCH" not in caplog.text


def test_changed_config_warns_and_keeps_id(tmp_path, caplog):
    first = wandb_identity.resolve(tmp_path, FIELDS)
    with caplog.at_level(logging.WARNING):
        assert wandb_identity.resolve(tmp_path, {**FIELDS, "model": "/weka/other"}) == first
    assert "W&B RUN CONFIG MISMATCH" in caplog.text and "new output.root" in caplog.text


def test_user_id_wins_and_writes_nothing(tmp_path):
    miles = {"use_wandb": True, "wandb_run_id": "mine"}
    assert wandb_identity.apply(tmp_path, _spec(), miles) == "mine"
    assert miles["wandb_run_id"] == "mine"
    assert not (tmp_path / "wandb_run_id").exists()


def test_disabled_wandb_sets_nothing(tmp_path):
    miles = {"use_wandb": False}
    assert wandb_identity.apply(tmp_path, _spec(), miles) is None
    assert "wandb_run_id" not in miles and not (tmp_path / "wandb_run_id").exists()


def test_apply_sets_same_id_across_attempts(tmp_path):
    first = {"use_wandb": True, "wandb_project": "p"}
    second = {"use_wandb": True, "wandb_project": "p"}
    wandb_identity.apply(tmp_path, _spec(), first)
    wandb_identity.apply(tmp_path, _spec(), second)
    assert first["wandb_run_id"] == second["wandb_run_id"]


def test_write_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "wandb_run_id"
    path.write_text("old\n")

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(wandb_identity.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        wandb_identity._write_atomic(path, "new\n")
    # The previous content survives and no temp file is left behind.
    assert path.read_text() == "old\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["wandb_run_id"]


def test_failed_first_write_leaves_no_partial_id(tmp_path, monkeypatch):
    calls = []
    original = wandb_identity._write_atomic

    def write(path, text):
        calls.append(path.name)
        if path.name == wandb_identity.ID_FILE:
            raise OSError("interrupted")
        original(path, text)

    monkeypatch.setattr(wandb_identity, "_write_atomic", write)
    with pytest.raises(OSError):
        wandb_identity.resolve(tmp_path, FIELDS)
    # The id file is the commit point; without it the next start creates a fresh id.
    assert not (tmp_path / "wandb_run_id").exists()
    monkeypatch.setattr(wandb_identity, "_write_atomic", original)
    run_id = wandb_identity.resolve(tmp_path, FIELDS)
    assert json.loads((tmp_path / "wandb_run_id.json").read_text())["id"] == run_id


def test_empty_id_file_is_replaced(tmp_path):
    (tmp_path / "wandb_run_id").write_text("")
    assert re.fullmatch(r"[0-9a-f]{32}", wandb_identity.resolve(tmp_path, FIELDS))
