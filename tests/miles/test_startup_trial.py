"""Startup qualification must still execute the complete training contract."""

import json

import pytest
from scripts.miles import startup_trial


def fixture_progress(root):
    stages = [{"stage": "initial_publication", "rollout_id": None}]
    stages += [
        {"stage": name, "rollout_id": update}
        for update in range(2)
        for name in ("generation_wait", "training", "publication")
    ]
    rows = [
        dict(
            event="optimizer",
            step=update + 1,
            optimizer_skipped=False,
            local_behavior_versions=[update],
            normalization={"samples": 4},
        )
        for update in range(2)
    ]
    for rank in range(2):
        (root / f"training_contract_rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (root / "publication.jsonl").write_text(
        "".join(json.dumps(dict(version=v, repeated_version=False)) + "\n" for v in range(3))
    )
    return stages


def test_complete_progress(tmp_path):
    startup_trial.validate_progress(tmp_path, fixture_progress(tmp_path))


@pytest.mark.parametrize("name", ("initial_publication", "generation_wait", "training", "publication"))
def test_missing_work_rejects_fast_run(tmp_path, name):
    stages = [row for row in fixture_progress(tmp_path) if row["stage"] != name]
    with pytest.raises(ValueError):
        startup_trial.validate_progress(tmp_path, stages)


def test_skipped_optimizer_rejected(tmp_path):
    stages = fixture_progress(tmp_path)
    path = tmp_path / "training_contract_rank1.jsonl"
    path.write_text(path.read_text().replace('"optimizer_skipped": false', '"optimizer_skipped": true'))
    with pytest.raises(ValueError, match="skipped"):
        startup_trial.validate_progress(tmp_path, stages)


def test_missing_publication_rejected(tmp_path):
    stages = fixture_progress(tmp_path)
    (tmp_path / "publication.jsonl").write_text("")
    with pytest.raises(ValueError, match="publication sequence"):
        startup_trial.validate_progress(tmp_path, stages)
