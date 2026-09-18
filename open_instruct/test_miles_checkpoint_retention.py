"""Rolling retention of committed native Core checkpoints."""

from open_instruct.miles import checkpoint, config


def test_retention_does_not_change_native_writer_options():
    defaults = config.CoreConfig().checkpoint_save_options()
    retained = config.CoreConfig(checkpoint_keep_last=1, checkpoint_keep_every=20)
    assert retained.checkpoint_save_options() == defaults
    threaded = config.CoreConfig(checkpoint_thread_count=2, checkpoint_keep_last=1)
    assert threaded.checkpoint_save_options() == {**defaults, "thread_count": 2}


def _committed(root, rollout_id, *, complete=True):
    path = root / "core" / f"rollout_{rollout_id:07d}"
    path.mkdir(parents=True)
    (path / "model").write_bytes(b"weights")
    (path / ("complete.json" if complete else "pending.json")).write_text("{}")
    (root / "rollout").mkdir(exist_ok=True)
    (root / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt").write_bytes(b"cursor")
    return path


def test_retention_set_keeps_newest_milestones_and_latest():
    ids = [4, 9, 14, 19, 24]
    assert checkpoint.retained_rollout_ids(ids, 24, keep_last=None, keep_every=None) == set(ids)
    assert checkpoint.retained_rollout_ids(ids, 24, keep_last=1, keep_every=None) == {24}
    assert checkpoint.retained_rollout_ids(ids, 24, keep_last=2, keep_every=10) == {9, 19, 24}
    # The latest commit is kept even when it is not yet listed as committed.
    assert checkpoint.retained_rollout_ids([4, 9], 14, keep_last=1, keep_every=None) == {9, 14}


def test_prune_deletes_only_committed_checkpoints_outside_the_retention_set(tmp_path):
    for rollout_id in (4, 9, 14, 19):
        _committed(tmp_path, rollout_id)
    _committed(tmp_path, 24, complete=False)
    (tmp_path / "core" / "rollout_0000012.incomplete-abc").mkdir()

    assert checkpoint.prune(tmp_path, 19, keep_last=None, keep_every=None) == []
    # keep_last=2 keeps 14 and 19; keep_every=10 keeps 9 (the tenth update) and 19.
    assert checkpoint.prune(tmp_path, 19, keep_last=2, keep_every=10) == [4]
    assert sorted(p.name for p in (tmp_path / "core").iterdir()) == [
        "rollout_0000009",
        "rollout_0000012.incomplete-abc",
        "rollout_0000014",
        "rollout_0000019",
        "rollout_0000024",
    ]
    assert sorted(p.name for p in (tmp_path / "rollout").iterdir()) == [
        "global_dataset_state_dict_14.pt",
        "global_dataset_state_dict_19.pt",
        "global_dataset_state_dict_24.pt",
        "global_dataset_state_dict_9.pt",
    ]
    assert checkpoint.prune(tmp_path, 19, keep_last=1, keep_every=None) == [9, 14]
    assert (tmp_path / "core" / "rollout_0000019" / "complete.json").exists()
    assert checkpoint.committed_rollout_ids(tmp_path) == [19]
