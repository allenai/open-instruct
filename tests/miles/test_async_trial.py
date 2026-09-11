"""Admission evidence must distinguish bounded lag from corrupted collections."""

import copy
from pathlib import Path

import pytest
from scripts.miles import async_trial


def samples(version=1):
    return [
        {"metadata": {"prepared_sample_id": str(group)}, "group_index": group, "weight_versions": [str(version)]}
        for group in range(4)
        for _ in range(4)
    ]


def test_async_accepts_complete_reordered_groups_with_one_step_lag():
    rows = list(reversed(samples()))
    ids, version, groups = async_trial.batch_membership(
        rows, {str(i): {} for i in range(4)}, rollout=2, asynchronous=True, consumed=set()
    )
    assert set(ids) == {"0", "1", "2", "3"}
    assert version == {str(i): 1 for i in range(4)} and groups == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "fault", ["partial", "merged", "split", "duplicate", "foreign", "mixed", "future", "stale", "missing"]
)
def test_rejects_corrupt_async_membership(fault):
    rows = copy.deepcopy(samples())
    consumed = set()
    if fault == "partial":
        rows.pop()
    elif fault == "merged":
        rows[4]["group_index"] = 0
    elif fault == "split":
        rows[0]["group_index"] = 8
    elif fault == "duplicate":
        consumed.add("0")
    elif fault == "foreign":
        rows[0]["metadata"]["prepared_sample_id"] = "heldout"
    elif fault == "mixed":
        rows[0]["weight_versions"] = ["0"]
    elif fault in ("future", "stale"):
        for row in rows:
            row["weight_versions"] = ["3" if fault == "future" else "0"]
    elif fault == "missing":
        rows[0]["weight_versions"] = []
    with pytest.raises(ValueError):
        async_trial.batch_membership(
            rows, {str(i): {} for i in range(4)}, rollout=2, asynchronous=True, consumed=consumed
        )


def test_synchronous_rejects_even_one_step_lag():
    with pytest.raises(ValueError, match="lag"):
        async_trial.batch_membership(
            samples(), {str(i): {} for i in range(4)}, rollout=2, asynchronous=False, consumed=set()
        )


def test_scheduling_configs_preserve_shared_recipe_and_actual_budget():
    sync = async_trial.configuration(Path("/prepared"), Path("/sync"), asynchronous=False)
    asynchronous = async_trial.configuration(Path("/prepared"), Path("/async"), asynchronous=True)
    assert sync.core.max_policy_lag == 0 and asynchronous.core.max_policy_lag == 1
    for config in (sync, asynchronous):
        assert "--no-use-wandb" not in config.arguments()
        assert "--no-fully-async" not in config.arguments()
        assert config.miles["num_rollout"] == config.miles["lr_decay_iters"] == 4
        assert config.miles["rollout_max_response_len"] == 4096
        assert config.miles["global_batch_size"] == 16
        assert config.miles["use_rollout_logprobs"]
        assert config.miles["disable_grpo_std_normalization"]
        assert config.miles["sglang_cuda_graph_backend_decode"] == "disabled"
        assert not any(key.startswith("eval_") for key in config.miles)
    assert asynchronous.miles["async_unused_samples_handler"] == "retry"


def test_async_accepts_different_homogeneous_groups_with_bounded_lag():
    rows = samples()
    for row in rows[8:]:
        row["weight_versions"] = ["2"]
    _, versions, _ = async_trial.batch_membership(
        rows, {str(i): {} for i in range(4)}, rollout=2, asynchronous=True, consumed=set()
    )
    assert versions == {"0": 1, "1": 1, "2": 2, "3": 2}


def test_async_rejects_one_response_spanning_versions():
    rows = samples()
    rows[0]["weight_versions"] = ["1", "2"]
    with pytest.raises(ValueError, match="response mixes"):
        async_trial.batch_membership(
            rows, {str(i): {} for i in range(4)}, rollout=2, asynchronous=True, consumed=set()
        )


def test_rank_versions_preserve_actual_strided_membership():
    rows = [{"metadata": {"prepared_sample_id": key}} for key in ["older", "current", "older", "current"]]
    assert async_trial.rank_versions(rows, {"older": 1, "current": 2}) == {"0": [1], "1": [2]}
