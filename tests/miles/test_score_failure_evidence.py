"""A rejected behavior policy must leave exact token evidence before raising."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from miles.backends.core_utils import actor, contract


def test_failure_records_scores_and_masked_token_alignment(tmp_path):
    rollout = {
        "tokens": [torch.tensor([11, 12, 13, 20, 21, 22])],
        "log_probs": [torch.tensor([-0.5, -9.0, -0.3])],
        "rollout_log_probs": [[-0.1, -1.0, -0.3]],
        "loss_masks": [torch.tensor([1, 0, 1])],
        "rewards": [0.0],
    }
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(
        save=str(tmp_path),
        olmo_core=SimpleNamespace(max_train_rollout_logprob_abs_diff=0.05, publication_mode="barrier"),
    )
    with (
        mock.patch.object(worker, "_agree", side_effect=lambda f: f()),
        mock.patch.object(contract.dist, "all_reduce"),
        mock.patch.object(contract.dist, "get_rank", return_value=0),
        pytest.raises(ValueError, match="logprob difference"),
    ):
        worker._score_contract(rollout, 7, "standalone")
    records = [json.loads(line) for line in (tmp_path / "training_contract_rank0.jsonl").read_text().splitlines()]
    assert [r["event"] for r in records] == ["scores", "score_mismatch"]
    assert records[0]["active_tokens"] == 2
    assert records[0]["mean_abs"] == pytest.approx(0.2)
    row = records[1]["sequences"][0]
    assert row["prompt_token_ids"] == [11, 12, 13]
    assert row["response_token_ids"] == [20, 21, 22]
    assert row["loss_mask"] == [True, False, True]
    assert row["trainer_minus_rollout"] == pytest.approx([-0.4, -8.0, 0.0])
    assert row["active_max_abs"] == pytest.approx(0.4)
