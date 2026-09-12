"""Actor-level scoring-pass behavior; needs the pinned MILES runtime the actor imports."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

pytest.importorskip("miles")

from open_instruct.miles import actor  # noqa: E402
from open_instruct.miles.config import CoreConfig  # noqa: E402
from open_instruct.miles.state import PolicyClock  # noqa: E402


def _worker(required=False, hf_config=None, **args):
    worker = actor.OLMoCoreTrainRayActor.__new__(actor.OLMoCoreTrainRayActor)
    worker.args = SimpleNamespace(
        global_batch_size=4,
        rollout_batch_size=1,
        n_samples_per_prompt=4,
        kl_coef=0.0,
        use_rollout_logprobs=False,
        olmo_core=CoreConfig(scoring_pass_required=required),
        **args,
    )
    if hf_config is not None:
        worker.hf_config = hf_config
    worker.clock = PolicyClock()
    return worker


def test_actor_resolves_the_decision_once_and_applies_the_model_check():
    hf_config = SimpleNamespace(to_dict=lambda: {"attention_dropout": 0.1})
    worker = _worker(hf_config=hf_config)
    decision = worker._scoring_pass()
    assert decision.standalone and decision.reason == "stochastic model configuration: attention_dropout"
    assert worker._scoring_pass() is decision
    assert not worker._scoring_check_due(decision)
    plain = _worker()
    assert not plain._scoring_pass().standalone
    assert plain._scoring_check_due(plain._scoring_pass())


def test_skip_attribute_is_scoped_to_the_update_even_on_failure():
    worker = _worker(skip_actor_forward_only=False)
    with worker._actor_forward_skipped(True):
        assert worker.args.skip_actor_forward_only is True
    assert worker.args.skip_actor_forward_only is False
    with pytest.raises(RuntimeError), worker._actor_forward_skipped(True):
        raise RuntimeError("update failed")
    assert worker.args.skip_actor_forward_only is False


def _distributed_check(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    actor.distributed_utils.init_gloo_group()
    try:
        worker = _worker()
        rollout = {"log_probs": [torch.zeros(2)], "loss_masks": [torch.ones(2)]}
        for fault in ("shape", "nonfinite"):
            scores = torch.zeros(3 if rank == 1 and fault == "shape" else 2)
            if rank == 1 and fault == "nonfinite":
                scores[0] = float("nan")
            with pytest.raises(RuntimeError, match="rank 1"):
                worker._check_training_scores(rollout, [scores])
        report = worker._check_training_scores(rollout, [torch.zeros(2)])
        assert report["active_tokens"] == 4 and report["mean_abs"] == 0
        Path(output, str(rank)).write_text("both rank-local failures coordinated; valid check passed")
    finally:
        dist.destroy_process_group()


def test_rank_local_scoring_check_failure_reaches_all_ranks(tmp_path):
    mp.spawn(_distributed_check, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True)
    assert all((tmp_path / str(rank)).is_file() for rank in (0, 1))
