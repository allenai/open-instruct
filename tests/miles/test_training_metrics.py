"""MILES metric normalization and secondary tracking without external services."""

from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from miles.backends.training_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo
from torch import distributed as dist
from torch import multiprocessing as mp

from open_instruct.miles import metrics


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        dp = GroupInfo(rank=rank, size=2, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=dp,
                intra_dp_cp=dp,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        # Unequal rank sample counts AND response lengths distinguish the global
        # average from averaging rank means or averaging microbatch means.
        samples = [(1, 2.0, 0.0), (3, 4.0, 0.5)] if rank == 0 else [(5, 8.0, 1.0)]
        for token_average in (False, True):
            batches = []
            for length, pg, clipped in samples:
                count = length if token_average else 1
                payload = {
                    "keys": ["pg_loss", "pg_clipfrac", "train_rollout_logprob_abs_diff"],
                    "values": torch.tensor([count, count * pg, count * clipped, count * pg / 10]),
                }
                batches.append(metrics.loss_metrics(payload, torch.tensor(pg / 10)))
            actual = metrics.aggregate_losses(batches)
            denominator = 9 if token_average else 3
            assert actual["pg_loss"] == pytest.approx((54 if token_average else 14) / denominator)
            assert actual["pg_clipfrac"] == pytest.approx((6.5 if token_average else 1.5) / denominator)
            assert actual["train_rollout_logprob_abs_diff"] == pytest.approx(actual["pg_loss"] / 10)
            assert "normalized_policy_objective" not in actual
            assert "_miles_metric_count" not in actual
            summary = metrics.step_summary(batches, {"load_balancing": 0.1 * rank, "router_z": 0.0}, 2.0 + rank)
            assert summary == pytest.approx(
                {
                    "policy_objective": 0.7,
                    "auxiliary_load_balancing": 0.05,
                    "auxiliary_router_z": 0.0,
                    "step_seconds": 3.0,
                }
            )
        # Both ranks participate in aggregation, but only rank zero joins/logs.
        args = SimpleNamespace(wandb_run_id="driver-run", entropy_coef=0, observe_training_entropy=False)
        with mock.patch.object(metrics.tracking, "init_tracking") as init:
            metrics.init_tracking(args)
            if rank == 0:
                init.assert_called_once_with(args, primary=False)
            else:
                init.assert_not_called()
        with mock.patch.object(metrics.tracking, "log") as log:
            output = metrics.log_step(
                args,
                losses={"pg_loss": 2, "entropy_loss": 0},
                summary=summary,
                scores={},
                clock=SimpleNamespace(completed_steps=11, published_step=10),
                rollout_id=10,
                lr_used=[1e-6],
                lr_next=[9e-7],
                optimizer_metrics={"optim/total grad norm": 4.0},
                gradient_stats={"router": {"local_l2": 99}},
            )
            if rank == 0:
                log.assert_called_once_with(args, output, step_key="train/step")
                assert output["train/step"] == 10
                assert output["train/completed_steps"] == 11
                assert output["train/published_step"] == 10
                assert output["train/grad_norm"] == 4.0
                assert output["train/rank0_local_pre_optimizer/router/l2"] == 99
                assert output["train/lr_used-pg_0"] == 1e-6
                assert output["train/lr-pg_0"] == 9e-7
                assert "train/entropy_loss" not in output
            else:
                log.assert_not_called()
                assert output is None
    finally:
        dist.destroy_process_group()


def test_global_sample_and_token_weighting_and_rank_zero_tracking(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_profile_overflow_is_not_logged_as_zero_or_exact_quantile():
    output = metrics.score_metrics(0.3, 12, {"max_abs": 5.0, "p50_upper": 0.1, "p95_upper": None, "p99_upper": None})
    assert output["collection_score_mean_abs"] == 0.3
    assert output["collection_score_active_tokens"] == 12
    assert output["collection_score_p50_upper"] == 0.1
    assert output["collection_score_p95_upper_overflow"] == 1
    assert "collection_score_p95_upper" not in output
    assert all(isinstance(value, (int, float)) for value in output.values())


@pytest.mark.parametrize("observe_entropy", [False, True])
def test_auxiliary_zero_and_no_diagnostic_probe_still_log(observe_entropy):
    args = SimpleNamespace(entropy_coef=0, observe_training_entropy=observe_entropy)
    with (
        mock.patch.object(metrics.dist, "get_rank", return_value=0),
        mock.patch.object(metrics.tracking, "log") as log,
    ):
        output = metrics.log_step(
            args,
            losses={"pg_loss": 0.25, "entropy_loss": 1.5 if observe_entropy else 0},
            summary={"auxiliary_load_balancing": 0, "auxiliary_router_z": 0},
            scores={},
            clock=SimpleNamespace(completed_steps=1, published_step=0),
            rollout_id=0,
            lr_used=[1e-6],
            lr_next=[1e-6],
            optimizer_metrics={},
        )
    log.assert_called_once()
    assert output["train/step"] == 0
    assert output["train/pg_loss"] == 0.25
    assert output["train/auxiliary_load_balancing"] == 0
    assert ("train/entropy_loss" in output) is observe_entropy
    assert "train/grad_norm" not in output
    assert not any("local_pre_optimizer" in key for key in output)
