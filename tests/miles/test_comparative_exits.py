"""Runtime unit tests: intervention scope, masking and conditional PPO gradients."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from miles.backends.core_utils import stopping, stopping_branches
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.types import Sample

from open_instruct.miles.rollout import comparative_exits


def parent_batch(advantage=0.0):
    branch = dict(
        tokens=[1, 2, 3, 4, 5, 6],
        loss_mask=[0, 0, 1, 1],
        log_probs=[0, 0, -2, -2],
        advantage=0.5,
        group_denominator=4,
        routed_experts=None,
    )
    info = dict(
        cut=1,
        close_ids=[8, 9, 10],
        guidance="first_token",
        advantage=advantage,
        parent_weight=4,
        training_branches=[branch],
    )
    return dict(
        unconcat_tokens=[torch.arange(6)],
        total_lengths=[6],
        response_lengths=[4],
        metadata=[{"stopping_probes": [info]}],
    )


def test_zero_stopping_advantage_still_trains_sampled_answers():
    contexts = stopping.auxiliary_batches([parent_batch()])
    assert len(contexts) == 1
    batch = contexts[0]
    logits = torch.zeros(1, 6, 11, requires_grad=True)
    stopping.anchor_closing_scores(batch, logits.detach())
    args = SimpleNamespace(
        eps_clip=0.2,
        eps_clip_high=0.28,
        use_tis=False,
        global_batch_size=4,
        olmo_core=SimpleNamespace(forced_exit_branch_coefficient=1.0),
    )
    loss, _ = stopping.auxiliary_loss(args, batch, logits, {}, 1)
    loss.backward()
    # Only rows predicting sampled answer tokens receive gradient; copied/forced rows do not.
    assert logits.grad[0, :3].count_nonzero() == 0
    assert logits.grad[0, 3, 5] < 0
    assert logits.grad[0, 4, 6] < 0
    assert logits.grad[0, 5].count_nonzero() == 0
    assert loss.item() == pytest.approx(-0.125)


def test_first_token_guidance_preserves_tag_spelling():
    contexts = stopping.auxiliary_batches([parent_batch(advantage=-0.5)])
    batch = contexts[1]
    logits = torch.zeros(1, 6, 11, requires_grad=True)
    scores = stopping.closing_scores(batch, logits)
    assert scores.numel() == 1
    loss = stopping.closing_objective(scores, scores.detach(), -0.5, 0.2, 0.28)
    loss.backward()
    assert logits.grad[0, 2, 8] > 0
    assert logits.grad[0, 3:].count_nonzero() == 0


def test_tis_uses_only_unmasked_behavior_probabilities():
    batch = stopping.auxiliary_batches([parent_batch()])[0]
    logits = torch.zeros(1, 6, 11)
    stopping.anchor_closing_scores(batch, logits)
    args = SimpleNamespace(
        eps_clip=0.2,
        eps_clip_high=0.28,
        use_tis=True,
        tis_clip_low=0,
        tis_clip=2,
        global_batch_size=4,
        olmo_core=SimpleNamespace(forced_exit_branch_coefficient=1.0),
    )
    branch = batch["stopping_context"]["branch"]
    branch["log_probs"][:2] = [float("nan")] * 2
    loss = stopping_branches.objective(args, batch, stopping.closing_scores(batch, logits), 1)
    assert torch.isfinite(loss)
    branch["log_probs"][-1] = float("nan")
    with pytest.raises(ValueError, match="behavior"):
        stopping_branches.objective(args, batch, stopping.closing_scores(batch, logits), 1)


def test_continue_suppresses_only_first_token_and_restores_budget():
    seen = []
    sample = Sample(
        tokens=[10, 1, 2],
        response="prefix",
        response_length=2,
        metadata={"comparative_continue_ids": [8, 9]},
        status=Sample.Status.PENDING,
    )

    async def generate(input):
        seen.append(dict(input.sampling_params))
        if len(seen) == 1:
            input.sample.tokens.append(3)
            input.sample.response_length += 1
            input.sample.status = Sample.Status.TRUNCATED
        else:
            assert input.sample.status == Sample.Status.PENDING
            assert input.sample.loss_mask == [0, 0, 0]
        return GenerateFnOutput(samples=input.sample)

    state = SimpleNamespace(generate_function=generate)
    asyncio.run(
        comparative_exits.generate_continuation(
            GenerateFnInput(state=state, sample=sample, sampling_params={"max_new_tokens": 100}, evaluation=False)
        )
    )
    assert seen[0]["max_new_tokens"] == 3
    assert seen[0]["logit_bias"] == {"8": -1e9, "9": -1e9}
    assert seen[1] == {"max_new_tokens": 100}


def test_comparison_attaches_separate_centered_branch_groups():
    worker = object.__new__(comparative_exits.ComparativeExitRollout)
    core = SimpleNamespace(
        forced_exit_initial_updates=4,
        forced_exit_initial_trials=2,
        forced_exit_trials=2,
        forced_exit_tie_bonus=0.01,
        forced_exit_tie_min_accuracy=0.5,
    )
    worker.state = SimpleNamespace(
        args=SimpleNamespace(olmo_core=core, reward_key=None, n_samples_per_prompt=4),
        tokenizer=SimpleNamespace(decode=lambda ids, **kw: "work"),
    )
    worker.close_ids = [8, 9, 10]
    parent = Sample(
        tokens=[10, 1, 2, 3], response_length=3, reward=1, status=Sample.Status.COMPLETED, weight_versions=[0]
    )

    async def branch(parent, pristine, cut, action, trial, step, position):
        return Sample(
            tokens=[10, 1, 8, 9, 10, 4],
            response_length=5,
            loss_mask=[0, 0, 0, 0, 1],
            rollout_log_probs=[0, 0, 0, 0, -1],
            reward=int(action == "continue" or trial == 0),
            weight_versions=[0],
            status=Sample.Status.COMPLETED,
            metadata={"probe_seconds": 0.1},
        )

    worker._branch = branch
    probe = asyncio.run(worker._score(parent, Sample(), 1, 0, 0))
    assert probe["advantage"] == -0.5
    assert probe["tie_bonus"] == 0
    assert [b["advantage"] for b in probe["training_branches"]] == [0.5, -0.5, 0, 0]
    assert all(b["group_denominator"] == 4 for b in probe["training_branches"])
    assert probe["parent_weight"] == 4


def test_continue_rejects_failed_suppression():
    sample = Sample(
        tokens=[10, 1],
        response_length=1,
        response="prefix",
        metadata={"comparative_continue_ids": [8]},
        status=Sample.Status.PENDING,
    )

    async def generate(input):
        input.sample.tokens.append(8)
        input.sample.response_length += 1
        return GenerateFnOutput(samples=input.sample)

    with pytest.raises(RuntimeError, match="suppression"):
        asyncio.run(
            comparative_exits.generate_continuation(
                GenerateFnInput(
                    state=SimpleNamespace(generate_function=generate),
                    sample=sample,
                    sampling_params={"max_new_tokens": 10},
                    evaluation=False,
                )
            )
        )
