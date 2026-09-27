"""Forced interventions retain honest probabilities and learn both closing directions."""

import asyncio
import contextlib
import json
from types import SimpleNamespace

import pytest
import torch
from miles.backends.core_utils import data, packing, stopping
from miles.backends.training_utils import loss as miles_loss
from miles.backends.training_utils import parallel
from miles.ray.rollout import train_data_conversion
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.types import Sample
from olmo_core.train.train_module.transformer import objective as core_objective
from test_contract import loss_args
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from open_instruct.miles.rewards import rewards
from open_instruct.miles.rollout import forced_exits


@pytest.mark.parametrize("advantage,direction", [(1.0, -1), (-1.0, 1), (0.0, 0)])
def test_closing_gradient(advantage, direction):
    current = torch.tensor([-3.0, -2.0], requires_grad=True)
    loss = stopping.closing_objective(current, current.detach(), advantage, 0.2, 0.2)
    loss.backward()
    assert current.grad.sign().tolist() == [direction, direction]


def test_guidance_clips_large_improvement():
    current = torch.tensor([-1.0], requires_grad=True)
    loss = stopping.closing_objective(current, torch.tensor([-3.0]), 1.0, 0.2, 0.2)
    loss.backward()
    assert current.grad.item() == 0


def test_positions_use_paragraphs_and_reserve_answer_budget():
    tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: "".join({1: "work", 2: "\n\n", 3: "next", 8: "</", 9: "think>"}[i] for i in ids)
    )
    tokens = [1, 2, 3, 2, 1, 2, 8, 9]
    assert forced_exits.uniform_positions(tokens, [8, 9], 5, 100, tokenizer) == [2, 4, 6]
    assert forced_exits.uniform_positions(tokens, [8, 9], 5, 4, tokenizer) == [2, 4]
    assert forced_exits.uniform_positions([1, 3], [8, 9], 5, 100, tokenizer) == []
    assert forced_exits.uniform_positions(tokens, [8, 9], 2, 100, tokenizer) == [4, 6]


def test_cut_groups_do_not_change_natural_advantages():
    args = SimpleNamespace(
        reward_key=None, advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=False
    )
    group = [Sample(index=i, group_index=0, reward=1) for i in range(4)]
    for parent in group:
        probes = [{"rewards": [0, 0, 0]}, {"rewards": [1, 1, 1]}]
        forced_exits.cut_advantages(parent.reward, probes)
        parent.train_metadata = {"stopping_probes": probes}
        assert probes[0]["advantage"] < 0 < probes[1]["advantage"]
    assert not forced_exits.has_signal(group, args)
    assert train_data_conversion._post_process_rewards(args, group, None)[1] == [0] * 4
    for parent in group:
        parent.reward = 0
        parent.train_metadata["stopping_probes"] = [{"rewards": [0, 0, 0]}]
        forced_exits.cut_advantages(0, parent.train_metadata["stopping_probes"])
    assert not forced_exits.has_signal(group, args)


def test_producer_returns_only_natural_samples(monkeypatch):
    args = SimpleNamespace(
        reward_key=None,
        rollout_max_response_len=2048,
        rollout_global_dataset=True,
        rollout_seed=17,
        router_load_balancing_method="round_robin",
        olmo_core=SimpleNamespace(
            forced_exit_positions=2, forced_exit_trials=3, forced_exit_parents=1, forced_exit_answer_tokens=1024
        ),
    )

    def encode(text, **kwargs):
        assert text == "</think>\n\n"
        return [8, 9]

    tokenizer = SimpleNamespace(
        encode=encode,
        decode=lambda ids, **kw: "".join({1: "work", 2: "\n\n", 7: "42", 8: "</", 9: "think>\n\n"}[i] for i in ids),
    )
    seen = []

    async def generate(state, sample, params):
        seen.append(sample)
        if sample.response_length:
            assert params["max_new_tokens"] - sample.response_length == 1024
            assert sample.response.endswith("</think>\n\n")
            sample.tokens += [7]
            sample.response_length += 1
            sample.response += "42"
            sample.reward = 0
        else:
            sample.tokens = [10, 1, 2, 1, 2, 8, 9, 7]
            sample.response_length = 7
            sample.response = "work\n\nwork\n\n</think>42"
            sample.rollout_log_probs = [-1.0] * 7
            sample.reward = 1
        sample.status = Sample.Status.COMPLETED
        sample.weight_versions = [0]
        return sample

    monkeypatch.setattr(forced_exits.common, "generate_and_rm", generate)
    monkeypatch.setattr(forced_exits.generate_endpoint_utils, "policy_uses_routing_key", lambda args: True)
    producer = object.__new__(forced_exits.ForcedExitRollout)
    producer.state = SimpleNamespace(args=args, tokenizer=tokenizer, sampling_params={})
    group = asyncio.run(producer._group([Sample(index=i, group_index=0) for i in range(4)]))
    assert len(group) == 4 and len(seen) == 10
    assert all(s.reward == 1 and s.loss_mask is None for s in group)
    assert len({s.routing_key for s in seen}) == len(seen)
    assert len({s.index for s in seen}) == len(seen)
    assert sorted(len(s.train_metadata["stopping_probes"]) for s in group) == [0, 0, 0, 2]


def test_branch_provenance_and_mask():
    parent = Sample(
        tokens=[10, 11, 1, 2, 3],
        response_length=3,
        rollout_log_probs=[-1, -2, -3],
        status=Sample.Status.TRUNCATED,
        reward=0,
        index=4,
    )
    original = Sample(index=5, group_index=1)
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))
    sample = forced_exits.prepare_branch(parent, original, tokenizer, 2, [8, 9], 0, 0)
    assert sample.tokens == [10, 11, 1, 2, 8, 9]
    assert sample.rollout_log_probs == [-1, -2, 0, 0]
    assert sample.loss_mask == [0, 0, 0, 0]
    assert sample.index == 5 and sample.group_index == 1 and sample.weight_versions == []
    assert original.tokens == [] and parent.rollout_log_probs == [-1, -2, -3]


def test_truncation_zero_keeps_negative_advantage_and_constant_groups_have_none():
    args = SimpleNamespace(
        reward_key=None, advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=False
    )
    samples = [
        Sample(group_index=0, index=i, reward=r, status=Sample.Status.TRUNCATED if i == 0 else Sample.Status.COMPLETED)
        for i, r in enumerate([0, 1, 1, 0])
    ]
    raw, advantages = train_data_conversion._post_process_rewards(args, samples, None)
    assert raw == [0, 1, 1, 0]
    assert advantages == [-0.5, 0.5, 0.5, -0.5]
    assert not any(s.remove_sample for s in samples)
    for s in samples:
        s.reward = 0
    assert train_data_conversion._post_process_rewards(args, samples, None)[1] == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "status,text,expected",
    [
        (Sample.Status.TRUNCATED, "</think>42", 0),
        (Sample.Status.TRUNCATED, "42", 0),
        (Sample.Status.COMPLETED, "42", 0),
        (Sample.Status.COMPLETED, "</think>  ", 0),
        (Sample.Status.COMPLETED, "wrong 7 </think>42", 1),
    ],
)
def test_reward_gates(monkeypatch, status, text, expected):
    async def verify(tokens, response, target, **kwargs):
        assert response == "42"
        return SimpleNamespace(score=1.0, cost=0)

    monkeypatch.setattr(rewards, "_registry", lambda path: {"math": SimpleNamespace(async_call=verify)})
    args = SimpleNamespace(
        olmo_core=SimpleNamespace(reward_config="unused", reward_zero_truncated=True, reward_final_answer_only=True)
    )
    sample = Sample(status=status, response=text, metadata={"verifiers": [{"name": "math", "target": "42"}]})
    assert asyncio.run(rewards._score(args, sample)) == expected
    assert not sample.remove_sample


def test_readiness_uses_pre_intervention_state_and_packed_offsets():
    model = torch.nn.Module()
    model.lm_head = torch.nn.Identity()
    info = {"cut": 2, "rewards": [1, 0, 1]}
    batch = {
        "total_lengths": [5, 6],
        "response_lengths": [3, 4],
        "metadata": [{"stopping_probes": [info]}, {"stopping_probes": [info]}],
        "doc_lens": torch.tensor([[5, 6]]),
        "log_probs": [torch.full((3,), -2.0), torch.full((4,), -2.0)],
    }
    hidden = torch.arange(22).reshape(1, 11, 2).float()
    records = []
    with stopping.capture_readiness(model, batch, records):
        model.lm_head(hidden)
    assert torch.equal(records[0]["state"], hidden[0, 3])
    assert torch.equal(records[1]["state"], hidden[0, 8])
    assert not model.lm_head._forward_pre_hooks
    with (
        pytest.raises(ValueError, match="context_parallel_size"),
        stopping.capture_readiness(model, batch, [], context_parallel_size=2),
    ):
        model.lm_head(hidden)
    del batch["doc_lens"]
    with pytest.raises(ValueError, match="document boundaries"), stopping.capture_readiness(model, batch, []):
        model.lm_head(hidden)
    assert not model.lm_head._forward_pre_hooks


def test_failed_group_joins_siblings():
    async def run():
        stopped = asyncio.Event()

        async def sibling():
            try:
                await asyncio.sleep(60)
            finally:
                stopped.set()

        async def fail():
            await asyncio.sleep(0)
            raise ValueError("failed")

        with pytest.raises(ValueError, match="failed"):
            await forced_exits.gather_complete([sibling(), fail()])
        assert stopped.is_set()

    asyncio.run(run())


def test_readiness_capture_persists_finiteness_evidence(tmp_path):
    records = [{"state": torch.tensor([3.0, 4.0]), "rewards": [1, 0, 1]}]
    stopping.save_readiness(tmp_path, 2, 0, records)
    folder = tmp_path / "readiness-probes"
    summary = json.loads((folder / "step2-rank0.json").read_text())
    assert summary["all_finite"] and summary["records"] == 1 and summary["hidden_size"] == 2
    assert summary["norm_min"] == summary["norm_max"] == 5
    saved = torch.load(folder / "step2-rank0.pt", weights_only=True)
    assert saved["records"][0]["rewards"] == [1, 0, 1]
    records[0]["state"][0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        stopping.save_readiness(tmp_path, 3, 0, records)
    assert not (folder / "step3-rank0.pt").exists()


def test_real_loss_packing_tis_and_guidance(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/group", rank=0, world_size=1)
    previous = parallel._parallel_state
    try:
        group = GroupInfo(rank=0, size=1, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=group,
                intra_dp_cp=group,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )
        args = loss_args(False)
        args.use_rollout_logprobs = False
        args.use_tis = True
        args.tis_clip_low, args.tis_clip = 0.9, 1.1
        args.custom_tis_function_path = None
        args.loss_type = "custom_loss"
        args.custom_loss_function_path = "miles.backends.core_utils.stopping.policy_loss"
        args.olmo_core = SimpleNamespace(forced_exit_coefficient=0.1)
        args.context_parallel_size = 1
        lengths, responses = [5, 8, 6, 7], [3, 5, 2, 4]
        generator = torch.Generator().manual_seed(71)
        initial = [torch.randn(1, n, 11, generator=generator) * 0.3 for n in lengths]
        tokens = [torch.arange(n) % 11 for n in lengths]
        scores = [
            miles_loss.get_log_probs_and_entropy(
                x, args=args, unconcat_tokens=[t], total_lengths=[n], response_lengths=[r], max_seq_lens=[n]
            )["log_probs"][0]
            for x, t, n, r in zip(initial, tokens, lengths, responses, strict=True)
        ]
        raw = dict(
            tokens=tokens,
            total_lengths=lengths,
            response_lengths=responses,
            loss_masks=[torch.ones(n) for n in responses],
            log_probs=scores,
            rewards=[-0.5, 0.5, -0.5, 0.5],
            advantages=[torch.full((n,), a) for n, a in zip(responses, [-0.5, 0.5, -0.5, 0.5])],
            rollout_log_probs=[s + 0.3 for s in scores],
            metadata=[
                {"stopping_probes": [{"cut": 1, "close_ids": [10], "advantage": a}]} for a in [-0.5, 0.5, -0.5, 0.5]
            ],
        )
        results = []
        for packed in (False, True):
            args.qkv_format = "thd" if packed else "bshd"
            values = [v.clone().requires_grad_() for v in initial]
            samples = data.sample_batches(raw, 16)
            indices = packing.plan(lengths, 16) if packed else [[i] for i in range(4)]
            batches = [packing.combine(samples, ids) for ids in indices] if packed else samples
            objective = 0
            for batch, ids in zip(batches, indices, strict=True):
                logits = torch.cat([values[i] for i in ids], dim=1)
                loss, _, _ = miles_loss.loss_function(args, batch, len(batches), logits)
                objective = objective + loss
            objective.backward()
            results.append((objective.detach(), [v.grad for v in values]))
        torch.testing.assert_close(results[0], results[1], rtol=1e-6, atol=1e-7)
        for i, (n, r) in enumerate(zip(lengths, responses, strict=True)):
            # Parent sampled-token PPO and alternate close-token guidance share the same forward.
            assert results[0][1][i][0, n - r - 1].count_nonzero() > 0
            assert results[0][1][i][0, -1].count_nonzero() == 0
    finally:
        parallel.set_parallel_state(previous)
        dist.destroy_process_group()


@pytest.mark.parametrize("advantage", [-1.0, 1.0])
def test_full_tag_guidance_trains_all_pieces_only(advantage):
    args = SimpleNamespace(
        eps_clip=0.2, eps_clip_high=0.28, global_batch_size=4, olmo_core=SimpleNamespace(forced_exit_coefficient=0.1)
    )
    info = {"cut": 2, "close_ids": [8, 9, 10], "advantage": advantage}
    parent = {
        "unconcat_tokens": [torch.arange(6)],
        "total_lengths": [6],
        "response_lengths": [4],
        "metadata": [{"stopping_probes": [info]}],
    }
    contexts = stopping.auxiliary_batches([parent], pad_to=2)
    assert contexts[0]["tokens"].tolist() == [[0, 1, 2, 3, 8, 9, 10]]
    assert contexts[0]["doc_lens"].tolist() == [[7]]
    assert contexts[0]["max_doc_lens"] == [7]
    assert contexts[1]["stopping_context"]["dummy"]
    assert contexts[1]["doc_lens"].tolist() == [[2]]
    logits = torch.zeros(1, 7, 11, requires_grad=True)
    stopping.anchor_closing_scores(contexts[0], logits.detach())
    template = {"_miles_metric_count": torch.tensor(1.0), "normalized_policy_objective": torch.tensor(0.0)}
    loss, metrics = stopping.auxiliary_loss(args, contexts[0], logits, template, 1)
    loss.backward()
    for row, token in zip([3, 4, 5], [8, 9, 10], strict=True):
        assert logits.grad[0, row, token].sign() == -advantage
    assert logits.grad[0, :3].count_nonzero() == 0
    assert logits.grad[0, 6:].count_nonzero() == 0
    assert metrics["_miles_metric_count"] == 0
    dummy_logits = torch.zeros(1, 2, 11, requires_grad=True)
    stopping.anchor_closing_scores(contexts[1], dummy_logits.detach())
    dummy_loss, _ = stopping.auxiliary_loss(args, contexts[1], dummy_logits, template, 1)
    dummy_loss.backward()
    assert dummy_logits.grad.count_nonzero() == 0


def test_bpe_merged_natural_close_is_detected():
    tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: {1: "work\n\n", 2: ".</", 3: "think", 4: ">\n\n", 5: "42\n\n"}[ids[0]]
    )
    assert forced_exits.thinking_end([1, 2, 3, 4, 5], tokenizer) == 1
    assert forced_exits.uniform_positions([1, 2, 3, 4, 5], [20, 21, 22], 5, 100, tokenizer) == [1]


def test_zero_advantage_cuts_skip_compute_without_reweighting_or_losing_labels():
    probes = [{"cut": 2, "close_ids": [8, 9, 10], "advantage": a} for a in [0.5, 0.0, -0.2]]
    parent = {
        "unconcat_tokens": [torch.arange(6)],
        "total_lengths": [6],
        "response_lengths": [4],
        "metadata": [{"stopping_probes": probes}],
    }
    contexts = stopping.auxiliary_batches([parent])
    assert len(contexts) == 2
    assert all(b["stopping_context"]["count"] == 3 for b in contexts)
    assert len(parent["metadata"][0]["stopping_probes"]) == 3
    for p in probes:
        p["advantage"] = 0.0
    assert stopping.auxiliary_batches([parent]) == []
    assert all(b["stopping_context"]["dummy"] for b in stopping.auxiliary_batches([parent], pad_to=2))


def _distributed_full_tag_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(101)
        raw_model = torch.nn.Sequential(torch.nn.Embedding(11, 4), torch.nn.Linear(4, 11))
        model = DistributedDataParallel(raw_model)
        advantages = [0.5, 0.0, -0.2] if rank == 0 else [0.0]
        probes = [{"cut": 2, "close_ids": [8, 9, 10], "advantage": a} for a in advantages]
        parent = {
            "tokens": torch.arange(6).unsqueeze(0),
            "unconcat_tokens": [torch.arange(6)],
            "total_lengths": [6],
            "response_lengths": [4],
            "metadata": [{"stopping_probes": probes}],
        }
        batches = stopping.auxiliary_batches([parent], pad_to=2)
        for batch in batches:
            with torch.no_grad():
                stopping.anchor_closing_scores(batch, model(batch["tokens"]))
        args = SimpleNamespace(
            eps_clip=0.2,
            eps_clip_high=0.28,
            global_batch_size=2,
            olmo_core=SimpleNamespace(forced_exit_coefficient=0.1),
        )
        template = {"_miles_metric_count": torch.tensor(0.0), "normalized_policy_objective": torch.tensor(0.0)}
        module = SimpleNamespace(
            model=model,
            _train_microbatch_context=lambda i, n: model.no_sync() if i < n - 1 else contextlib.nullcontext(),
        )

        def objective(module, batch):
            return stopping.auxiliary_loss(args, batch, module.model(batch["tokens"]), template, 2)

        core_objective.train_batch_with_loss(module, batches, objective)
        torch.manual_seed(101)
        reference = torch.nn.Sequential(torch.nn.Embedding(11, 4), torch.nn.Linear(4, 11))
        reference_parent = dict(
            parent,
            metadata=[
                {"stopping_probes": [{"cut": 2, "close_ids": [8, 9, 10], "advantage": a} for a in [0.5, 0.0, -0.2]]}
            ],
        )
        expected = sum(
            -0.1
            / 2
            / 3
            * b["stopping_context"]["info"]["advantage"]
            * stopping.closing_scores(b, reference(b["tokens"])).sum()
            for b in stopping.auxiliary_batches([reference_parent])
        )
        expected.backward()
        for actual, target in zip(raw_model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, target.grad, rtol=1e-5, atol=1e-7)
    finally:
        dist.destroy_process_group()


def test_distributed_auxiliary_padding_and_normalization(tmp_path):
    mp.spawn(_distributed_full_tag_worker, args=(f"file://{tmp_path}/aux-group",), nprocs=2, join=True)
