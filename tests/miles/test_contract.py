"""Independent numerical references, real MILES losses and real Gloo reductions.

Run in the pinned image. CPU distributed tests qualify the reduction algebra,
not Core's EP kernels; the GPU runtime tests separately exercise native Core.
"""

import contextlib
import json
from datetime import timedelta
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from miles.backends.training_utils import loss as miles_loss
from miles.backends.training_utils import parallel
from miles.utils.ft_utils.process_group_utils import GroupInfo
from olmo_core.nn.moe import loss as auxiliary
from olmo_core.nn.moe.v2 import replay
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.train.train_module.transformer.objective import train_batch_with_loss
from torch import distributed as dist
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint as recompute

from open_instruct.miles import contract
from open_instruct.miles.state import PolicyClock


def loss_args(token_average):
    return SimpleNamespace(
        calculate_per_token_loss=token_average,
        global_batch_size=4,
        qkv_format="bshd",
        loss_type="policy_loss",
        advantage_estimator="grpo",
        use_rollout_logprobs=True,
        skip_actor_forward_only=False,
        entropy_coef=0.0,
        observe_training_entropy=False,
        use_opsm=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        use_tis=False,
        get_mismatch_metrics=False,
        use_kl_loss=False,
        custom_pg_loss_reducer_function_path=None,
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        allgather_cp=False,
        log_probs_chunk_size=32,
        recompute_loss_function=False,
        use_dynamic_global_batch_size=False,
        multi_lora=False,
    )


def fixture_data():
    generator = torch.Generator().manual_seed(173)
    logits = torch.randn(4, 8, 11, generator=generator) * 0.3
    samples = []
    for i, (total, response) in enumerate(((5, 3), (8, 5), (6, 2), (7, 4))):
        tokens = (torch.arange(total) + i) % 11
        mask = torch.ones(response)
        if response > 2:
            mask[1] = 0  # tool/interior token, deliberately not a trailing pad
        old = (
            logits[i, total - response - 1 : total - 1]
            .log_softmax(-1)
            .gather(-1, tokens[-response:, None])
            .squeeze(-1)
        )
        # Exercise both clipped branches and unclipped terms with both advantage signs.
        old = old - torch.linspace(-0.4, 0.4, response)
        samples.append(
            dict(tokens=tokens, total=total, response=response, mask=mask, old=old, advantage=(-1.0 if i % 2 else 1.0))
        )
    return logits, samples


def batch_for(samples, indices):
    selected = [samples[i] for i in indices]
    return dict(
        indices=torch.tensor(indices),
        tokens=torch.stack([F.pad(s["tokens"], (0, 8 - len(s["tokens"]))) for s in selected]),
        unconcat_tokens=[s["tokens"] for s in selected],
        total_lengths=[s["total"] for s in selected],
        response_lengths=[s["response"] for s in selected],
        max_seq_lens=[8] * len(selected),
        loss_masks=[s["mask"] for s in selected],
        ref_log_probs=[s["old"] + 0.17 for s in selected],
        log_probs=[s["old"] for s in selected],
        rollout_log_probs=[s["old"] for s in selected],
        advantages=[torch.full((s["response"],), s["advantage"]) for s in selected],
    )


def reference_loss(logits, samples, token_average, regularized=False):
    """Plain PyTorch expression independent of MILES slicing, reducers and PPO helpers."""
    terms = []
    for i, sample in enumerate(samples):
        start = sample["total"] - sample["response"]
        values = []
        for j in range(sample["response"]):
            if sample["mask"][j] == 0:
                continue
            logp = torch.log_softmax(logits[i, start + j - 1], 0)[sample["tokens"][start + j]]
            ratio = torch.exp(logp - sample["old"][j])
            a = sample["advantage"]
            term = -torch.minimum(ratio * a, ratio.clamp(0.8, 1.2) * a)
            if regularized:
                distribution = logits[i, start + j - 1].log_softmax(0)
                entropy = -(distribution.exp() * distribution).sum()
                log_ratio = sample["old"][j] + 0.17 - logp
                term = term - 0.02 * entropy + 0.03 * (log_ratio.exp() - log_ratio - 1)
            values.append(term)
        terms.append(torch.stack(values).sum() if token_average else torch.stack(values).mean())
    return torch.stack(terms).sum() / (sum(int(s["mask"].sum()) for s in samples) if token_average else len(samples))


class LogitTable(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.logits = nn.Parameter(initial.clone())

    def forward(self, indices):
        return self.logits[indices]


class AccumulationModule:
    def __init__(self, model):
        self.model = model

    def _train_microbatch_context(self, index, count):
        return (
            self.model.no_sync()
            if isinstance(self.model, DistributedDataParallel) and index < count - 1
            else contextlib.nullcontext()
        )


def _worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world, timeout=timedelta(seconds=90)
    )
    try:
        group = GroupInfo(rank=rank, size=world, group=dist.group.WORLD, gloo_group=dist.group.WORLD)
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
        initial, samples = fixture_data()
        measurements = []
        for token_average, regularized in product((False, True), repeat=2):
            for chunk_size in (1, 2, 4):
                table = LogitTable(initial)
                wrapped = DistributedDataParallel(table) if world > 1 else table
                module = AccumulationModule(wrapped)
                indices = list(range(rank, 4, world))
                batches = [batch_for(samples, indices[i : i + chunk_size]) for i in range(0, len(indices), chunk_size)]
                norm = contract.step_normalization(batches, 4)
                args = loss_args(token_average)
                if regularized:
                    args.entropy_coef = 0.02
                    args.use_kl_loss = True
                    args.use_unbiased_kl = False
                    args.kl_loss_type = "k3"
                    args.kl_loss_coef = 0.03

                count = len(batches)

                def objective(module, batch, args=args, count=count, norm=norm):
                    value, _, _ = miles_loss.loss_function(
                        args, batch, count, module.model(batch["indices"]), apply_megatron_loss_scaling=False
                    )
                    if args.calculate_per_token_loss:
                        value = norm.scale_token_loss(value)
                    return value, {"loss": value}

                metrics = train_batch_with_loss(module, batches, objective)
                actual_loss = torch.stack([m["loss"] for m in metrics]).sum()
                dist.all_reduce(actual_loss)
                actual_loss /= world
                reference = initial.clone().requires_grad_()
                expected_loss = reference_loss(reference, samples, token_average, regularized)
                expected_loss.backward()
                torch.testing.assert_close(actual_loss, expected_loss.detach(), atol=2e-7, rtol=2e-6)
                torch.testing.assert_close(table.logits.grad, reference.grad, atol=2e-7, rtol=2e-6)
                # Prompt, masked, final-unscored and padded positions must have zero gradient.
                assert torch.count_nonzero(reference.grad == 0) > reference.numel() // 2
                actual_optim = torch.optim.AdamW(table.parameters(), lr=0.003, weight_decay=0.01)
                reference_optim = torch.optim.AdamW([reference], lr=0.003, weight_decay=0.01)
                actual_optim.step()
                reference_optim.step()
                torch.testing.assert_close(table.logits, reference, atol=2e-7, rtol=2e-6)
                measurements.append(
                    dict(
                        world_size=world,
                        microbatch_size=chunk_size,
                        token_average=token_average,
                        entropy_and_kl=regularized,
                        loss=float(actual_loss),
                        gradient_max_abs_error=float((table.logits.grad - reference.grad).abs().max()),
                        update_max_abs_error=float((table.logits - reference).detach().abs().max()),
                    )
                )
        # One skipped rank must leave every rank's policy clock untouched.
        clock = PolicyClock(completed_steps=3, published_step=3)
        with pytest.raises(ValueError, match="Optimizer step skipped"):
            contract.validate_step_transition(clock, SimpleNamespace(step_skipped=rank == 0), torch.device("cpu"))
        assert clock.completed_steps == 3 and clock.published_step == 3
        contract.validate_step_transition(clock, SimpleNamespace(step_skipped=False), torch.device("cpu"))
        if world > 1:
            clock.completed_steps += rank
            with pytest.raises(ValueError, match="disagree on policy clock"):
                contract.validate_step_transition(clock, SimpleNamespace(step_skipped=False), torch.device("cpu"))
            with pytest.raises(ValueError, match="same number"):
                contract.validate_batch_schedule(2 + rank * 2, 2, torch.device("cpu"))
        # A large last-token outlier must remain visible despite a small mean.
        profile_input = dict(
            log_probs=[torch.tensor([0.0, 0.01, 99.0, 0.8])],
            rollout_log_probs=[torch.zeros(4)],
            loss_masks=[torch.tensor([1, 1, 0, 1])],
        )
        profile = contract.probability_profile(profile_input)
        assert profile["active_tokens"] == 3 * world
        assert profile["max_abs"] == pytest.approx(0.8)
        assert profile["p95_upper"] == 1.0
        assert profile["groups"]["last_third"]["mean_abs"] == pytest.approx(0.8)
        with pytest.raises(ValueError, match="Invalid optimizer batch counts"):
            contract.step_normalization([dict(tokens=torch.ones(1, 2), loss_masks=[torch.zeros(1)])], world)
        if rank == 0:
            Path(output).write_text(json.dumps(measurements, indent=2))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2])
def test_same_global_batch_same_gradient_and_update(world, tmp_path):
    mp.spawn(
        _worker, args=(world, str(tmp_path / "rdzv"), str(tmp_path / "measurement.json")), nprocs=world, join=True
    )
    results = json.loads((tmp_path / "measurement.json").read_text())
    assert len(results) == 12
    print("CONTRACT_MEASUREMENT", json.dumps(results))


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("microbatch", [1, 2, 4])
def test_auxiliary_gradients_match_independent_reference(world, microbatch):
    torch.manual_seed(17)
    initial = torch.randn(4, 5, 4) * 0.4
    expected_logits = initial.clone().requires_grad_()
    ids = initial.topk(2, dim=-1).indices
    # Deliberately imbalanced, frozen routes: replay does not eliminate balancing gradients.
    ids[..., 0] = 0
    ids[..., 1] = 2
    counts = F.one_hot(ids, 4).sum(dim=(1, 2)).float()
    p = expected_logits.softmax(-1)
    reference_lb = 2 * (p.mean(1) * counts).sum() / 20
    reference_z = torch.logsumexp(expected_logits, -1).square().mean()
    expected = 0.01 * reference_lb + 1e-5 * reference_z
    expected.backward()
    actual = initial.clone().requires_grad_()
    losses = []
    for rank in range(world):
        indices = list(range(rank, 4, world))
        for start in range(0, len(indices), microbatch):
            idx = indices[start : start + microbatch]
            logits = actual[idx]
            lb = auxiliary.load_balancing_loss(
                num_experts=4,
                top_k=2,
                expert_scores=logits.softmax(-1),
                batch_size_per_expert=counts[idx].sum(0),
                batched_batch_size_per_expert=counts[idx],
                granularity=auxiliary.MoELoadBalancingLossGranularity.instance,
                loss_div_factor=20 / world,
            )
            z = auxiliary.router_z_loss(expert_logits=logits, loss_div_factor=20 / world)
            losses.append((0.01 * lb + 1e-5 * z) / world)
    total = torch.stack(losses).sum()
    total.backward()
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(actual.grad, expected_logits.grad)
    assert actual.grad.abs().sum() > 0
    print(
        "AUXILIARY_CONTRACT",
        json.dumps(
            dict(
                world_size=world,
                microbatch_size=microbatch,
                loss=float(total.detach()),
                gradient_max_abs_error=float((actual.grad - expected_logits.grad).abs().max()),
            )
        ),
    )


@pytest.mark.parametrize("checkpointed", [False, True])
def test_replay_policy_auxiliary_and_combined_gradients(checkpointed):
    torch.manual_seed(42)
    router = MoERouterConfigV2(d_model=8, num_experts=4, top_k=2, lb_loss_weight=0.01, z_loss_weight=1e-5).build()
    nn.init.normal_(router.weight, std=0.1)
    wrapper = nn.Module()
    wrapper.add_module("routed_experts_router", router)
    routes = torch.tensor([[[0, 2], [0, 2], [1, 3]]])
    source = torch.randn(1, 3, 8)
    gradients = []
    for mode in ("policy", "auxiliary", "combined"):
        router.zero_grad()
        x = source.clone().requires_grad_()
        observed = []

        def forward(x, observed=observed):
            weights, actual, _, info = router(x, scores_only=False)
            observed.append(actual.detach().clone())
            aux = router.compute_aux_loss(*info, accumulate_metrics=False)
            policy = (weights * torch.tensor([1.0, -0.7])).sum()
            return policy, aux

        with replay.replay_routes(wrapper, {"routed_experts_router": routes}):
            policy, aux = recompute(forward, x, use_reentrant=False) if checkpointed else forward(x)
            value = policy if mode == "policy" else aux if mode == "auxiliary" else policy + aux
            value.backward()
        assert all(torch.equal(actual, routes) for actual in observed)
        assert not checkpointed or len(observed) >= 2
        assert router.weight.grad is not None and router.weight.grad.isfinite().all()
        assert router.weight.grad.abs().sum() > 0
        gradients.append(router.weight.grad.clone())
    torch.testing.assert_close(gradients[2], gradients[0] + gradients[1], atol=2e-6, rtol=2e-5)
    print(
        "REPLAY_CONTRACT",
        json.dumps(
            dict(
                checkpointed=checkpointed,
                policy_gradient_l2=float(gradients[0].norm()),
                auxiliary_gradient_l2=float(gradients[1].norm()),
                combined_gradient_l2=float(gradients[2].norm()),
                superposition_max_abs=float((gradients[2] - gradients[0] - gradients[1]).abs().max()),
            )
        ),
    )


def test_active_nonfinite_rejected_masked_nonfinite_ignored():
    rollout = dict(rewards=[1.0], loss_masks=[torch.tensor([1, 0])], advantages=[torch.tensor([1.0, float("nan")])])
    contract.validate_training_data(rollout)
    rollout["advantages"][0][0] = float("nan")
    with pytest.raises(ValueError, match="active advantages"):
        contract.validate_training_data(rollout)


def test_schedule_rejected_before_optimizer():
    clock = PolicyClock(completed_steps=3)
    schedule = SimpleNamespace(last_epoch=3, get_last_lr=lambda: [0.001])
    contract.validate_schedule(clock, schedule)
    schedule.last_epoch = 4
    with pytest.raises(ValueError, match="clock disagree"):
        contract.validate_schedule(clock, schedule)
    schedule.last_epoch = 3
    schedule.get_last_lr = lambda: [float("nan")]
    with pytest.raises(ValueError, match="learning rate"):
        contract.validate_schedule(clock, schedule)
