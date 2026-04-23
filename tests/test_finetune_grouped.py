"""CPU-only unit tests for the grouped-SFT-loss feature in finetune.py.

Covers:
- DistributedGroupedBatchSampler: sharding, variable group sizes, sort-balance, trim.
- compute_grouped_step: numerical parity with a reference implementation, and the
  headline regression test — invariance of loss + gradients across chunk size `M`.
"""

from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState

from open_instruct.finetune import DistributedGroupedBatchSampler, compute_grouped_step


@pytest.fixture(scope="session", autouse=True)
def _init_accelerate_state():
    """Accelerate's logger refuses to log until its process state is initialized.
    DistributedGroupedBatchSampler uses that logger at construction, so tests that
    instantiate the sampler need the state set up exactly once per session."""
    PartialState()
    yield


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class TinyCausalLM(nn.Module):
    """Minimal causal-LM shim: embedding + linear head, returns an object with .logits."""

    def __init__(self, vocab_size: int = 16, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, **_):
        h = self.embed(input_ids)
        logits = self.head(h)
        return SimpleNamespace(logits=logits)


class MockAccelerator:
    """Enough of the Accelerator API for compute_grouped_step on CPU, single-process."""

    def __init__(self, sync_gradients: bool = True):
        self.device = torch.device("cpu")
        self.sync_gradients = sync_gradients
        self.num_processes = 1  # skips the all-reduce in _global_max_num_chunks
        self.state = SimpleNamespace(deepspeed_plugin=None)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    @contextmanager
    def no_sync(self, model):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prompt_ids(group_sizes: list[int]) -> list[int]:
    ids: list[int] = []
    for pid, sz in enumerate(group_sizes):
        ids.extend([pid] * sz)
    return ids


def _build_batch(group_sizes: list[int], seq_len: int = 5, vocab_size: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    total = sum(group_sizes)
    input_ids = torch.randint(0, vocab_size, (total, seq_len))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    prompt_ids = torch.tensor(_make_prompt_ids(group_sizes))
    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    return batch, prompt_ids


def _grads_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def _reference_hierarchical_loss(model: nn.Module, batch: dict, prompt_ids: torch.Tensor) -> torch.Tensor:
    """Full-batch single-backward reference for the hierarchical mean loss."""
    _, inverse, counts = prompt_ids.unique(return_inverse=True, return_counts=True)
    num_prompts = counts.shape[0]
    weights = 1.0 / (num_prompts * counts[inverse].float())

    model_inputs = {k: v for k, v in batch.items() if k != "labels"}
    outputs = model(**model_inputs, use_cache=False)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    per_token = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none", ignore_index=-100
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    per_seq = (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return (per_seq * weights).sum()


# ---------------------------------------------------------------------------
# Sampler tests
# ---------------------------------------------------------------------------


def test_distributed_grouped_sampler_sharding():
    # 12 prompts × 3 rows = 36 rows. batch_size=2 per rank, world_size=2 → global_batch=4 prompts.
    # 12 / 4 = 3 global batches. Each rank yields 3 batches, 2 prompts × 3 rows = 6 rows each.
    group_sizes = [3] * 12
    prompt_ids = _make_prompt_ids(group_sizes)
    samplers = [
        DistributedGroupedBatchSampler(prompt_ids=prompt_ids, batch_size=2, shuffle=False, rank=r, world_size=2)
        for r in range(2)
    ]
    batches_per_rank = [list(iter(s)) for s in samplers]

    # Each rank yields 3 batches of 6 rows each
    assert all(len(b) == 3 for b in batches_per_rank)
    for batches in batches_per_rank:
        assert all(len(batch) == 6 for batch in batches)

    # Disjoint prompts across ranks within the same global step
    for r0_batch, r1_batch in zip(batches_per_rank[0], batches_per_rank[1], strict=True):
        r0_prompts = {prompt_ids[i] for i in r0_batch}
        r1_prompts = {prompt_ids[i] for i in r1_batch}
        assert r0_prompts.isdisjoint(r1_prompts)
        assert len(r0_prompts) == 2 and len(r1_prompts) == 2


def test_distributed_grouped_sampler_variable_sizes():
    # 4 prompts with sizes {4, 7, 2, 8} = 21 rows. world_size=1, batch_size=4 → one global batch.
    group_sizes = [4, 7, 2, 8]
    prompt_ids = _make_prompt_ids(group_sizes)
    sampler = DistributedGroupedBatchSampler(prompt_ids=prompt_ids, batch_size=4, shuffle=False, rank=0, world_size=1)
    batches = list(iter(sampler))
    assert len(batches) == 1
    assert len(batches[0]) == 21
    # Each prompt appears as a contiguous run (order may differ from input because of sort)
    actual_counts = Counter(prompt_ids[i] for i in batches[0])
    assert actual_counts == Counter({0: 4, 1: 7, 2: 2, 3: 8})


def test_distributed_grouped_sampler_lpt_balance():
    """LPT-with-capacity gives exact per-batch row balance for this symmetric
    sorted input — much tighter than round-robin (which would give 28 vs 24)."""
    # 8 prompts, sizes [10, 9, 8, 7, 6, 5, 4, 3] = 52 rows. world_size=2, batch_size=4.
    group_sizes = [10, 9, 8, 7, 6, 5, 4, 3]
    prompt_ids = _make_prompt_ids(group_sizes)
    samplers = [
        DistributedGroupedBatchSampler(prompt_ids=prompt_ids, batch_size=4, shuffle=False, rank=r, world_size=2)
        for r in range(2)
    ]
    batches = [next(iter(s)) for s in samplers]
    row_counts = [len(b) for b in batches]
    assert row_counts == [26, 26]


def test_distributed_grouped_sampler_lpt_ws3():
    """LPT on 6 prompts sized [10..5] with ws=3: every rank gets total 15."""
    group_sizes = [10, 9, 8, 7, 6, 5]
    prompt_ids = _make_prompt_ids(group_sizes)
    samplers = [
        DistributedGroupedBatchSampler(prompt_ids=prompt_ids, batch_size=2, shuffle=False, rank=r, world_size=3)
        for r in range(3)
    ]
    batches = [next(iter(s)) for s in samplers]
    row_counts = [len(b) for b in batches]
    assert sorted(row_counts) == [15, 15, 15]


def test_distributed_grouped_sampler_lpt_skewed():
    """LPT shines on skewed size distributions — much tighter balance than
    alternatives like simple snake or plain round-robin."""
    group_sizes = [100, 50, 40, 30, 20, 10, 5, 1]  # sum 256, avg 128
    prompt_ids = _make_prompt_ids(group_sizes)
    samplers = [
        DistributedGroupedBatchSampler(prompt_ids=prompt_ids, batch_size=4, shuffle=False, rank=r, world_size=2)
        for r in range(2)
    ]
    batches = [next(iter(s)) for s in samplers]
    row_counts = sorted([len(b) for b in batches])
    # LPT traversal: 100→r0, 50→r1, 40→r1, 30→r1, 20→r0, 10→r0, 5→r1 (r1 now full), 1→r0.
    # rank 0 = 100+20+10+1 = 131; rank 1 = 50+40+30+5 = 125. Diff 6.
    # (Snake would give 151/105 diff=46; plain round-robin 161/95 diff=66.)
    assert row_counts == [125, 131]


def test_distributed_grouped_sampler_equal_prompt_counts():
    """LPT-with-capacity always gives exactly `batch_size` prompts to every rank —
    the local-weights loss math requires this invariant."""
    group_sizes = [8, 6, 7, 4, 9, 3, 5, 8, 2, 6]  # 10 prompts, arbitrary order
    prompt_ids = _make_prompt_ids(group_sizes)
    for world_size, batch_size in [(2, 5), (5, 2), (2, 2), (1, 10)]:
        if len(group_sizes) < world_size * batch_size:
            continue
        samplers = [
            DistributedGroupedBatchSampler(
                prompt_ids=prompt_ids, batch_size=batch_size, shuffle=False, rank=r, world_size=world_size
            )
            for r in range(world_size)
        ]
        first_batches = [next(iter(s)) for s in samplers]
        for r, batch_indices in enumerate(first_batches):
            unique_prompts = {prompt_ids[i] for i in batch_indices}
            assert len(unique_prompts) == batch_size, (
                f"world_size={world_size}, batch_size={batch_size}, rank={r}: "
                f"got {len(unique_prompts)} prompts, expected {batch_size}"
            )


def test_distributed_grouped_sampler_lpt_disjoint_across_ranks():
    """All ranks together should cover the global batch exactly (no duplicates,
    no drops within the usable portion)."""
    group_sizes = [7, 3, 9, 5, 4, 8, 2, 6]  # 8 prompts, arbitrary order
    prompt_ids = _make_prompt_ids(group_sizes)
    world_size = 4
    batch_size = 2
    samplers = [
        DistributedGroupedBatchSampler(
            prompt_ids=prompt_ids, batch_size=batch_size, shuffle=False, rank=r, world_size=world_size
        )
        for r in range(world_size)
    ]
    first_batches = [next(iter(s)) for s in samplers]
    all_prompts: list[int] = []
    for indices in first_batches:
        all_prompts.extend(prompt_ids[i] for i in indices)
    # 8 prompts × their group sizes rows — same total as input
    assert Counter(all_prompts) == Counter(prompt_ids)


def test_distributed_grouped_sampler_trim():
    # 7 prompts with 2 rows each. batch_size=2 per rank, world_size=2 → global_batch=4.
    # 7 // 4 = 1 usable global batch; 3 prompts dropped for rank divisibility.
    group_sizes = [2] * 7
    prompt_ids = _make_prompt_ids(group_sizes)
    sampler = DistributedGroupedBatchSampler(prompt_ids=prompt_ids, batch_size=2, shuffle=False, rank=0, world_size=2)
    assert sampler._n_dropped == 3
    assert len(sampler) == 1


def test_distributed_grouped_sampler_len_matches_iter():
    group_sizes = [3] * 8
    prompt_ids = _make_prompt_ids(group_sizes)
    sampler = DistributedGroupedBatchSampler(prompt_ids=prompt_ids, batch_size=2, shuffle=True, rank=1, world_size=2)
    assert len(sampler) == len(list(iter(sampler)))


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


def test_grouped_loss_numerics_matches_reference():
    """compute_grouped_step (no chunking) must match a from-scratch full-batch reference."""
    torch.manual_seed(42)
    init_state = TinyCausalLM().state_dict()
    batch, prompt_ids = _build_batch([2, 4, 3], seed=0)
    total_rows = batch["input_ids"].shape[0]

    # Reference: full batch, single backward
    ref_model = TinyCausalLM()
    ref_model.load_state_dict(init_state)
    ref_loss = _reference_hierarchical_loss(ref_model, batch, prompt_ids)
    ref_loss.backward()
    ref_grads = _grads_snapshot(ref_model)

    # Ours
    model = TinyCausalLM()
    model.load_state_dict(init_state)
    accelerator = MockAccelerator()
    loss_detached = compute_grouped_step(model, batch, prompt_ids, accelerator, max_rows_per_forward=total_rows)
    new_grads = _grads_snapshot(model)

    assert torch.allclose(loss_detached, ref_loss.detach().float(), atol=1e-5)
    for name in ref_grads:
        assert torch.allclose(ref_grads[name], new_grads[name], atol=1e-5), f"grad mismatch on {name}"


@pytest.mark.parametrize("M_divisor", [1, 2, 3, 5])
def test_grouped_loss_invariance_across_M(M_divisor: int):
    """Headline regression: loss + grads must be identical regardless of chunk size M.

    Failure of this test means the chunking changed the gradient semantics — the
    per-chunk-backward refactor's core correctness claim is broken.
    """
    torch.manual_seed(7)
    init_state = TinyCausalLM().state_dict()
    batch, prompt_ids = _build_batch([2, 4, 3], seed=1)
    total_rows = batch["input_ids"].shape[0]  # 9
    M = max(1, total_rows // M_divisor)

    # Reference: M = total_rows (no chunking)
    ref_model = TinyCausalLM()
    ref_model.load_state_dict(init_state)
    ref_accelerator = MockAccelerator()
    ref_loss = compute_grouped_step(ref_model, batch, prompt_ids, ref_accelerator, max_rows_per_forward=total_rows)
    ref_grads = _grads_snapshot(ref_model)

    # Chunked
    model = TinyCausalLM()
    model.load_state_dict(init_state)
    accelerator = MockAccelerator()
    loss = compute_grouped_step(model, batch, prompt_ids, accelerator, max_rows_per_forward=M)
    grads = _grads_snapshot(model)

    assert torch.allclose(loss, ref_loss, atol=1e-5), (
        f"loss differs at M={M} (total={total_rows}): chunked={loss.item()} vs full={ref_loss.item()}"
    )
    for name in ref_grads:
        assert torch.allclose(ref_grads[name], grads[name], atol=1e-5), f"grad mismatch on {name} at M={M}"


def test_grouped_loss_weights_sum_to_one_per_batch():
    """Sanity check on the normalization invariant that makes the math work."""
    _, prompt_ids = _build_batch([2, 4, 3], seed=0)
    _, inverse, counts = prompt_ids.unique(return_inverse=True, return_counts=True)
    num_prompts = counts.shape[0]
    weights = 1.0 / (num_prompts * counts[inverse].float())
    assert abs(weights.sum().item() - 1.0) < 1e-6
