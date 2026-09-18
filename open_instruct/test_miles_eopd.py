"""CPU checks for the EOPD forward-KL helpers, including the tensor-parallel shard path."""

import math
import os
import tempfile

import pytest
import torch
from torch import distributed, multiprocessing

from open_instruct.miles import eopd_math


def _teacher_topk(teacher_logits, k):
    log_probs = torch.log_softmax(teacher_logits, dim=-1)
    top = log_probs.topk(k, dim=-1)
    return top.indices, top.values


def test_student_log_probs_and_forward_kl_match_dense_definitions():
    torch.manual_seed(0)
    rows, vocab, k = 5, 40, 4
    student_logits = torch.randn(rows, vocab, requires_grad=True)
    ids, teacher = _teacher_topk(torch.randn(rows, vocab) * 2, k)
    student = eopd_math.student_log_probs_at(student_logits, ids)
    dense = torch.log_softmax(student_logits, dim=-1).gather(-1, ids)
    torch.testing.assert_close(student, dense)
    q = torch.softmax(teacher, dim=-1)
    expected = (q * (q.log() - dense)).sum(-1)
    torch.testing.assert_close(eopd_math.forward_kl(teacher, student), expected)
    torch.testing.assert_close(eopd_math.student_log_probs_at(student_logits, ids, chunk_size=2), dense)
    # Gradients flow to the student logits (dense form: q~ at the ids minus the student softmax mass).
    eopd_math.forward_kl(teacher, student).sum().backward()
    assert student_logits.grad is not None and torch.isfinite(student_logits.grad).all()


def test_gate_proxy_entropy_and_mass():
    k = 16
    flat = torch.full((2, k), math.log(1 / 64))  # 16 of 64 equally likely tokens: mass 0.25, entropy log 16
    peaked = torch.tensor([[0.0] + [-30.0] * (k - 1)])
    torch.testing.assert_close(eopd_math.proxy_entropy(flat), torch.full((2,), math.log(k)))
    torch.testing.assert_close(eopd_math.topk_mass(flat), torch.full((2,), 0.25))
    assert eopd_math.gate(flat, 0.8).tolist() == [1.0, 1.0]
    assert eopd_math.gate(peaked, 0.8).tolist() == [0.0]
    assert eopd_math.forward_kl(peaked, peaked).abs().max() < 1e-6


def test_parse_top_entries_and_sample_tensors_validate_shapes():
    entries = [[[-0.1, 2], [-2.0, 7]], [[-0.5, 3], [-0.9, 8]]]
    ids, log_probs = eopd_math.parse_top_entries(entries, 2)
    assert ids.tolist() == [[2, 7], [3, 8]]
    assert log_probs.dtype == torch.float32
    with pytest.raises(ValueError, match="expected 2"):
        eopd_math.parse_top_entries([[[-0.1, 2]]], 2)
    with pytest.raises(ValueError, match="nonfinite"):
        eopd_math.parse_top_entries([[[float("nan"), 2], [-1.0, 3]]], 2)
    metadata = {"eopd_topk_ids": ids.tolist(), "eopd_topk_logprobs": log_probs.tolist()}
    back = eopd_math.sample_tensors(metadata, 2)
    torch.testing.assert_close(back[1], log_probs)
    with pytest.raises(ValueError, match=r"expected \[R, 3\]"):
        eopd_math.sample_tensors(metadata, 3)
    with pytest.raises(ValueError, match="lacks"):
        eopd_math.sample_tensors({}, 2)
    with pytest.raises(ValueError, match="outside the student vocabulary"):
        eopd_math.student_log_probs_at(torch.zeros(1, 4), torch.tensor([[4, 1]]))


def test_settings_round_trip_through_the_environment():
    assert not eopd_math.Settings.from_distillation({"eopd": False}).enabled
    settings = eopd_math.Settings.from_distillation(
        {"eopd": True, "eopd_top_k": 8, "eopd_alpha": 0.5, "eopd_tau": 0.7}
    )
    assert settings == eopd_math.Settings.from_environment(settings.environment())
    assert not eopd_math.Settings.from_environment({}).enabled
    with pytest.raises(ValueError, match="Invalid EOPD"):
        eopd_math.Settings.from_environment({"OI_OPD_EOPD_TOP_K": "4", "OI_OPD_EOPD_ALPHA": "0"})


def _shard_worker(rank, world, init_file, out_dir, logits, ids):
    distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    try:
        local_size = logits.shape[-1] // world
        shard = logits[:, rank * local_size : (rank + 1) * local_size].clone().requires_grad_(True)
        value = eopd_math.student_log_probs_at(
            shard, ids, vocab_start=rank * local_size, group=distributed.group.WORLD, chunk_size=2
        )
        (value * torch.arange(1, value.numel() + 1).reshape(value.shape)).sum().backward()
        torch.save({"value": value.detach(), "grad": shard.grad}, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        distributed.destroy_process_group()


def test_sharded_student_log_probs_match_the_full_vocabulary():
    """Two gloo ranks each hold half the vocabulary, as Megatron tensor parallelism does."""
    torch.manual_seed(1)
    rows, vocab, k, world = 5, 32, 3, 2
    logits = torch.randn(rows, vocab)
    ids = torch.stack([torch.randperm(vocab)[:k] for _ in range(rows)])
    with tempfile.TemporaryDirectory() as folder:
        multiprocessing.spawn(
            _shard_worker, args=(world, os.path.join(folder, "init"), folder, logits, ids), nprocs=world, join=True
        )
        outputs = [torch.load(os.path.join(folder, f"rank{rank}.pt")) for rank in range(world)]
    reference = logits.clone().requires_grad_(True)
    dense = torch.log_softmax(reference, dim=-1).gather(-1, ids)
    (dense * torch.arange(1, dense.numel() + 1).reshape(dense.shape)).sum().backward()
    for rank, output in enumerate(outputs):
        torch.testing.assert_close(output["value"], dense.detach())
        local_size = vocab // world
        torch.testing.assert_close(output["grad"], reference.grad[:, rank * local_size : (rank + 1) * local_size])
