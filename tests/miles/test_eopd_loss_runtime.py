"""Load and execute the EOPD custom loss against the pinned Miles runtime."""

import pytest
import torch
from miles.backends.training_utils import parallel
from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.loss.hub import losses
from miles.utils.ft_utils.process_group_utils import GroupInfo
from test_contract import batch_for, fixture_data, loss_args


@pytest.mark.parametrize("qkv_format", ["bshd", "thd"])
@pytest.mark.parametrize("tau", [0.5, 0.8])
def test_eopd_custom_loss_matches_forward_kl_and_gradients(monkeypatch, qkv_format, tau):
    trivial = GroupInfo(rank=0, size=1, group=None)
    state = parallel.ParallelState(
        intra_dp=trivial,
        intra_dp_cp=trivial,
        cp=trivial,
        tp=trivial,
        pp=trivial,
        ep=trivial,
        etp=trivial,
        indep_dp=trivial,
    )
    monkeypatch.setattr(parallel, "_parallel_state", state)
    monkeypatch.setenv("OI_OPD_EOPD_TOP_K", "2")
    monkeypatch.setenv("OI_OPD_EOPD_ALPHA", "0.7")
    monkeypatch.setenv("OI_OPD_EOPD_TAU", str(tau))
    initial, samples = fixture_data()
    batch = batch_for(samples, list(range(len(samples))))
    batch["metadata"] = []
    for sample in samples:
        probabilities = torch.tensor([[0.3, 0.3], [0.8, 0.01]])
        rows = torch.arange(sample["response"]) % 2
        batch["metadata"].append(
            {"eopd_topk_ids": [[1, 7]] * sample["response"], "eopd_topk_logprobs": probabilities[rows].log().tolist()}
        )
    if qkv_format == "thd":
        initial = torch.cat([row[: s["total"]] for row, s in zip(initial, samples, strict=True)]).unsqueeze(0)
    args = loss_args(False)
    args.qkv_format = qkv_format
    args.log_probs_chunk_size = 2
    args.loss_type = "custom_loss"
    args.custom_loss_function_path = "open_instruct.miles.distillation.eopd_loss.policy_loss"
    reducer = context_parallel.get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        False,
        qkv_format,
        batch["max_seq_lens"],
    )
    # Exercise native custom-loss loading, response slicing, policy loss and reducer.
    logits = initial.clone().requires_grad_()
    loss, metrics = losses.get_loss_function(args)(args, batch, logits, reducer)
    loss.backward()

    reference_logits = initial.clone().requires_grad_()
    base_loss, base_metrics = losses.policy_loss_function(args, batch, reference_logits, reducer)
    expected_fkl = reference_logits.new_zeros(())
    expected_gate = reference_logits.new_zeros(())
    offset = 0
    for index, (sample, metadata) in enumerate(zip(samples, batch["metadata"], strict=True)):
        start, end = sample["total"] - sample["response"] - 1, sample["total"] - 1
        response_logits = (
            reference_logits[0, offset + start : offset + end]
            if qkv_format == "thd"
            else reference_logits[index, start:end]
        )
        offset += sample["total"]
        teacher = torch.tensor(metadata["eopd_topk_logprobs"]).softmax(-1)
        student_log_probs = response_logits.log_softmax(-1)[:, [1, 7]]
        gate = (-(teacher * teacher.log()).sum(-1) > tau).float()
        fkl = (teacher * (teacher.log() - student_log_probs)).sum(-1)
        mask = sample["mask"]
        expected_fkl = expected_fkl + (mask * gate * fkl).sum() / mask.sum()
        expected_gate = expected_gate + (mask * gate).sum() / mask.sum()
    expected = base_loss + 0.7 * expected_fkl
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    torch.testing.assert_close(metrics["loss"], loss.detach())
    torch.testing.assert_close(metrics["pg_loss"], base_metrics["pg_loss"])
    torch.testing.assert_close(metrics["eopd_fkl_loss"], expected_fkl.detach())
    torch.testing.assert_close(metrics["eopd_gate_frac"], expected_gate)
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() > 0
    assert all(not metric.requires_grad for metric in metrics.values())
    if tau == 0.5:
        assert 0 < expected_gate < len(samples)
        assert expected_fkl > 0
    else:
        assert expected_gate == expected_fkl == 0
