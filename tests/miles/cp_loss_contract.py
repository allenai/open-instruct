"""Two-process CPU contract for native MILES Ulysses response loss/gradients."""

import json

import torch
from miles.backends.core_utils import context_parallel
from miles.backends.training_utils import parallel
from miles.backends.training_utils.data import context_parallel as response_cp
from miles.backends.training_utils.loss.hub import logit_processors
from miles.utils.ft_utils.process_group_utils import GroupInfo
from torch import distributed as dist


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    try:
        trivial, group = GroupInfo(0, 1, None), GroupInfo(rank, 2, dist.group.WORLD)
        parallel.set_parallel_state(
            parallel.ParallelState(
                intra_dp=trivial,
                intra_dp_cp=group,
                cp=group,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
                cp_comm_type="a2a",
            )
        )
        torch.manual_seed(77)
        logits = torch.randn(1, 32, 17, dtype=torch.float32)
        tokens = [torch.arange(9) % 17, (torch.arange(12) + 3) % 17]
        lengths, responses = [9, 12], [4, 6]
        masks = [torch.tensor([1, 0, 1, 1]), torch.tensor([1, 1, 0, 1, 1, 1])]
        reference = logits.clone().requires_grad_()
        expected, cursor = [], 0
        for sample, total, response in zip(tokens, lengths, responses, strict=True):
            rows = reference[0, cursor + total - response - 1 : cursor + total - 1]
            expected.append(rows.log_softmax(-1).gather(-1, sample[-response:, None]).squeeze(-1))
            cursor += total
        target = sum((value * mask).sum() / mask.sum() for value, mask in zip(expected, masks, strict=True))
        target.backward()
        local = logits[:, rank * 16 : (rank + 1) * 16].clone().requires_grad_()
        args = type("Args", (), dict(qkv_format="thd", allgather_cp=True, true_on_policy_mode=False))()
        chunks = logit_processors._iter_response_chunks(
            local,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=lengths,
            response_lengths=responses,
            max_seq_lens=None,
            include_response_indices=False,
        )
        result = {"log_probs": [rows.log_softmax(-1).gather(-1, ids[:, None]).squeeze(-1) for rows, ids, _ in chunks]}
        response_cp.allgather_cp_redistribute(
            result, logits=local, args=args, total_lengths=lengths, response_lengths=responses
        )
        reducer = response_cp.get_sum_of_sample_mean(lengths, responses, masks)
        loss = reducer(torch.cat(result["log_probs"])) + local.sum() * 0
        loss.backward()
        torch.testing.assert_close(local.grad, reference.grad[:, rank * 16 : (rank + 1) * 16], rtol=1e-6, atol=1e-7)
        full = context_parallel.full_scores(
            result["log_probs"], dict(total_lengths=lengths, response_lengths=responses)
        )
        for actual, wanted in zip(full, expected, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=1e-7)
        total_loss = loss.detach().clone()
        dist.all_reduce(total_loss)
        torch.testing.assert_close(total_loss, target.detach(), rtol=1e-6, atol=1e-7)
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        passed=True,
                        ranks=2,
                        docs=lengths,
                        response_lengths=responses,
                        local_tokens=16,
                        loss=float(total_loss),
                        max_gradient_tolerance=1e-6,
                    )
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
