"""H235 fixed-budget Dr. GRPO reduction through MILES' custom-loss hook.

The outer MILES objective retains global response-count / DP scaling. This
adapter replaces every response's active-length denominator by the fixed
32768-token generation budget, including when TIS rebuilds the reducer.
"""

import torch
from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.loss.hub import losses

MAX_TOKENS = 32768


def policy_loss(args, batch, logits, sum_of_sample_mean):
    if args.calculate_per_token_loss or args.grpo_std_normalization or args.normalize_advantages:
        raise ValueError("Dr. GRPO requires response-count outer scaling and no std/advantage normalization")
    if args.rollout_max_response_len != MAX_TOKENS:
        raise ValueError("H235 Dr. GRPO requires a fixed 32768-token generation budget")
    if args.custom_pg_loss_reducer_function_path is not None:
        raise ValueError("A second policy-loss reducer would override Dr. GRPO")
    fixed_batch = dict(batch)
    fixed_batch["rollout_mask_sums"] = [
        torch.tensor(MAX_TOKENS, dtype=torch.float32, device=mask.device) for mask in batch["loss_masks"]
    ]
    reducer = context_parallel.get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        False,
        args.qkv_format,
        batch.get("max_seq_lens"),
        denominators=fixed_batch["rollout_mask_sums"],
    )
    return losses.policy_loss_function(args, fixed_batch, logits, reducer)
