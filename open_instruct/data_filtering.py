import logging
from typing import Any

import numpy as np

from open_instruct.grpo_fast import Args
from open_instruct.model_utils import Batch
from open_instruct.queue_types import GenerationResult
from open_instruct.rl_utils import timer

logger = logging.getLogger(__name__)


def filter_zero_gradient_prompts(
    scores: np.ndarray,
    advantages: np.ndarray,
    responses: list[list[int]],
    masks: list[list[int]],
    batch: Batch,
    finish_reasons: list[str],
    vllm_logprobs: list[Any],
    num_samples_per_prompt: int,
) -> tuple[np.ndarray, np.ndarray, list, list, Batch, list, list, dict[str, Any]]:
    scores_per_prompt = scores.reshape(-1, num_samples_per_prompt)
    non_zero_std_mask = scores_per_prompt.std(axis=-1) != 0
    expanded_mask = np.repeat(non_zero_std_mask, num_samples_per_prompt)
    non_zero_gradient_index = np.where(expanded_mask)[0]

    real_batch_size_ratio = non_zero_std_mask.sum() * num_samples_per_prompt / len(scores)

    stats = {
        "num_filtered_prompts": (~non_zero_std_mask).sum(),
        "num_filtered_responses": len(scores) - len(non_zero_gradient_index),
        "real_batch_size_ratio": real_batch_size_ratio,
    }

    return (
        scores[non_zero_gradient_index],
        advantages[non_zero_gradient_index],
        [responses[i] for i in non_zero_gradient_index],
        [masks[i] for i in non_zero_gradient_index],
        batch[non_zero_gradient_index.tolist()],
        [finish_reasons[i] for i in non_zero_gradient_index],
        [vllm_logprobs[i] for i in non_zero_gradient_index],
        stats,
    )


def maybe_mask_truncated_completions(
    scores: np.ndarray,
    advantages: np.ndarray,
    responses: list[list[int]],
    masks: list[list[int]],
    batch: Batch,
    finish_reasons: list[str],
    vllm_logprobs: list[Any],
    mask_truncated: bool,
) -> tuple[np.ndarray, np.ndarray, list, list, Batch, list, list, dict[str, Any]]:
    if not mask_truncated:
        stats = {"num_truncated": 0, "retention_rate": 1.0}
        return scores, advantages, responses, masks, batch, finish_reasons, vllm_logprobs, stats

    stop_idxes = [i for i in range(len(finish_reasons)) if finish_reasons[i] == "stop"]
    num_truncated = len(finish_reasons) - len(stop_idxes)
    retention_rate = len(stop_idxes) / len(finish_reasons) if len(finish_reasons) > 0 else 1.0

    stats = {"num_truncated": num_truncated, "retention_rate": retention_rate}

    return (
        scores[stop_idxes],
        advantages[stop_idxes],
        [responses[i] for i in stop_idxes],
        [masks[i] for i in stop_idxes],
        batch[stop_idxes],
        [finish_reasons[i] for i in stop_idxes],
        [vllm_logprobs[i] for i in stop_idxes],
        stats,
    )


def maybe_fill_completions(
    scores: np.ndarray,
    advantages: np.ndarray,
    responses: list[list[int]],
    masks: list[list[int]],
    batch: Batch,
    finish_reasons: list[str],
    vllm_logprobs: list[Any],
    target_prompt_count: int,
    num_samples_per_prompt: int,
    fill_completions: bool,
) -> tuple[np.ndarray, np.ndarray, list, list, Batch, list, list, dict[str, Any]]:
    if not fill_completions:
        stats = {"num_filled_prompts": 0, "num_filled_responses": 0}
        return scores, advantages, responses, masks, batch, finish_reasons, vllm_logprobs, stats

    current_batch_size = len(scores)
    current_prompt_cnt = current_batch_size // num_samples_per_prompt
    need_to_fill_prompt = target_prompt_count - current_prompt_cnt

    if need_to_fill_prompt <= 0 or current_prompt_cnt == 0:
        stats = {"num_filled_prompts": 0, "num_filled_responses": 0}
        return scores, advantages, responses, masks, batch, finish_reasons, vllm_logprobs, stats

    k = num_samples_per_prompt
    scores_matrix = scores.reshape(current_prompt_cnt, k)
    stds = scores_matrix.std(axis=1) + 1e-8
    probs = stds / stds.sum()

    sampled_prompt_ids = np.random.choice(current_prompt_cnt, size=need_to_fill_prompt, replace=True, p=probs)

    sampled_indices = []
    for pid in sampled_prompt_ids:
        start = pid * k
        sampled_indices.extend(range(start, start + k))

    stats = {"num_filled_prompts": need_to_fill_prompt, "num_filled_responses": len(sampled_indices)}

    return (
        np.concatenate([scores, scores[sampled_indices]]),
        np.concatenate([advantages, advantages[sampled_indices]]),
        responses + [responses[i] for i in sampled_indices],
        masks + [masks[i] for i in sampled_indices],
        batch.concatenate(batch[sampled_indices]),
        finish_reasons + [finish_reasons[i] for i in sampled_indices],
        vllm_logprobs + [vllm_logprobs[i] for i in sampled_indices],
        stats,
    )


@timer("📦 [Data Preparation Thread] Filtering sequences")
def apply_sequence_filters(
    scores: np.ndarray, advantages: np.ndarray, result: GenerationResult, batch: Batch, args: Args
) -> tuple[np.ndarray, np.ndarray, list, list, Batch, list, list, dict[str, Any]]:
    original_batch_size = len(scores)
    all_stats = {}

    max_possible_score = 0
    if args.apply_verifiable_reward:
        max_possible_score += args.verification_reward
    if args.apply_r1_style_format_reward and args.additive_format_reward:
        max_possible_score += args.r1_style_format_reward
    unsolved_batch_size_ratio = ((scores != max_possible_score) > 0).sum() / len(scores)
    all_stats["max_possible_score"] = max_possible_score
    all_stats["unsolved_batch_size_ratio"] = unsolved_batch_size_ratio

    scores_per_prompt = scores.reshape(-1, args.num_samples_per_prompt_rollout)
    all_zero_groups = (scores_per_prompt == 0).all(axis=-1).sum()
    total_groups = len(scores_per_prompt)
    all_stats["all_zero_groups"] = all_zero_groups
    all_stats["total_groups"] = total_groups
    logger.info(
        f"[Reward Summary] Groups with all zero rewards: {all_zero_groups}/{total_groups} "
        f"({all_zero_groups / total_groups:.1%})"
    )

    scores, advantages, responses, masks, batch, finish_reasons, vllm_logprobs, zero_grad_stats = (
        filter_zero_gradient_prompts(
            scores,
            advantages,
            result.responses,
            result.masks,
            batch,
            result.finish_reasons,
            result.logprobs,
            args.num_samples_per_prompt_rollout,
        )
    )
    all_stats["zero_gradient"] = zero_grad_stats

    real_batch_size_ratio = zero_grad_stats["real_batch_size_ratio"]
    if zero_grad_stats["num_filtered_responses"] > 0:
        logger.info(
            f"[Zero-gradient filtering] Filtered {zero_grad_stats['num_filtered_prompts']} prompts with zero std "
            f"({zero_grad_stats['num_filtered_responses']} responses). "
            f"Retention rate: {real_batch_size_ratio:.2%}"
        )

    scores, advantages, responses, masks, batch, finish_reasons, vllm_logprobs, truncated_stats = (
        maybe_mask_truncated_completions(
            scores, advantages, responses, masks, batch, finish_reasons, vllm_logprobs, args.mask_truncated_completions
        )
    )
    all_stats["truncated"] = truncated_stats

    if truncated_stats["num_truncated"] > 0:
        logger.info(
            f"[Truncated completions filtering] Filtered {truncated_stats['num_truncated']} responses that didn't finish with 'stop'. "
            f"Retention rate: {truncated_stats['retention_rate']:.2%}"
        )

    target_prompt_count = original_batch_size // args.num_samples_per_prompt_rollout
    scores, advantages, responses, masks, batch, finish_reasons, vllm_logprobs, fill_stats = maybe_fill_completions(
        scores,
        advantages,
        responses,
        masks,
        batch,
        finish_reasons,
        vllm_logprobs,
        target_prompt_count,
        args.num_samples_per_prompt_rollout,
        args.fill_completions,
    )
    all_stats["fill"] = fill_stats

    if fill_stats["num_filled_prompts"] > 0:
        logger.info(
            f"[Refill completions] Filled {fill_stats['num_filled_prompts']} prompts "
            f"({fill_stats['num_filled_responses']} responses) to maintain batch size"
        )

    return scores, advantages, responses, masks, batch, finish_reasons, vllm_logprobs, all_stats
