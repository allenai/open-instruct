# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

"""Utilities for converting vLLM prompt-logprob payloads into sparse tensors."""

from typing import Any

import torch


def process_prompt_logprobs(prompt_logprobs: Any, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract dense top-k token ids/logprobs from flat vLLM prompt logprob outputs."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    start_pos = 0
    if len(prompt_logprobs) > 0:
        first_start = prompt_logprobs.start_indices[0]
        first_end = prompt_logprobs.end_indices[0]
        if first_end - first_start == 0:
            start_pos = 1

    num_prompt_tokens = len(prompt_logprobs) - start_pos
    if num_prompt_tokens <= 0:
        return torch.empty((0, k), dtype=torch.long), torch.empty((0, k), dtype=torch.float32)

    top_indices = torch.zeros((num_prompt_tokens, k), dtype=torch.long, device="cpu")
    top_values = torch.full((num_prompt_tokens, k), fill_value=float("-inf"), dtype=torch.float32, device="cpu")

    seq_ids = []
    rank_ids = []
    token_ids_to_copy = []
    logprobs_to_copy = []

    for pos_id in range(start_pos, len(prompt_logprobs)):
        seq_id = pos_id - start_pos
        start_idx = prompt_logprobs.start_indices[pos_id]
        end_idx = prompt_logprobs.end_indices[pos_id]
        for i in range(start_idx, end_idx):
            rank = prompt_logprobs.ranks[i]
            if rank is None or rank > k:
                continue
            seq_ids.append(seq_id)
            rank_ids.append(rank - 1)
            token_ids_to_copy.append(prompt_logprobs.token_ids[i])
            logprobs_to_copy.append(prompt_logprobs.logprobs[i])

    if token_ids_to_copy:
        seq_idx_tensor = torch.tensor(seq_ids, dtype=torch.long)
        rank_idx_tensor = torch.tensor(rank_ids, dtype=torch.long)
        top_indices[seq_idx_tensor, rank_idx_tensor] = torch.tensor(token_ids_to_copy, dtype=top_indices.dtype)
        top_values[seq_idx_tensor, rank_idx_tensor] = torch.tensor(logprobs_to_copy, dtype=top_values.dtype)

    return top_indices, top_values


def extract_response_topk_from_prompt_logprobs(
    prompt_logprobs: Any, *, prompt_len: int, response_len: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract top-k targets for response tokens from a combined prompt payload.

    vLLM may omit the first prompt token's logprob entry. This helper accepts
    either alignment and returns rows corresponding to response tokens in
    `prompt_tokens + response_tokens`.
    """
    top_indices, top_values = process_prompt_logprobs(prompt_logprobs, k=k)
    combined_len = prompt_len + response_len

    if top_indices.shape[0] == combined_len:
        response_start = prompt_len
    elif top_indices.shape[0] == combined_len - 1:
        response_start = max(prompt_len - 1, 0)
    else:
        raise ValueError(
            f"Unexpected prompt_logprobs length after processing: got {top_indices.shape[0]}, "
            f"expected {combined_len} or {combined_len - 1}"
        )

    response_end = response_start + response_len
    if response_end > top_indices.shape[0]:
        raise ValueError(
            f"Not enough prompt_logprobs rows for response: start={response_start}, "
            f"response_len={response_len}, available={top_indices.shape[0]}"
        )
    return top_indices[response_start:response_end], top_values[response_start:response_end]
