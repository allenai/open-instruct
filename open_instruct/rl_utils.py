# flake8: noqa
import contextlib
import time
from dataclasses import dataclass, field
from typing import Generic, List, TypeVar

import numpy as np
import torch
from rich.pretty import pprint

from open_instruct import logger_utils

T = TypeVar("T")
logger = logger_utils.setup_logger(__name__)


@dataclass
class Timer(contextlib.ContextDecorator):
    """A context manager and decorator for timing code blocks"""

    description: str
    noop: bool = False
    start_time: float = field(init=False)
    end_time: float = field(init=False)
    duration: float = field(init=False)

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        if not self.noop:
            logger.info(f"{self.description}: {self.duration:.3f} seconds")


@dataclass
class PackedSequences(Generic[T]):
    query_responses: list[torch.Tensor]
    """packed query and response (batch_size, pack_length)"""
    attention_masks: list[torch.Tensor]
    """3D attention mask for packed sequences (batch_size, pack_length, pack_length);
    it basically uses a intra-document mask for each query response pair;
    see https://huggingface.co/blog/sirluk/llm-sequence-packing for more details
    """
    response_masks: list[torch.Tensor]
    """bool response mask for packed sequences (batch_size, pack_length)"""
    original_responses: list[list[int]]
    """need the original response for broadcast (batch_size, response_length)"""
    advantages: list[torch.Tensor] | None = None
    """packed advantages (batch_size, pack_length) (to be filled in by the main process)"""
    num_actions: list[torch.Tensor] | None = None
    """packed number of actions (batch_size, pack_length)"""
    position_ids: list[torch.Tensor] | None = None
    """packed position ids (batch_size, pack_length)"""
    packed_seq_lens: list[torch.Tensor] | None = None
    """packed sequence lengths (batch_size, pack_length)"""
    vllm_logprobs: list[torch.Tensor] | None = None
    """packed vLLM logprobs for comparison (batch_size, pack_length)"""
    dones: list[torch.Tensor] | None = None
    """packed dones (batch_size, pack_length), specifies the sequence boundaries
    E.g., [0, 0, 0, 0, 1, 0, 0, 0, 0, 2] means the first sequence ends at index 4, and the
    second sequence ends at index 9
    """
    rewards: list[torch.Tensor] | None = None
    """packed rewards (batch_size, pack_length)"""


def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def pack_sequences(
    queries: List[List[int]],
    responses: List[List[int]],
    masks: List[List[int]],
    pack_length: int,
    pad_token_id: int,
    vllm_logprobs: List[List[float]],
    min_num_batches: int = 1,
    mask_tool_use: bool = False,
) -> PackedSequences:
    """Pack query-response pairs into sequences for training.

    Args:
        queries: List of query token sequences
        responses: List of response token sequences
        masks: List of tool masks for each response
        pack_length: Maximum length of each packed sequence
        pad_token_id: Token ID used for padding
        vllm_logprobs: Log probabilities from vLLM for each response
        min_num_batches: Minimum number of packed batches to produce.
            Used to ensure we have a batch for each rank in distributed training.

    Returns:
        PackedSequences containing the packed training data.
    """
    assert not any(pad_token_id in query for query in queries)

    # Calculate total tokens to determine effective pack_length
    total_tokens = 0
    for query, response in zip(queries, responses):
        query_len = len(query)
        response_len = sum(1 for t in response if t != pad_token_id)
        total_tokens += query_len + response_len

    # Reduce pack_length if needed to ensure min_num_batches
    # Note: sequences longer than effective_pack_length will naturally get their own pack(s)
    # since the packing loop starts a new pack when a sequence doesn't fit
    if total_tokens > 0 and min_num_batches > 1:
        target_pack_length = total_tokens // min_num_batches
        # Don't exceed the original pack_length
        effective_pack_length = min(target_pack_length, pack_length)
    else:
        effective_pack_length = pack_length

    # TODO: for some reason vLLM *can* generate the padding token in the responses; investigate
    # assert not any(pad_token_id in response for response in responses)

    query_responses = []
    attention_masks = []
    response_masks = []
    dones = []
    num_actions = []
    packed_seq_lens = []
    packed_vllm_logprobs = []
    cur_data = []
    cur_response_mask = []
    cur_num_actions = []
    cur_packed_seq_lens = []
    cur_attention_mask = []
    cur_dones = []
    cur_vllm_logprobs = []
    offset = 0
    for i in range(len(queries)):
        query = queries[i]
        response = responses[i]
        mask = masks[i]
        # remove padding (but using vllm so this should not be needed, but just in case)
        query = [t for t in query if t != pad_token_id]

        # Filter out padding tokens from response, mask, and logprobs together
        response_logprobs_unfiltered = vllm_logprobs[i]

        assert len(response_logprobs_unfiltered) == len(response), (
            f"Response {i}: logprobs length ({len(response_logprobs_unfiltered)}) != response length ({len(response)})"
        )

        filtered_response = []
        filtered_mask = []
        filtered_logprobs = []
        for j, (token, mask_val) in enumerate(zip(response, mask)):
            if token != pad_token_id:
                filtered_response.append(token)
                filtered_mask.append(mask_val)
                filtered_logprobs.append(response_logprobs_unfiltered[j])

        response = filtered_response
        response_tool_mask = filtered_mask
        response_logprobs = filtered_logprobs

        query_response = query + response

        # Process vLLM logprobs
        # For query tokens, we set logprobs to NaN, for response tokens we use vLLM logprobs
        query_logprobs = [float("nan")] * len(query)
        assert len(response_logprobs) == len(response), (
            f"Response {i}: logprobs length {len(response_logprobs)} != response length {len(response)}. "
            f"Original lengths before filtering: response={len(responses[i])}, logprobs={len(vllm_logprobs[i])}. "
            f"This can happen if vLLM returns N-1 logprobs for N tokens (missing first token logprob)."
        )
        combined_logprobs = query_logprobs + response_logprobs
        # only flush if we have data and we exceed the pack length.
        if len(query_response) + len(cur_data) > effective_pack_length and len(cur_data) > 0:
            query_responses.append(cur_data)
            response_masks.append(cur_response_mask)
            attention_masks.append(cur_attention_mask)
            num_actions.append(cur_num_actions)
            packed_seq_lens.append(cur_packed_seq_lens)
            dones.append(cur_dones)
            packed_vllm_logprobs.append(cur_vllm_logprobs)
            cur_data = []
            cur_response_mask = []
            cur_attention_mask = []
            cur_num_actions = []
            cur_packed_seq_lens = []
            cur_dones = []
            cur_vllm_logprobs = []
            offset = i
        cur_data.extend(query_response)
        cur_vllm_logprobs.extend(combined_logprobs)
        cur_num_actions.append(len(response))
        cur_packed_seq_lens.append(len(query_response))

        query_mask = [0] * len(query)
        if mask_tool_use:
            response_mask = [(i + 1) if m else 0 for m in response_tool_mask]
        else:
            response_mask = [i + 1] * len(response)
        cur_response_mask.extend(query_mask + response_mask)
        cur_attention_mask.extend([i + 1 - offset for _ in range(len(query_response))])
        cur_dones.extend([0 for _ in range(len(query) + len(response) - 1)] + [i + 1])

    # Handle leftover data
    if len(cur_data) > 0:
        query_responses.append(cur_data)
        response_masks.append(cur_response_mask)
        attention_masks.append(cur_attention_mask)
        num_actions.append(cur_num_actions)
        packed_seq_lens.append(cur_packed_seq_lens)
        dones.append(cur_dones)
        packed_vllm_logprobs.append(cur_vllm_logprobs)
    attention_masks_list = [torch.tensor(t) for t in attention_masks]
    return PackedSequences(
        query_responses=[torch.tensor(t) for t in query_responses],
        attention_masks=attention_masks_list,
        position_ids=[reset_position_ids(t.unsqueeze(0)).squeeze(0) for t in attention_masks_list],
        response_masks=[torch.tensor(t, dtype=torch.long) for t in response_masks],
        original_responses=responses,
        num_actions=[torch.tensor(t) for t in num_actions],
        packed_seq_lens=[torch.tensor(t) for t in packed_seq_lens],
        dones=[torch.tensor(t) for t in dones],
        vllm_logprobs=[torch.tensor(t, dtype=torch.float) for t in packed_vllm_logprobs],
    )


def print_diff(actual: torch.Tensor, expected: torch.Tensor):
    atol = torch.abs(actual - expected)
    rtol = atol / expected
    print(f"{atol.mean()=}, {rtol.mean()=}")


def calculate_advantages(values: np.ndarray, rewards: np.ndarray, gamma: float, lam: float):
    """Vanilla implementation of GAE. Each row is a separate padded sequence."""
    lastgaelam = 0
    advantages_reversed = []
    gen_length = values.shape[1]
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = np.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def calculate_advantages_packed(
    values: np.ndarray, rewards: np.ndarray, gamma: float, lam: float, dones: np.ndarray, response_masks: np.ndarray
):
    """Packed implementation of GAE. Each row is a packed sequence.
    The `dones` specifies the sequence boundaries, and the `response_masks` specifies the query boundaries.
    """
    response_masks = response_masks.clip(0, 1)
    dones = dones.clip(0, 1)
    lastgaelam = 0
    advantages_reversed = []
    gen_length = values.shape[1]
    for t in reversed(range(gen_length)):
        nonterminal = 1 - dones[:, t]
        nonquery = response_masks[:, t]
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues * nonterminal * nonquery - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam * nonterminal * nonquery
        # print(
        #     f"t: {t}, rewards: {rewards[:, t]}, nextvalues: {nextvalues}, nonterminal: {nonterminal}, "
        #     f"delta: {delta}, lastgaelam: {lastgaelam}"
        # )
        advantages_reversed.append(lastgaelam)
    advantages = np.stack(advantages_reversed[::-1], axis=1)
    returns = advantages + values
    return advantages, returns


def masked_mean(
    values: torch.Tensor, mask: torch.Tensor, axis: int | None = None, denominator: float | None = None
) -> torch.Tensor:
    """Compute mean of tensor with masked values."""
    extra_dims = values.ndim - mask.ndim
    if axis is None:
        sum_dims = tuple(range(extra_dims, values.ndim))
    elif axis >= 0:
        sum_dims = axis + extra_dims
    else:
        sum_dims = axis
    numerator = (values * mask).sum(dim=sum_dims)
    denom = mask.sum(dim=axis) if denominator is None else denominator
    result = numerator / denom
    return result.flatten(extra_dims).mean(-1) if result.ndim > extra_dims else result
