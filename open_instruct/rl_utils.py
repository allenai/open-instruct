# flake8: noqa
import time
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import numpy as np
import torch
from rich.pretty import pprint

from open_instruct import logger_utils

T = TypeVar("T")
logger = logger_utils.setup_logger(__name__)


class Timer:
    """A context manager for timing code blocks"""

    def __init__(self, description: str, noop: bool = False):
        self.description = description
        self.noop = noop

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
    query_responses: np.ndarray
    """packed query and response (batch_size, pack_length)"""
    attention_masks: np.ndarray
    """3D attention mask for packed sequences (batch_size, pack_length, pack_length);
    it basically uses a intra-document mask for each query response pair;
    see https://huggingface.co/blog/sirluk/llm-sequence-packing for more details
    """
    response_masks: np.ndarray
    """response mask for packed sequences (batch_size, pack_length)"""
    original_responses: np.ndarray
    """need the original response for broadcast (batch_size, response_length)"""
    tool_masks: Optional[np.ndarray] = None
    """tool mask for packed sequences (batch_size, pack_length)"""
    advantages: Optional[np.ndarray] = None
    """packed advantages (batch_size, pack_length) (to be filled in by the main process)"""
    num_actions: Optional[np.ndarray] = None
    """packed number of actions (batch_size, pack_length)"""
    position_ids: Optional[np.ndarray] = None
    """packed position ids (batch_size, pack_length)"""
    packed_seq_lens: Optional[np.ndarray] = None
    vllm_logprobs: Optional[np.ndarray] = None
    """packed vLLM logprobs for comparison (batch_size, pack_length)"""
    """packed sequence lengths (batch_size, pack_length)"""
    dones: Optional[np.ndarray] = None
    """packed dones (batch_size, pack_length), specifies the sequence boundaries
    E.g., [0, 0, 0, 0, 1, 0, 0, 0, 0, 2] means the first sequence ends at index 4, and the
    second sequence ends at index 9
    """
    rewards: Optional[np.ndarray] = None
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
) -> PackedSequences:
    assert not any(pad_token_id in query for query in queries)
    # TODO: for some reason vLLM *can* generate the padding token in the responses; investigate
    # assert not any(pad_token_id in response for response in responses)

    query_responses = []
    tool_masks = []
    attention_masks = []
    response_masks = []
    dones = []
    num_actions = []
    packed_seq_lens = []
    packed_vllm_logprobs = []
    cur_data = []
    cur_tool_mask = []
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
        query_tool_mask = [1 for t in query if t != pad_token_id]
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
        mask = query_tool_mask + response_tool_mask

        # Process vLLM logprobs
        # For query tokens, we set logprobs to NaN, for response tokens we use vLLM logprobs
        query_logprobs = [float("nan")] * len(query)
        assert len(response_logprobs) == len(response), (
            f"Response {i}: logprobs length {len(response_logprobs)} != response length {len(response)}. "
            f"Original lengths before filtering: response={len(responses[i])}, logprobs={len(vllm_logprobs[i])}. "
            f"This can happen if vLLM returns N-1 logprobs for N tokens (missing first token logprob)."
        )
        combined_logprobs = query_logprobs + response_logprobs
        if len(query_response) + len(cur_data) > pack_length:
            query_responses.append(cur_data)
            tool_masks.append(cur_tool_mask)
            response_masks.append(cur_response_mask)
            attention_masks.append(cur_attention_mask)
            num_actions.append(cur_num_actions)
            packed_seq_lens.append(cur_packed_seq_lens)
            dones.append(cur_dones)
            packed_vllm_logprobs.append(cur_vllm_logprobs)
            cur_data = []
            cur_tool_mask = []
            cur_response_mask = []
            cur_attention_mask = []
            cur_num_actions = []
            cur_packed_seq_lens = []
            cur_dones = []
            cur_vllm_logprobs = []
            offset = i
        cur_data.extend(query_response)
        cur_tool_mask.extend(mask)
        cur_vllm_logprobs.extend(combined_logprobs)
        cur_num_actions.append(len(response))
        cur_packed_seq_lens.append(len(query_response))

        # @vwxyzjn: here we use i + 1 to avoid 0 as a response mask token;
        # the actual number should corresponds to the response's index.
        cur_response_mask.extend([0 for _ in range(len(query))] + [i + 1 for _ in range(len(response))])
        cur_attention_mask.extend([i + 1 - offset for _ in range(len(query_response))])
        cur_dones.extend([0 for _ in range(len(query) + len(response) - 1)] + [i + 1])

    # Handle leftover data
    if len(cur_data) > 0:
        query_responses.append(cur_data)
        tool_masks.append(cur_tool_mask)
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
        response_masks=[torch.tensor(t) for t in response_masks],
        original_responses=responses,
        num_actions=[torch.tensor(t) for t in num_actions],
        packed_seq_lens=[torch.tensor(t) for t in packed_seq_lens],
        dones=[torch.tensor(t) for t in dones],
        tool_masks=[torch.tensor(t) for t in tool_masks],
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
