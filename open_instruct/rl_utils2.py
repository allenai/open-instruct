import time
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import numpy as np
import torch
from rich.pretty import pprint

T = TypeVar("T")


class Timer:
    """A context manager for timing code blocks"""

    def __init__(self, description: str, noop: int = 0):
        self.description = description
        self.noop = noop

    def __enter__(self):
        if self.noop:
            return
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if self.noop:
            return
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.description}: {self.duration} seconds")


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
    advantages: Optional[np.ndarray] = None
    """packed advantages (batch_size, pack_length) (to be filled in by the main process)"""
    num_actions: Optional[np.ndarray] = None
    """packed number of actions (batch_size, pack_length)"""
    position_ids: Optional[np.ndarray] = None
    """packed position ids (batch_size, pack_length)"""
    packed_seq_lens: Optional[np.ndarray] = None
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
    pack_length: int,
    pad_token_id: int,
) -> PackedSequences:
    # assert padding token does not exist in queries and responses
    assert not any(pad_token_id in query for query in queries)
    # assert not any(pad_token_id in response for response in responses)

    query_responses = []
    attention_masks = []
    response_masks = []
    dones = []
    num_actions = []
    packed_seq_lens = []
    cur_data = []
    cur_response_mask = []
    cur_num_actions = []
    cur_packed_seq_lens = []
    cur_attention_mask = []
    cur_dones = []
    offset = 0
    for i in range(len(queries)):
        query = queries[i]
        response = responses[i]
        # remove padding (but using vllm so this should not be needed, but just in case)
        query = [t for t in query if t != pad_token_id]
        response = [t for t in response if t != pad_token_id]
        query_response = query + response
        if len(query_response) + len(cur_data) > pack_length:
            query_responses.append(cur_data)
            response_masks.append(cur_response_mask)
            attention_masks.append(cur_attention_mask)
            num_actions.append(cur_num_actions)
            packed_seq_lens.append(cur_packed_seq_lens)
            dones.append(cur_dones)
            cur_data = []
            cur_response_mask = []
            cur_attention_mask = []
            cur_num_actions = []
            cur_packed_seq_lens = []
            cur_dones = []
            offset = i
        cur_data.extend(query_response)
        cur_num_actions.append(len(response))
        cur_packed_seq_lens.append(len(query_response))

        # @vwxyzjn: here we use i + 1 to avoid 0 as a response mask token; the actual number should corresponds to
        # the response's index
        cur_response_mask.extend([0 for _ in range(len(query))] + [i + 1 for _ in range(len(response))])
        cur_attention_mask.extend([i + 1 - offset for _ in range(len(query_response))])
        cur_dones.extend([0 for _ in range(len(query) + len(response) - 1)] + [i + 1])
    if len(cur_data) > 0:
        query_responses.append(cur_data)
        response_masks.append(cur_response_mask)
        attention_masks.append(cur_attention_mask)
        num_actions.append(cur_num_actions)
        packed_seq_lens.append(cur_packed_seq_lens)
        dones.append(cur_dones)
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
    )


# TODO: still need to whiten the advantages
def get_test_data():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    prompts = [
        "User: Hello, how are you?\nAssistant: <think>",
        "User: What is the capital of France?\nAssistant: <think>",
        "User: What is the capital of Germany?\nAssistant: <think>",
    ]
    outputs = [
        "I'm good, thank you!",
        "Paris",
        "Berlin",
    ]
    queries = [tokenizer.encode(prompt) for prompt in prompts]
    responses = [tokenizer.encode(response) for response in outputs]
    prompt_max_len = 20
    generate_max_len = 20
    assert all(len(query) <= prompt_max_len for query in queries)
    assert all(len(response) <= generate_max_len for response in responses)
    return queries, responses, tokenizer.pad_token_id


def test_pack_sequences():
    queries, responses, pad_token_id = get_test_data()
    pack_length = 40
    with Timer("pack_sequences"):
        packed_sequences = pack_sequences(
            queries=queries, responses=responses, pack_length=pack_length, pad_token_id=pad_token_id
        )

    # uncomment to debug
    for q, r in zip(queries, responses):
        pprint([q, r])
    pprint(packed_sequences)


    # assert first and second sequence is correct
    offset = 0
    np.testing.assert_allclose(
        packed_sequences.query_responses[0][offset:offset + len(queries[0]) + len(responses[0])],
        np.array(sum([queries[0], responses[0]], [])),
    )
    offset += len(queries[0]) + len(responses[0])
    np.testing.assert_allclose(
        packed_sequences.query_responses[0][offset:offset + len(queries[1]) + len(responses[1])],
        np.array(sum([queries[1], responses[1]], [])),
    )
    offset = 0
    np.testing.assert_allclose(
        packed_sequences.query_responses[1][offset:offset + len(queries[2]) + len(responses[2])],
        np.array(sum([queries[2], responses[2]], [])),
    )


def print_diff(actual: torch.Tensor, expected: torch.Tensor):
    atol = torch.abs(actual - expected)
    rtol = atol / expected
    print(f"{atol.mean()=}, {rtol.mean()=}")


@torch.no_grad()
def test_pack_sequences_logits():
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-14m",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    # set seed for reproducibility
    torch.manual_seed(0)
    value_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m", num_labels=1)
    value_model.save_pretrained("pythia-14m-critic")
    queries, responses, pad_token_id = get_test_data()
    query_responses = [q + r for q, r in zip(queries, responses)]
    pack_length = 40
    with Timer("pack_sequences"):
        packed_sequences = pack_sequences(
            queries=queries, responses=responses, pack_length=pack_length, pad_token_id=pad_token_id
        )
    # NOTE: it's very important to use [:, :-1] here, because the produced tokens's index is shifted by 1
    s = model.forward(
        input_ids=packed_sequences.query_responses[0].unsqueeze(0)[:, :-1],
        attention_mask=packed_sequences.attention_masks[0].unsqueeze(0)[:, :-1],
        position_ids=packed_sequences.position_ids[0].unsqueeze(0)[:, :-1],
    )
    lm_backbone = getattr(value_model, value_model.base_model_prefix)
    _ = lm_backbone(
        input_ids=packed_sequences.query_responses[0].unsqueeze(0),
        attention_mask=packed_sequences.attention_masks[0].unsqueeze(0),
        position_ids=packed_sequences.position_ids[0].unsqueeze(0),
        return_dict=True,
        output_hidden_states=True,
    )
    # scalar_logits = value_model.score(v.hidden_states[-1]).squeeze(-1)
    # scalar_logits = torch.where(packed_sequences.response_masks[0].unsqueeze(0) == 0, float('-inf'), scalar_logits)
    # pprint("scalar_logits")
    # pprint(scalar_logits)

    # apply softmax to get logprobs
    all_logprobs = torch.nn.functional.log_softmax(s.logits, dim=-1)
    logprobs = torch.gather(
        all_logprobs, 2, packed_sequences.query_responses[0].unsqueeze(0).unsqueeze(-1)[:, 1:]
    ).squeeze(-1)
    logprobs = torch.where(packed_sequences.response_masks[0].unsqueeze(0)[:, 1:] == 0, 1.0, logprobs)
    pprint("logprobs")
    pprint(logprobs)

    logprobs = []
    for i in range(len(query_responses)):
        query_response = query_responses[i]
        query = queries[i]
        s2 = model.forward(
            input_ids=torch.tensor([query_response])[:, :-1],
            attention_mask=torch.tensor([query_response])[:, :-1] != pad_token_id,
        )
        all_logprobs = torch.nn.functional.log_softmax(s2.logits, dim=-1)
        logprob = torch.gather(all_logprobs, 2, torch.tensor([query_response]).unsqueeze(-1)[:, 1:]).squeeze(-1)
        logprobs.append(logprob[:, len(query) - 1 :])
    pprint(logprobs)


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


def calculate_advantages_packed(values: torch.Tensor, rewards: torch.Tensor, gamma: float, lam: float, dones: torch.Tensor, response_masks: torch.Tensor):
    """Packed implementation of GAE. Each row is a packed sequence.
    The `dones` specifies the sequence boundaries, and the `response_masks` specifies the query boundaries.
    """
    lastgaelam = 0
    advantages_reversed = []
    gen_length = values.shape[1]
    for t in reversed(range(gen_length)):
        nonterminal = 1 - dones[:, t]
        nonquery = response_masks[:, t]
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues * nonterminal * nonquery - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam * nonterminal * nonquery
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages, returns

def test_calculate_advantages_packed():
    gamma = 0.99
    lam = 0.98
    values = np.array([
        [1, 1, 1, 1, 1, 0, 0, 0 ,0, 0, 0, 0],
        [2, 2, 2, 2, 2, 2, 0, 0 ,0, 0, 0, 0],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    ])
    rewards = np.array([
        [0, 0, 0, 0, 10, 0, 0, 0 ,0, 0, 0, 0],
        [0, 0, 0, 0, 0, 20, 0, 0 ,0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30],
    ])
    adv, ret = calculate_advantages(values, rewards, gamma, lam)

    # here we assume -1 is the prompt token that should be ignored
    # 50256 is the pad token
    packed_values = np.array([[
        -1, -1, -1, -1, 1, 1, 1, 1, 1,
        -1, -1, -1, 2, 2, 2, 2, 2, 2,
        -1, -1, -1, -1, -1, -1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 50256,
    ]])
    packed_rewards = np.array([[
        0, 0, 0, 0, 0, 0, 0, 0, 10,
        0, 0, 0, 0, 0, 0, 0, 0, 20,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 50256,
    ]])
    packed_dones = np.array([[
        0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 50256,
    ]])
    packed_response_masks = np.array([[
        0, 0, 0, 0, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 50256,
    ]])
    packed_adv, packed_ret = calculate_advantages_packed(packed_values,
        packed_rewards,
        gamma,
        lam,
        packed_dones,
        packed_response_masks
    )
    # uncomment to debug
    # print("vanilla GAE implementation")
    # print(adv.round(2))
    # print(ret.round(2))
    # print("packed GAE implementation with masked prompts")
    # print(packed_adv.round(2))
    # print(packed_ret.round(2))

    # actual test cases
    np.testing.assert_allclose(adv[0, :5], packed_adv[0, 4:9])
    np.testing.assert_allclose(ret[1, :6], packed_ret[0, 12:18])
    np.testing.assert_allclose(adv[2, :12], packed_adv[0, 24:36])


if __name__ == "__main__":
    test_pack_sequences()
    # test_pack_sequences_logits()
    test_calculate_advantages_packed()
