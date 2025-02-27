from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

import numpy as np

T = TypeVar("T")


def intra_document_mask(lenghts: list[int], max_length: Optional[int] = None) -> list[int]:
    total_length = sum(lenghts)
    L = total_length
    if max_length is not None:
        assert max_length >= total_length
        L = max_length
    mask = []
    cur_length = 0
    for i in range(len(lenghts)):
        for j in range(1, lenghts[i] + 1):
            mask.append(
                [0 for _ in range(cur_length)] + [1 for _ in range(j)] + [0 for _ in range(L - cur_length - j)]
            )
        cur_length += lenghts[i]
    mask.extend([0 for _ in range(L)] for _ in range(L - len(mask)))
    return np.array(mask)


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


def pack_sequences(
    queries: List[List[int]],
    responses: List[List[int]],
    pack_length: int,
    pad_token_id: int,
    response_mask_token_id: int = 0,
) -> PackedSequences:
    query_responses = []
    attention_masks = []
    response_masks = []
    cur_data = []
    cur_response_mask = []
    cur_lengths = []
    for i in range(len(queries)):
        query = queries[i]
        response = responses[i]
        # remove padding (but using vllm so this should not be needed, but just in case)
        query = [t for t in query if t != pad_token_id]
        response = [t for t in response if t != pad_token_id]
        query_response = query + response
        if len(query_response) + len(cur_data) > pack_length:
            cur_data.extend([pad_token_id for _ in range(pack_length - len(cur_data))])
            cur_response_mask.extend([response_mask_token_id for _ in range(pack_length - len(cur_response_mask))])
            query_responses.append(cur_data)
            response_masks.append(cur_response_mask)
            attention_masks.append(intra_document_mask(cur_lengths, pack_length))
            cur_data = []
            cur_lengths = []
            cur_response_mask = []
        cur_data.extend(query_response)
        cur_lengths.append(len(query_response))
        # here we use i + 1 to avoid 0 as a response mask token; the actual number should corresponds to
        # the response's index
        cur_response_mask.extend([0 for _ in range(len(query))] + [i + 1 for _ in range(len(response))])
    if len(cur_data) > 0:
        cur_data.extend([pad_token_id for _ in range(pack_length - len(cur_data))])
        cur_response_mask.extend([response_mask_token_id for _ in range(pack_length - len(cur_response_mask))])
        query_responses.append(cur_data)
        response_masks.append(cur_response_mask)
        attention_masks.append(intra_document_mask(cur_lengths, pack_length))
    return PackedSequences(
        query_responses=np.stack(query_responses),
        attention_masks=np.stack(attention_masks),
        response_masks=np.stack(response_masks),
        original_responses=responses,
    )


def test_intra_document_mask():
    np.testing.assert_allclose(
        intra_document_mask([3, 2, 3], 9),
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )


def test_intra_document_mask_error():
    # this goes wrong because the total length is greater than the max length
    try:
        intra_document_mask([3, 2, 3], 3)
    except AssertionError:
        pass
    else:
        raise AssertionError("should raise assertion error")


def get_test_data():
    p = 100  # padding token id
    o = 1  # observation (prompt / input ids)
    a = 2  # action (response ids)

    queries = [
        [p, p, o, o, o],
        [p, o, o, o, o],
        [p, p, p, o, o],
        [o, o, o, o, o],
    ]
    responses = [
        [a, p, p, p, p],
        [a, a, p, p, p],
        [a, p, p, p, p],
        [a, a, a, a, a],
    ]
    return queries, responses, p, o, a


def test_pack_sequences():
    queries, responses, p, o, a = get_test_data()
    pack_length = 13
    packed_sequences = pack_sequences(queries=queries, responses=responses, pack_length=pack_length, pad_token_id=p)
    np.testing.assert_allclose(
        packed_sequences.query_responses,
        np.array([[o, o, o, a, o, o, o, o, a, a, o, o, a], [o, o, o, o, o, a, a, a, a, a, p, p, p]]),
    )
    np.testing.assert_allclose(
        packed_sequences.attention_masks,
        np.array(
            [
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                ],
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        ),
    )
    np.testing.assert_allclose(
        packed_sequences.response_masks,
        np.array([[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]]),
    )


def test_pack_sequences_logits():
    queries, responses, p, _, _ = get_test_data()
    pack_length = 13
    query_responses = [q + r for q, r in zip(queries, responses)]
    packed_sequences = pack_sequences(queries=queries, responses=responses, pack_length=pack_length, pad_token_id=p)
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    print(torch.tensor(packed_sequences.attention_masks).unsqueeze(1).bool().shape)
    s = model.forward(
        input_ids=torch.tensor(packed_sequences.query_responses),
        attention_mask=torch.tensor(packed_sequences.attention_masks).unsqueeze(1).bool(),
    )
    s2 = model.forward(
        input_ids=torch.tensor(query_responses),
        attention_mask=torch.tensor(query_responses) != p,
    )
    print(s.logits.shape)
    print(s2.logits.shape)
    # test packed logits should be the same as raw logits
    torch.testing.assert_close(s.logits[0, 12], s2.logits[2, 5], atol=1e-4, rtol=1e-4)
    # test last sequence's logits should be the same
    torch.testing.assert_close(s.logits[1, 9], s2.logits[3, 9], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    test_intra_document_mask()
    test_intra_document_mask_error()
    test_pack_sequences()
    test_pack_sequences_logits()
