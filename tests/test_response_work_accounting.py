"""CPU contract checks without importing the CUDA/Ray training stack."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generic, TypeVar

import numpy as np
import pytest
import torch


def extracted(names, path, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture
def native():
    root = Path(__file__).resolve().parents[1] / "open_instruct"
    namespace = {
        "torch": torch,
        "np": np,
        "Any": Any,
        "dataclass": dataclass,
        "Generic": Generic,
        "T": TypeVar("T"),
        "data_types": SimpleNamespace(CollatedBatchData=lambda **kw: SimpleNamespace(**kw)),
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    extracted(
        ["PackedSequences", "reset_position_ids", "pack_sequences", "summarize_response_work"],
        root / "rl_utils.py",
        namespace,
    )
    extracted(["collate_fn", "prepare_collated_data_for_workers"], root / "data_loader.py", namespace)
    return namespace


@pytest.mark.parametrize("dp", [3, 4, 5])
@pytest.mark.parametrize("micro", [1, 2, 4])
def test_packing_matches_independent_tokens_and_ids(native, dp, micro):
    lengths = [2, 4, 9, 12] * 64
    packed = native["pack_sequences"](
        queries=[[1]] * 256,
        responses=[[2] * n for n in lengths],
        masks=[[1] * n for n in lengths],
        pack_length=16,
        pad_token_id=0,
        vllm_logprobs=[[-1.0] * n for n in lengths],
        rollout_sample_ids=list(range(256)),
        min_num_batches=dp,
    )
    packed.advantages = [torch.ones_like(mask, dtype=torch.float) for mask in packed.response_masks]
    workers = native["prepare_collated_data_for_workers"](packed, dp, micro, 0, pin_memory=False)
    received = native["summarize_response_work"](packed.response_masks, packed.rollout_sample_ids)
    masks = [m for w in workers for m in w.response_masks]
    ids = [i for w in workers for i in w.rollout_sample_ids]
    prepared = native["summarize_response_work"](masks, ids, shifted=True)
    keep = len(packed.query_responses) // dp * dp
    oracle_ids = set()
    oracle_tokens = 0
    for mask, sample_ids in zip(packed.response_masks[:keep], packed.rollout_sample_ids[:keep], strict=True):
        for valid, sample_id in zip(mask.tolist(), sample_ids.tolist(), strict=True):
            if valid > 0:
                oracle_tokens += 1
                oracle_ids.add(sample_id)
    assert received["tokens"] == sum(lengths)
    assert received["sample_ids"] == list(range(256))
    assert prepared == {"tokens": oracle_tokens, "packs": keep, "sample_ids": sorted(oracle_ids)}
    assert received["tokens"] - prepared["tokens"] >= 0


def test_masks_ids_zero_padding_tools_and_shift(native):
    masks = [torch.tensor([[1, 2, 0, 2, 0], [0, 0, 0, 0, 0]])]
    ids = [torch.tensor([[0, 7, 7, 7, -1], [-1, -1, -1, -1, -1]])]
    count = native["summarize_response_work"]
    assert count(masks, ids) == {"tokens": 3, "packs": 2, "sample_ids": [0, 7]}
    assert count(masks, ids, shifted=True) == {"tokens": 2, "packs": 2, "sample_ids": [7]}
    assert count([], []) == {"tokens": 0, "packs": 0, "sample_ids": []}


@pytest.mark.parametrize(
    "masks,ids",
    [([torch.ones(2)], []), ([torch.ones(2)], [torch.zeros(3)]), ([torch.ones(2)], [torch.full((2,), -1)])],
)
def test_invalid_identity_fails_closed(native, masks, ids):
    with pytest.raises(ValueError):
        native["summarize_response_work"](masks, ids)
