"""Offline guards for the study's live transport gate; GPU evidence is separate."""

import copy

import pytest
import torch
from scripts.miles import qualify_math_weight_publication as gate


def scores(delta=0.0):
    return [{"meta_info": {"input_token_logprobs": [[None, 0]] + [[-2.0 + delta, i] for i in range(1, 40)]}}]


def test_fixed_token_gate_rejects_stale_alignment_nonfinite_and_drift():
    reference = scores()
    assert gate.compare_scores(reference, reference, "bf16", unchanged=True)["max_abs"] == 0
    with pytest.raises(ValueError, match="mismatch"):
        gate.compare_scores(scores(1e-5), reference, "bf16", unchanged=True)
    assert gate.compare_scores(scores(1e-5), reference, "bf16")["tokens"] == 39
    with pytest.raises(ValueError, match="mismatch"):
        gate.compare_scores(scores(0.01), reference, "bf16")
    for mutation in ("nan", "tokens", "missing", "null"):
        actual = copy.deepcopy(reference)
        if mutation == "nan":
            actual[0]["meta_info"]["input_token_logprobs"][4][0] = float("nan")
        elif mutation == "tokens":
            actual[0]["meta_info"]["input_token_logprobs"][4][1] = 999
        elif mutation == "missing":
            actual[0]["meta_info"]["input_token_logprobs"].pop()
        else:
            actual[0]["meta_info"]["input_token_logprobs"][4][0] = None
        with pytest.raises(ValueError):
            gate.compare_scores(actual, reference, "bf16")


def test_buckets_preserve_every_name_and_force_gate_up_split_across_layers():
    state = {"lm_head.weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    for layer in (1, 8, 15):
        for expert in (0, 1, 511):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                state[f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"] = torch.ones(
                    2, 2, dtype=torch.bfloat16
                )
    state["model.layers.1.mlp.shared_expert.up_proj.weight"] = torch.ones(2, 2, dtype=torch.bfloat16)

    class Reader:
        def keys(self):
            return state.keys()

        def get_tensor(self, name):
            return state[name]

    for variant in (0, 1, 2):
        buckets = list(gate.weight_buckets(Reader(), variant, 32))
        flattened = [(n, t) for bucket in buckets for n, t in bucket]
        assert sorted(n for n, _ in flattened) == sorted(state)
        assert sum(len(bucket) for bucket in buckets) == len(state)
        for bucket in buckets:
            assert sum(t.nbytes for _, t in bucket) <= 32
            for name, tensor in bucket:
                match = gate.EXPERT.fullmatch(name)
                if match and match[2] == "0":
                    assert len(bucket) == 1
                expected = 1.0 if not match or variant == 0 else (0.875, 1.125)[variant - 1]
                assert torch.equal(tensor, torch.full_like(tensor, expected))
        assert all(torch.equal(t, torch.ones_like(t)) for t in state.values())
