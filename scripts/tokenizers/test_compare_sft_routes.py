import copy

import pytest
import torch
from scripts.tokenizers import compare_sft_routes


def probe():
    return {
        "input_ids": torch.tensor([[10, 11, 12, 0]]),
        "doc_lens": torch.tensor([3]),
        "valid_mask": torch.tensor([[True, True, True, False]]),
        "routes": {"layer0": torch.tensor([[[0, 1], [1, 2], [2, 3], [0, 1]]])},
    }


def test_identical_routes_and_padding():
    before, after = probe(), probe()
    after["routes"]["layer0"][0, 3] = torch.tensor([5, 6])
    result = compare_sft_routes.compare_routes(before, after)["layer0"]
    assert result["tokens"] == 3
    assert result["changed_expert_set_count"] == 0
    assert result["retained_expert_fraction"] == 1
    assert result["expert_assignment_counts_before"] == {"0": 1, "1": 2, "2": 2, "3": 1}


def test_order_is_distinct_from_membership():
    before, after = probe(), probe()
    after["routes"]["layer0"][0, 0] = torch.tensor([1, 0])
    after["routes"]["layer0"][0, 1] = torch.tensor([1, 3])
    result = compare_sft_routes.compare_routes(before, after)["layer0"]
    assert result["changed_expert_set_count"] == 1
    assert result["order_only_change_count"] == 1
    assert result["retained_expert_fraction"] == pytest.approx(5 / 6)
    assert result["expert_assignment_counts_after"]["3"] == 2


@pytest.mark.parametrize("field", ["input_ids", "doc_lens", "valid_mask"])
def test_refuses_unaligned_inputs(field):
    before = probe()
    after = copy.deepcopy(before)
    after[field].flatten()[0] = 0
    with pytest.raises(ValueError, match=field):
        compare_sft_routes.compare_routes(before, after)


@pytest.mark.parametrize("fault", ["missing_layer", "shape", "duplicates", "float_ids"])
def test_refuses_invalid_routes(fault):
    before, after = probe(), probe()
    if fault == "missing_layer":
        after["routes"] = {}
    elif fault == "shape":
        after["routes"]["layer0"] = after["routes"]["layer0"][:, :3]
    elif fault == "duplicates":
        after["routes"]["layer0"][0, 0] = 1
    else:
        after["routes"]["layer0"] = after["routes"]["layer0"].float()
    with pytest.raises(ValueError):
        compare_sft_routes.compare_routes(before, after)
