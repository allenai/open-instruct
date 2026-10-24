"""Reject incomplete or numerically different mesh qualification evidence."""

import copy

import pytest
from compare_preflight import compare


def valid_receipt():
    return {
        "passed": True,
        "succeeded_pods": 32,
        "matched_hashes": 160,
        "expected_hashes": 160,
        "batch_hash_mismatches": [],
        "failures": [],
        "missing": [],
        "loss_parts": {str(i): {"tokens": 100, "ce": 1.0, "moe_lb": 0.0} for i in range(1, 21)},
        "drop_fraction": 0.0,
        "max_step_drop_fraction": 0.0,
        "job": "fixture",
    }


def test_complete_matched_pair():
    receipt = valid_receipt()
    result = compare(receipt, copy.deepcopy(receipt))
    assert result["passed"] and len(result["rows"]) == 20


@pytest.mark.parametrize("mutation", ["gap", "tokens", "incomplete", "aux"])
def test_invalid_pair_rejected(mutation):
    a = valid_receipt()
    b = copy.deepcopy(a)
    if mutation == "gap":
        b["loss_parts"]["6"]["ce"] = 1.002
    elif mutation == "tokens":
        b["loss_parts"]["20"]["tokens"] = 101
    elif mutation == "incomplete":
        b["matched_hashes"] = 159
    elif mutation == "aux":
        b["loss_parts"]["2"]["moe_lb"] = 0.2
    with pytest.raises(AssertionError):
        compare(a, b)
