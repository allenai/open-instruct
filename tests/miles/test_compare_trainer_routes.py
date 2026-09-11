"""Independent token-axis and unordered expert-set checks for native/serving traces."""

import pytest
import torch
from scripts.miles import compare_trainer_routes


def example():
    case = {"case_id": "fixture", "input_ids": [1, 2, 3, 4], "response_length": 2, "loss_mask": [1, 0]}
    logits = torch.tensor([[1.0, 4.0, 3.0, 0.0]]).repeat(4, 1)
    ids = torch.tensor([[1, 2]]).repeat(4, 1)
    weights = torch.tensor([[0.7, 0.3]]).repeat(4, 1)
    native = {
        "case_id": "fixture",
        "positions": [0, 1, 2, 3],
        "routes": {1: {"logits": logits, "topk_ids": ids, "topk_weights": weights}},
        "log_probs": torch.tensor([-0.5, -0.6]),
        "alignment_max_abs": 0.0,
    }
    serving = {
        "routes": {
            "model.layers.1.mlp.topk": {
                "logits": logits.clone(),
                "topk_ids": ids.flip(-1),
                "topk_weights": weights.flip(-1),
            }
        }
    }
    response = {
        "meta_info": {"input_token_logprobs": [[None, 1, None], [-10.0, 2, None], [-0.5, 3, None], [-0.6, 4, None]]}
    }
    return native, serving, response, case


def test_slot_reordering_and_correct_target_alignment():
    result = compare_trainer_routes.compare_case(*example())
    assert result["layers"][1]["exact_set_fraction"] == 1
    assert result["layers"][1]["weights"]["exact_values"]
    assert result["response_log_probs"]["exact_values"]
    assert result["response_log_probs"]["rows"]["positions"] == [2, 3]
    assert result["active_response_log_probs"]["rows"]["positions"] == [2]


def test_changed_expert_set_is_measurement_not_automatic_failure():
    native, serving, response, case = example()
    native["routes"][1]["topk_ids"][2, 1] = 3
    native["routes"][1]["logits"][2, 3] = 3.5
    result = compare_trainer_routes.compare_case(native, serving, response, case)
    assert result["layers"][1]["exact_set_fraction"] == 0.75
    assert result["layers"][1]["changed_set_input_positions"] == [2]


def test_serving_token_mismatch_and_missing_layer_rejected():
    native, serving, response, case = example()
    response["meta_info"]["input_token_logprobs"][2][1] = 99
    with pytest.raises(ValueError, match="token alignment"):
        compare_trainer_routes.compare_case(native, serving, response, case)
    response["meta_info"]["input_token_logprobs"][2][1] = 3
    serving["routes"] = {}
    with pytest.raises(ValueError, match="routed-layer"):
        compare_trainer_routes.compare_case(native, serving, response, case)


def test_masked_log_probability_difference_is_separate():
    native, serving, response, case = example()
    native["log_probs"][1] = -0.9
    result = compare_trainer_routes.compare_case(native, serving, response, case)
    assert result["response_log_probs"]["max_abs"] > 0.29
    assert result["active_response_log_probs"]["exact_values"]


def test_megatron_physical_mlp_layers_map_to_hf_logical_blocks():
    routes = {3: {"marker": "first MoE"}, 39: {"marker": "last MoE"}}
    assert compare_trainer_routes.logical_routes(routes, "megatron") == {1: routes[3], 19: routes[39]}
    assert compare_trainer_routes.logical_routes(routes, "olmo_core") is routes
    with pytest.raises(ValueError, match="odd physical"):
        compare_trainer_routes.logical_routes({2: {}}, "megatron")
