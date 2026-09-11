"""Check immutable objective injection and canonical FP32 layout preservation."""

import copy
import json
import sys
from types import SimpleNamespace

import core_policy_contract as capture
import pytest
import torch
from miles.utils import arguments
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.moe.v2.hf.modeling_olmo3moe import Olmo3MoeForCausalLM
from olmo_miles.evaluation import policy_contract_schema as schema

from open_instruct.miles import data


def fixture():
    return {
        "schema_version": 1,
        "objective": "policy_only",
        "vocab_size": 256,
        "seed": 173,
        "reduction": "response_mean",
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0, "clip_grad": 0.005},
        "samples": [
            {
                "id": str(index),
                "tokens": [1, 2, 3, 4 + index],
                "response_length": 2,
                "loss_mask": [1, 0],
                "advantages": [1.0 + index, -0.5],
                "old_log_probs": [-8.0, -3.0],
            }
            for index in range(2)
        ],
    }


def test_injection_preserves_advantages_masks_and_fixed_anchor():
    value = fixture()
    original = copy.deepcopy(value)
    rollout = capture.rollout_from_fixture(value, "cpu")
    capture.inject_advantages(value, rollout)
    assert [a.tolist() for a in rollout["advantages"]] == [a["advantages"] for a in value["samples"]]
    assert [a.tolist() for a in rollout["rollout_log_probs"]] == [a["old_log_probs"] for a in value["samples"]]
    assert [a.tolist() for a in rollout["loss_masks"]] == [a["loss_mask"] for a in value["samples"]]
    assert value == original
    rollout["tokens"].reverse()
    with pytest.raises(ValueError, match="sample order"):
        capture.inject_advantages(value, rollout)


def test_immutable_fixture_and_checkpoint_hashes(tmp_path):
    value = fixture()
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    (hf / "model.safetensors").write_bytes(b"explicit hash fixture")
    value["checkpoint"] = schema.checkpoint_inventory(hf)
    (tmp_path / "fixture.json").write_text(json.dumps(value))
    (tmp_path / "fixture.sha256").write_text(schema.fixture_digest(value))
    assert capture.load_fixture(tmp_path)[0] == value
    (hf / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        capture.load_fixture(tmp_path)
    value["samples"][0]["advantages"][0] *= 2
    (tmp_path / "fixture.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fixture digest"):
        capture.load_fixture(tmp_path)


def test_canonical_conversion_preserves_fp32_gradients(tmp_path, monkeypatch):
    hf = Olmo3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        attention_hidden_size=16,
        head_dim=8,
        dense_mlp_intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_head_qk_norm=True,
        use_rope=False,
        attention_gate_type="elementwise",
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        latent_moe_dim=8,
        layer_types=["linear_attention", "full_attention"],
        dense_layers_indices=[0],
        dense_layers_use_shared_expert=True,
        use_peri_ln=True,
        max_position_embeddings=128,
    )
    model = Olmo3MoeForCausalLM(hf)
    reference = {name: value.detach().float().clone() for name, value in model.state_dict().items()}
    native = capture.convert.convert_state_from_hf(hf, reference, model_type="olmo3moe")
    prefix = "blocks.1.attention."
    native[prefix + "w_qkv.weight"] = torch.cat(
        [native.pop(prefix + part + ".weight") for part in ("w_q", "w_k", "w_v")]
    )
    worker = SimpleNamespace(
        hf_config=hf, model=SimpleNamespace(get_submodule=lambda name: SimpleNamespace(shared_experts=True))
    )
    canonical = capture.canonical_state(worker, {"module." + name: value for name, value in native.items()})
    roundtrip = capture.convert.convert_state_from_hf(hf, canonical, model_type="olmo3moe")
    expected = capture.convert.convert_state_from_hf(hf, reference, model_type="olmo3moe")
    assert roundtrip.keys() == expected.keys()
    for name, value in roundtrip.items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    model.save_pretrained(tmp_path / "hf")
    (tmp_path / "prompts.jsonl").write_text('{"input":"fixture","label":"fixture"}\n')
    capture.moe_models.register_hf_classes()
    config = capture.configuration(tmp_path, tmp_path / "out", fixture())
    monkeypatch.setattr(sys, "argv", ["core-policy-test", *config.arguments()])
    args = arguments.parse_args()
    assert args.use_rollout_logprobs
    assert not args.normalize_advantages and not args.use_kl_loss and not args.use_rollout_routing_replay
    assert args.olmo_core.router_aux_loss_weight == args.olmo_core.router_z_loss_weight == 0
    assert args.clip_grad == 0.005 and args.lr == 1e-4 and args.eps_clip_high == 0.28


@pytest.mark.parametrize(
    "field", ["loss_masks", "advantages", "rollout_log_probs", "response_lengths", "unconcat_tokens"]
)
def test_rejects_changed_consumed_objective_inputs(field):
    value = fixture()
    rollout = capture.rollout_from_fixture(value, "cpu")
    capture.inject_advantages(value, rollout)
    batch = data.sample_batches(rollout, 128)[0]
    assert capture.assert_objective_inputs(value, batch)["id"] == "0"
    if field == "response_lengths":
        batch[field] = [1]
    else:
        batch[field] = [batch[field][0].clone()]
        batch[field][0][0] += 1
    with pytest.raises(ValueError, match="Objective"):
        capture.assert_objective_inputs(value, batch)


@pytest.mark.parametrize(
    "arm,lb,z", [("policy", 0.0, 0.0), ("lb", 0.01, 0.0), ("z", 0.0, 1e-5), ("combined", 0.01, 1e-5)]
)
def test_auxiliary_arm_uses_explicit_coefficients_and_immutable_policy_inputs(tmp_path, arm, lb, z):
    value = fixture()
    value["objective"] = "auxiliary_contract"
    value["auxiliary"] = {"arm": arm, "lb": lb, "z": z}
    if arm in ("lb", "z"):
        for sample in value["samples"]:
            sample["advantages"] = [0.0] * sample["response_length"]
    schema.validate_fixture(value)
    config = capture.configuration(tmp_path, tmp_path / "output", value)
    assert config.core.router_aux_loss_weight == lb
    assert config.core.router_z_loss_weight == z
    assert config.miles["use_rollout_logprobs"]
    rollout = capture.rollout_from_fixture(value, "cpu")
    capture.inject_advantages(value, rollout)
    assert [x.tolist() for x in rollout["advantages"]] == [x["advantages"] for x in value["samples"]]
    assert [x.tolist() for x in rollout["rollout_log_probs"]] == [x["old_log_probs"] for x in value["samples"]]
