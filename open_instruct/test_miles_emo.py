"""EMO preparation is explicit, immutable and included in prepared/resume identity."""

import copy
import json
from pathlib import Path

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.execution import emo, workflow
from open_instruct.test_miles_run_spec import document
from open_instruct.test_miles_workflow import Spec


def metadata():
    return {
        "model_type": "olmo3moe",
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "emo_min_document_expert_pool": 2,
        "emo_max_document_expert_pool": 4,
        "emo_eval_document_expert_pool": 3,
        "emo_eos_token_id": 0,
        "emo_source_config": None,
    }


def test_full_pool_preparation_preserves_source_and_weight_links(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weight fixture")
    spec = Spec(tmp_path / "run", source)
    source = Path(spec.model["source"])
    original = metadata()
    (source / "config.json").write_text(json.dumps(original))
    before = workflow.model_identity(source)
    spec.model["emo_routing_mode"] = "full_pool"
    target = Path(workflow.prepare_model(spec))
    resolved = json.loads((target / "config.json").read_text())
    assert resolved["emo_routing_mode"] == "full_pool"
    assert resolved["emo_eval_document_expert_pool"] == 8
    assert resolved["emo_source_config"] == {key: original[key] for key in emo.FIELDS}
    assert (target / "model.safetensors").resolve() == source / "model.safetensors"
    assert workflow.model_identity(source) == before
    assert workflow.prepare_model(spec) == str(target)
    del spec.model["emo_routing_mode"]
    with pytest.raises(InputError, match="identity changed"):
        workflow.prepare_model(spec)


@pytest.mark.parametrize("evaluation", [None, 3, 8])
def test_only_explicit_resolution_changes_source_execution(evaluation):
    config = metadata() | {"emo_eval_document_expert_pool": evaluation}
    original = copy.deepcopy(config)
    with pytest.raises(InputError, match="requires model.emo_routing_mode"):
        emo.resolve_hf(config, None)
    resolved = emo.resolve_hf(config, "full_pool")
    assert resolved["emo_eval_document_expert_pool"] == 8
    assert resolved["emo_source_config"]["emo_eval_document_expert_pool"] == evaluation
    assert emo.resolve_hf(resolved, None) == resolved
    assert emo.resolve_hf(resolved, "full_pool") == resolved
    assert config == original


@pytest.mark.parametrize(
    "changes",
    [
        {"emo_min_document_expert_pool": None},
        {"emo_max_document_expert_pool": True},
        {"emo_max_document_expert_pool": 1},
        {"emo_eval_document_expert_pool": 9},
        {"emo_eos_token_id": -1},
        {"num_experts_per_tok": 0},
        {"gating_function": "topk_softmax"},
        {"normalize_expert_weights": None},
    ],
)
def test_malformed_or_unsupported_source_rejected(changes):
    with pytest.raises(InputError, match="EMO"):
        emo.resolve_hf(metadata() | changes, "full_pool")


def test_ancestry_only_export_keeps_ordinary_routing():
    config = {"model_type": "olmo3moe", **dict.fromkeys(emo.FIELDS)}
    assert emo.resolve_hf(config, None) == config


def test_native_named_blocks_and_overrides_preserve_pretraining_settings():
    router = {
        "num_experts": 8,
        "top_k": 2,
        "emo": {key.removeprefix("emo_"): value for key, value in metadata().items() if key in emo.FIELDS},
    }
    block = {"routed_experts_router": router}
    original = {"block": {"attention": block}, "block_overrides": {3: copy.deepcopy(block)}}
    before = copy.deepcopy(original)
    resolved = emo.resolve_native(original, "full_pool")
    for converted in (resolved["block"]["attention"], resolved["block_overrides"][3]):
        converted_router = converted["routed_experts_router"]["emo"]
        assert converted_router["full_pool"] is True
        assert converted_router["eval_document_expert_pool"] == 8
        assert converted_router["source_config"]["emo_eval_document_expert_pool"] == 3
    assert original == before
    with pytest.raises(InputError, match="requires model.emo_routing_mode"):
        emo.resolve_native(original, None)


def test_plan_reports_execution_and_rejects_unqualified_training(tmp_path):
    payload = document(tmp_path)
    payload["model"]["emo_routing_mode"] = "full_pool"
    payload["trainer"] = {"use_rollout_routing_replay": True, "router_aux_loss_weight": 0, "router_z_loss_weight": 0}
    run = RunSpec.from_dict(payload, config_path=tmp_path / "emo.toml")
    assert run.plan()["model"]["emo_routing_mode"] == "full_pool"
    for key, value in (
        ("use_rollout_routing_replay", False),
        ("router_aux_loss_weight", 0.01),
        ("router_z_loss_weight", 1e-5),
        ("router_aux_count_source", "current"),
    ):
        changed = copy.deepcopy(payload)
        changed["trainer"][key] = value
        with pytest.raises(InputError, match="EMO"):
            RunSpec.from_dict(changed, config_path=tmp_path / "emo.toml").compile()
