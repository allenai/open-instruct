"""CPU-only checks of researcher sections, workflow boundaries and runtime translation."""

import copy
import json
from pathlib import Path

import pytest

from open_instruct.miles.run_spec import RunSpec


def document(tmp_path):
    return {
        "schema_version": 1,
        "name": "researcher-trial",
        "model": {"source": "model", "format": "hf"},
        "output": {"root": str(tmp_path / "run")},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 16}]},
    }


def spec(tmp_path, **sections):
    return RunSpec.from_dict(document(tmp_path) | sections, config_path=tmp_path / "run.toml")


def test_async_defaults_are_real_tis_and_eight_by_eight(tmp_path):
    config = spec(tmp_path, **{"async": {"fully_async": True}}).compile()
    assert config.miles["use_tis"] is True
    assert config.miles["use_rollout_logprobs"] is False
    assert config.miles["async_data_buffer_capacity_factor"] == 2
    assert config.miles["colocate"] is False
    assert config.miles["rollout_batch_size"] == config.miles["n_samples_per_prompt"] == 8
    assert config.miles["global_batch_size"] == 64
    assert config.core.max_policy_lag == 1
    assert "--use-tis" in config.arguments()
    assert "--use-rollout-logprobs" not in config.arguments()


def test_colocation_and_launch_defaults(tmp_path):
    run = spec(tmp_path)
    config = run.compile()
    assert config.miles["colocate"] is True
    assert config.miles["offload_train"] is False
    assert config.miles["rollout_num_gpus"] == config.miles["actor_num_gpus_per_node"] == 2
    assert run.launch["priority"] == "urgent"
    assert run.launch["min_runtime"] == "1h"
    assert run.launch["workspace"] == "ai2/open-instruct-dev"
    assert run.plan()["runtime_validated"] is False
    assert config.miles["use_tis"] is False


def test_section_translation_and_consistent_direct_overrides(tmp_path):
    run = spec(
        tmp_path,
        trainer={"gpus": 2, "expert_parallel_size": 2, "trainer_flash_attention_version": 4},
        inference={"placement_mode": "disaggregated", "gpus": 1, "max_context_length": 6144},
        optimizer={"learning_rate": 2e-6},
        training={"num_rollouts": 4, "collect_dashboard": True},
        tracking={"wandb_mode": "offline"},
        runtime={"row_specialization": "dynamic"},
        core={"row_specialization": "dynamic"},
        miles={"lr": 2e-6, "sglang_json_model_override_args": {"max_position_embeddings": 6144}},
    )
    config = run.compile()
    assert config.miles["lr"] == 2e-6
    assert config.miles["num_rollout"] == 4
    assert config.miles["use_miles_dashboard"] is True
    assert config.miles["use_wandb"] is True
    assert config.miles["num_gpus_per_node"] == 3
    assert config.core.attention_backend == "flash_4"
    assert config.core.max_sequence_length == config.miles["sglang_context_length"] == 6144


@pytest.mark.parametrize(
    "sections",
    [
        {"optimizer": {"learning_rate": 1e-6}, "miles": {"lr": 2e-6}},
        {"runtime": {"row_specialization": "dynamic"}, "core": {"row_specialization": "static"}},
        {"inference": {"placement_mode": "colocated"}, "miles": {"colocate": False}},
        {"inference": {"radix_cache": True}, "miles": {"sglang_disable_radix_cache": True}},
        {"training": {"save_checkpoints": False, "save_interval": 4}},
        {"inference": {"max_context_length": 6144}, "core": {"max_sequence_length": 8192}},
    ],
)
def test_duplicate_semantic_options_conflict(tmp_path, sections):
    with pytest.raises(ValueError, match="[Cc]onflict"):
        spec(tmp_path, **sections)


def test_prepared_payload_changes_only_artifact_bindings(tmp_path):
    run = spec(tmp_path)
    prepared = {
        "hf_checkpoint": "/prepared/hf",
        "prompt_data": "/prepared/train.jsonl",
        "eval_prompt_data": ["gsm8k", "/prepared/heldout.jsonl"],
        "reward_config": "/prepared/rewards.json",
        "manifest": "/prepared/manifest.json",
    }
    config = run.compile(prepared)
    assert config.miles["hf_checkpoint"] == prepared["hf_checkpoint"]
    assert config.miles["prompt_data"] == prepared["prompt_data"]
    assert config.miles["eval_prompt_data"] == prepared["eval_prompt_data"]
    assert config.core.reward_config == prepared["reward_config"]
    assert config.miles["global_batch_size"] == 64
    assert run.compile().miles["prompt_data"] != prepared["prompt_data"]
    assert config.miles["save"].endswith("run/checkpoints")


def test_serialization_keeps_paths_and_does_not_mutate_input(tmp_path):
    payload = document(tmp_path)
    payload["miles"] = {"load": "previous", "wandb_dir": "wandb"}
    original = copy.deepcopy(payload)
    run = RunSpec.from_dict(payload, config_path=tmp_path / "config.toml")
    serialized = json.loads(json.dumps(run.to_dict()))
    restored = RunSpec.from_dict(serialized, config_path="/unrelated/job/run.toml")
    assert run.compile().arguments() == restored.compile().arguments()
    assert payload == original
    serialized["model"]["source"] = "other"
    assert run.model["source"] == str(tmp_path / "model")


def test_repeatable_toml_overrides_before_compilation(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text(
        'schema_version=1\nname="test"\n[model]\nsource="model"\n[output]\nroot="run"\n'
        '[[data.tasks]]\ntask="gsm8k"\ntrain_count=8\neval_count=4\n'
    )
    run = RunSpec.load(
        path, ["async.fully_async=true", "optimizer.learning_rate=2e-6", "optimizer.learning_rate=3e-6"]
    )
    assert run.compile().miles["lr"] == 3e-6
    assert run.compile().miles["use_tis"] is True
    with pytest.raises(ValueError, match="quote strings"):
        RunSpec.load(path, ["model.source=unquoted"])


@pytest.mark.parametrize(
    "sections,match",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"model": {"source": "model", "format": "megatron"}}, "Megatron"),
        ({"model": {"source": "model", "format": "olmo_core"}}, "hf_template"),
        ({"conversion_validation": {"min_logit_cosine": 0.99}}, "parity"),
        ({"conversion": {"output": "megatron"}}, "conversion"),
        ({"trainer": {"trainer_backend": "optimized"}}, "Megatron"),
        ({"trainer": {"recompute_mode": "selective"}}, "selective"),
        ({"training": {"async_save": True}}, "async_save"),
        ({"training": {"save_retain_interval": 20}}, "retention"),
        ({"runtime": {"fla_prewarm": True}}, "prewarming"),
        ({"async": {"off_policy_correction": "icepop"}}, "qualified"),
        ({"async": {"policy_drift_action": "warn"}}, "warn"),
        ({"data": {"recipe": "not-ported"}}, "recipes"),
        ({"data": {"tasks": [{"task": "bad", "train_count": 8}]}}, "Unsupported task"),
        ({"data": {"rl_manifest": "manifest.json", "tasks": []}}, "exactly one"),
        ({"data": {"prompt_data": "train.jsonl"}}, "reward_config"),
        ({"inference": {"max_response_length": 6144, "max_context_length": 6144}}, "smaller"),
        ({"trainer": {"gpus": True}}, "expects int"),
        ({"launch": {"priority": "urgent", "auto_resume": "true"}}, "boolean"),
        ({"miles": {"unknown_option": 1}}, "Unknown MILES option"),
    ],
)
def test_unsupported_or_ambiguous_workflows_fail_before_execution(tmp_path, sections, match):
    with pytest.raises(ValueError, match=match):
        spec(tmp_path, **sections)


def test_native_source_template_and_final_export_are_explicit(tmp_path):
    run = spec(
        tmp_path,
        model={"source": "native/step100", "format": "olmo_core", "hf_template": "template"},
        conversion={"hf_output": "converted"},
        output={"root": str(tmp_path / "run"), "export_hf": True},
    )
    assert run.compile().miles["hf_checkpoint"] == str(tmp_path / "converted")
    assert run.model["hf_template"] == str(tmp_path / "template")
    assert run.plan()["stages"][-1] == "export_hf"
    assert "save_hf" not in run.compile().miles


def test_disabled_checkpoint_cadence_and_empty_heldout(tmp_path):
    run = spec(tmp_path, training={"save_checkpoints": False}, data={"tasks": [{"task": "gsm8k", "train_count": 8}]})
    config = run.compile({"eval_prompt_data": []})
    assert "save_interval" not in config.miles
    assert "eval_interval" not in config.miles
    assert "save" in config.miles  # Root remains available to cursor/diagnostics.


def test_model_context_and_multiple_optimizer_steps_need_consistent_contracts(tmp_path):
    with pytest.raises(ValueError, match="max_policy_lag"):
        spec(tmp_path, inference={"global_batch_size": 32})
    config = spec(tmp_path, inference={"global_batch_size": 32}, **{"async": {"fully_async": True}}).compile()
    assert config.plan()["shape"]["optimizer_steps_per_collection"] == 2
    with pytest.raises(ValueError, match="cannot both"):
        spec(tmp_path, **{"async": {"fully_async": True}, "miles": {"use_tis": True, "use_rollout_logprobs": True}})


def test_explicit_prepared_input_and_tracking_disable(tmp_path):
    run = spec(
        tmp_path,
        data={"prompt_data": "train.jsonl", "reward_config": "verifiers.json", "eval_prompt_data": []},
        tracking={"wandb_mode": "offline"},
        miles={"use_wandb": False},
    )
    config = run.compile()
    assert config.miles["prompt_data"] == str(tmp_path / "train.jsonl")
    assert config.core.reward_config == str(tmp_path / "verifiers.json")
    assert config.miles["use_wandb"] is False


def test_all_researcher_examples_compile_to_native_arguments():
    root = Path(__file__).parents[1] / "configs" / "miles"
    paths = [*sorted((root / "examples").glob("*.toml")), root / "qualification" / "workflow-async-gsm8k.toml"]
    assert len(paths) >= 5
    for path in paths:
        run = RunSpec.load(path)
        config = run.compile()
        assert config.arguments()
        assert config.miles["rollout_batch_size"] == config.miles["n_samples_per_prompt"] == 8
        if config.miles["fully_async"]:
            assert config.miles["use_tis"] is True
            assert config.miles["use_rollout_logprobs"] is False


@pytest.mark.parametrize(
    "sections",
    [
        {"inference": {"rollout_batch_size": "8"}},
        {"inference": {"samples_per_prompt": False}},
        {"inference": {"sglang_max_running_requests": "64"}},
        {"core": {"max_sequence_length": "6144"}},
    ],
)
def test_types_fail_before_topology_arithmetic(tmp_path, sections):
    with pytest.raises(ValueError):
        spec(tmp_path, **sections)


def test_model_config_path_roundtrips_from_runtime_section(tmp_path):
    run = spec(tmp_path, runtime={"model_config": "native-model.json"}, miles={"ref_load": "reference"})
    restored = RunSpec.from_dict(run.to_dict(), config_path="/elsewhere/run.toml")
    assert run.compile().arguments() == restored.compile().arguments()
    assert run.compile().core.model_config == str(tmp_path / "native-model.json")


@pytest.mark.parametrize(
    "launch,match",
    [
        ({"env": {"WANDB_API_KEY": "literal"}, "secrets": {"WANDB_API_KEY": "secret-name"}}, "overlap"),
        ({"env": {"INVALID-NAME": "x"}}, "identifiers"),
        ({"secrets": {"RAY_ADDRESS": "other-ray"}}, "runtime owns"),
        ({"env": {"PYTHONPATH": "/other/code"}}, "runtime owns"),
        (
            {"weka_mounts": [{"weka": "a", "mount_path": "/weka/a"}, {"weka": "b", "mount_path": "/weka/a/child"}]},
            "nonoverlapping",
        ),
        ({"weka_mounts": [{"weka": "a", "mount_path": "/weka/a"}, {"weka": "a", "mount_path": "/weka/b"}]}, "repeat"),
        ({"weka_mounts": [{"weka": "a", "mount_path": "/"}]}, "filesystem root"),
    ],
)
def test_mount_and_environment_ownership(tmp_path, launch, match):
    with pytest.raises(ValueError, match=match):
        spec(tmp_path, launch=launch)


def test_conversion_reference_is_not_silently_reinterpreted_as_kl_reference(tmp_path):
    with pytest.raises(ValueError, match="conversion validation"):
        spec(tmp_path, model={"source": "model", "reference_hf": "reference"})
    run = spec(tmp_path, optimizer={"kl_loss_coef": 0.01}, miles={"ref_load": "reference"})
    assert run.compile().miles["ref_load"] == str(tmp_path / "reference")
    assert run.compile().miles["use_kl_loss"] is True
