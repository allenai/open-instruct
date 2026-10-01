"""CPU-only configuration checks for the MILES expert-packing facility."""

import dataclasses
import json

import pytest

from open_instruct.miles.configuration.config import EXPERT_SCHEDULE, CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec


def configuration(**changes):
    core = dict(
        sequence_packing=True,
        attention_backend="flash_4",
        expert_balanced_packing=True,
        expert_parallel_size=2,
        router_aux_loss_weight=0,
        max_sequence_length=12,
    )
    core.update(changes)
    return RunConfig(
        CoreConfig(**core),
        dict(
            hf_checkpoint="/hf",
            global_batch_size=16,
            rollout_batch_size=4,
            n_samples_per_prompt=4,
            actor_num_gpus_per_node=4,
            use_rollout_routing_replay=True,
            use_miles_router=True,
        ),
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"expert_parallel_size": 1}, "world >"),
        ({"expert_parallel_size": 4}, "world >"),
        ({"sequence_packing": False}, "sequence_packing"),
        ({"router_aux_loss_weight": 0.01}, "router_aux_loss_weight"),
        ({"model_config": "/custom"}, "HF configuration"),
    ],
)
def test_core_guards(changes, match):
    with pytest.raises(ValueError, match=match):
        configuration(**changes).validate()


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"use_rollout_routing_replay": False}, "use_rollout_routing_replay"),
        ({"use_dynamic_global_batch_size": True}, "use_dynamic_global_batch_size"),
        ({"balance_data": True}, "balance_data"),
        ({"custom_reward_post_process_path": "custom.fn"}, "custom_reward"),
        ({"custom_convert_samples_to_train_data_path": "custom.fn"}, "custom_convert"),
        ({"rollout_sample_filter_path": "custom.fn"}, "conflicts"),
    ],
)
def test_miles_guards(changes, match):
    config = configuration()
    with pytest.raises(ValueError, match=match):
        RunConfig(config.core, {**config.miles, **changes}).validate()


def test_enabled_hook_and_disabled_native_arguments():
    config = configuration()
    argv = config.arguments()
    assert argv[argv.index("--rollout-sample-filter-path") + 1] == EXPERT_SCHEDULE
    disabled = RunConfig(dataclasses.replace(config.core, expert_balanced_packing=False), config.miles)
    argv = disabled.arguments()
    assert "--rollout-sample-filter-path" not in argv
    encoded = json.loads(argv[argv.index("--olmo-core-config") + 1])
    expected = dataclasses.asdict(disabled.core)
    for name in (
        "expert_balanced_packing",
        "expert_balance_layer_stride",
        "expert_balance_search_proposals",
        "expert_balance_search_seconds",
    ):
        del expected[name]
    assert encoded == expected
    with pytest.raises(ValueError, match="requires core.expert"):
        RunConfig(disabled.core, {**config.miles, "rollout_sample_filter_path": EXPERT_SCHEDULE}).validate()


def test_structured_trainer_fields(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("""schema_version = 1
name = "expert-packing"
[model]
source = "/model"
format = "hf"
[output]
root = "/output"
[data]
prompt_data = "/data/train.jsonl"
reward_config = "/data/rewards.json"
[trainer]
expert_balanced_packing = true
expert_balance_layer_stride = 2
expert_parallel_size = 2
sequence_packing = true
trainer_flash_attention_version = 4
router_aux_loss_weight = 0.0
trainer_num_nodes = 1
gpus = 4
[miles]
use_rollout_routing_replay = true
use_miles_router = true
""")
    # Parsing the structured Core fields must not need a CUDA runtime.
    spec = RunSpec.load(path)
    assert spec.compile().core.expert_balanced_packing
    assert spec.compile().core.expert_balance_layer_stride == 2


@pytest.mark.parametrize("mode", ["barrier", "engine_drain", "refresh"])
def test_managed_hook_works_in_all_publication_modes(mode):
    config = configuration(publication_mode=mode, max_policy_lag=2)
    config = RunConfig(
        config.core,
        {
            **config.miles,
            "fully_async": True,
            "use_tis": True,
            "sglang_cuda_graph_backend_decode": "disabled",
            "sglang_cuda_graph_backend_prefill": "disabled",
        },
    )
    assert config.plan()["miles"]["rollout_sample_filter_path"] == EXPERT_SCHEDULE
    with pytest.raises(ValueError, match="conflicts"):
        RunConfig(config.core, {**config.miles, "rollout_sample_filter_path": "custom.fn"}).validate()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_layer_stride_requires_positive_integer(value):
    with pytest.raises(ValueError, match="positive integer"):
        configuration(expert_balance_layer_stride=value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("expert_balance_search_seconds", -1),
        ("expert_balance_search_seconds", float("nan")),
        ("expert_balance_search_proposals", -1),
        ("expert_balance_search_proposals", 1.5),
    ],
)
def test_search_limits_are_validated(name, value):
    with pytest.raises(ValueError):
        CoreConfig(**{name: value}).validate()
