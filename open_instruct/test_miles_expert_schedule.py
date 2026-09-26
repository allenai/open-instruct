"""Expert scheduling preserves replay, block membership and the executed pack schedule."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from open_instruct.miles.configuration.config import EXPERT_SCHEDULE, CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.training import data
from open_instruct.miles.training import expert_schedule as schedule


def configuration(**changes):
    core = dict(
        sequence_packing=True,
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


def sample_groups(count=16, sample_type=SimpleNamespace):
    samples = []
    for i in range(count):
        hot = [0, 1] if i % 4 < 2 else [2, 3]
        routes = np.broadcast_to(np.array(hot, dtype=np.int32), (5, 2, 2)).copy()
        routes[:, 0] = -1  # Dense layer: never dispatched or counted.
        samples.append(
            sample_type(
                index=i,
                group_index=i // 4,
                rollout_id=None,
                tokens=[i % 11] * 6,
                response_length=2,
                reward=float(i % 3),
                rollout_routed_experts=routes,
                weight_versions=[str(i // 16)],
                rollout_log_probs=[-1.0, -1.0],
                loss_mask=[1, 1],
                remove_sample=False,
            )
        )
    return [samples[i : i + 4] for i in range(0, count, 4)]


def hook_args(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="olmo3moe",
                n_routed_experts=4,
                num_hidden_layers=2,
                num_experts_per_tok=2,
                dense_layers_indices=[0],
            )
        )
    )
    config = configuration()
    return SimpleNamespace(
        **{**config.miles, "hf_checkpoint": str(tmp_path)}, actor_num_nodes=1, olmo_core=config.core
    )


def histograms(samples):
    return [
        schedule.destination_histogram(
            s.rollout_routed_experts, len(s.tokens), num_experts=4, ep_degree=2, num_layers=2, top_k=2, layers=(1,)
        )
        for s in samples
    ]


def test_plan_improves_split_routing_and_preserves_every_sample():
    samples = sum(sample_groups(), [])
    lengths = [len(s.tokens) for s in samples]
    params = dict(world=4, ep_degree=2, max_tokens=12)
    order, before, after = schedule.plan_order(lengths, histograms(samples), **params)
    assert sorted(order) == list(range(16))
    assert order == schedule.plan_order(lengths, histograms(samples), **params)[0]
    assert after["skew_mean"] < before["skew_mean"]
    assert after["critical_work_proxy"] < before["critical_work_proxy"]
    membership = schedule.schedule(order, lengths, world=4, max_tokens=12)
    assert all(sum(lengths[i] for i in pack) <= 12 for rank in membership for pack in rank)
    assert all(sum(map(len, rank)) == 4 for rank in membership)
    json.dumps(after, allow_nan=False)


def test_actual_boundaries_and_world_wide_equalization():
    # Near-equal row lengths, equal pack counts, different pack membership.
    lengths = [56, 56, 45, 44, 44, 44]
    ranks = schedule.schedule(list(range(6)), lengths, world=2, max_tokens=100)
    assert ranks == [[[0], [2, 4]], [[1, 3], [5]]]
    # Equalization must include other EP groups, not just each group's ranks.
    lengths = [2, 2, 9, 9] * 4
    ranks = schedule.schedule(list(range(16)), lengths, world=4, max_tokens=12)
    assert [len(rank) for rank in ranks] == [4, 4, 4, 4]


def test_histograms_match_trainer_replay_including_each_tail():
    samples = sum(sample_groups(), [])[:3]
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleDict({"1": torch.nn.Module()})
    model.blocks["1"].routed_experts_router = torch.nn.Identity()
    batch = dict(
        tokens=torch.zeros(1, 18),
        total_lengths=[6] * 3,
        rollout_routed_experts=[torch.from_numpy(s.rollout_routed_experts) for s in samples],
    )
    replay = data.router_routes(model, batch)["blocks.1.routed_experts_router"]
    expected = torch.bincount(replay.flatten() // 2, minlength=2).numpy()
    np.testing.assert_array_equal(np.stack(histograms(samples)).sum(axis=0)[0], expected)
    for sample in samples:
        cpu = histograms([sample])[0]
        sample.rollout_routed_experts = torch.from_numpy(sample.rollout_routed_experts)
        np.testing.assert_array_equal(histograms([sample])[0], cpu)


@pytest.mark.parametrize("malformation", ["missing", "length", "dtype", "negative", "range"])
def test_invalid_replay_fails(malformation):
    sample = sample_groups()[0][0]
    routes = sample.rollout_routed_experts
    if malformation == "missing":
        sample.rollout_routed_experts = None
    elif malformation == "length":
        sample.rollout_routed_experts = routes[:-1]
    elif malformation == "dtype":
        sample.rollout_routed_experts = routes.astype(float)
    else:
        routes[0, 1, 0] = -1 if malformation == "negative" else 4
    with pytest.raises(ValueError):
        histograms([sample])


def test_hook_preserves_blocks_metadata_and_trimmed_tail(tmp_path):
    args = hook_args(tmp_path)
    groups = sample_groups(36)
    original = sum(groups, [])
    metadata = {
        s.index: (s.group_index, s.reward, list(s.weight_versions), s.rollout_routed_experts.copy()) for s in original
    }
    schedule.reorder_samples(args, groups)
    reordered = sum(groups, [])
    assert [s.index for s in reordered] != [s.index for s in original]
    for start in [0, 16]:
        assert {s.index for s in reordered[start : start + 16]} == set(range(start, start + 16))
    assert all(a is b for a, b in zip(reordered[32:], original[32:], strict=True))
    for s in reordered:
        group, reward, versions, routes = metadata[s.index]
        assert (s.group_index, s.reward, s.weight_versions) == (group, reward, versions)
        np.testing.assert_array_equal(s.rollout_routed_experts, routes)


@pytest.mark.parametrize(
    "field,value", [("group_index", None), ("index", None), ("rollout_id", 3), ("rollout_routed_experts", None)]
)
def test_hook_rejects_unsafe_inputs_before_mutating(tmp_path, field, value):
    args = hook_args(tmp_path)
    groups = sample_groups()
    original = [s.index for s in sum(groups, [])]
    setattr(groups[-1][-1], field, value)
    with pytest.raises(ValueError):
        schedule.reorder_samples(args, groups)
    if field != "index":
        assert [s.index for s in sum(groups, [])] == original


def test_identity_fallback_never_regresses_selected_load_metrics():
    rng = np.random.default_rng(71)
    for _ in range(40):
        lengths = rng.integers(2, 13, size=16).tolist()
        counts = np.stack([rng.multinomial(n * 2, [0.2, 0.8], size=3) for n in lengths])
        order, before, after = schedule.plan_order(lengths, counts, world=4, ep_degree=2, max_tokens=12)
        for key in before:
            assert after[key] <= before[key]
        assert sorted(order) == list(range(16))
    identical = np.ones((16, 2, 2), dtype=np.int64)
    assert schedule.plan_order([6] * 16, identical, world=4, ep_degree=2, max_tokens=12)[0] == list(range(16))


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
