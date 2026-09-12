"""Packing keeps sample identity, boundaries and optimizer membership intact."""

import pytest
import torch
from torch import nn

from open_instruct.miles import data, packing
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.run_spec import RunSpec


def fixture():
    lengths = [5, 8, 6, 7]
    return {
        "tokens": [torch.arange(n) + 20 * i for i, n in enumerate(lengths)],
        "total_lengths": lengths,
        "response_lengths": [2] * 4,
        "loss_masks": [torch.tensor([1, 0])] * 4,
        "weight_versions": [[str(i)] for i in range(4)],
        "rewards": [float(i) for i in range(4)],
        "rollout_routed_experts": [torch.full((n - 1, 1, 2), 2, dtype=torch.int32) for n in lengths],
    }


def test_plan_and_split_preserve_order_budget_and_membership():
    plans = [packing.plan([5, 8, 6, 7], 16), packing.plan([15, 15, 2, 2], 16)]
    assert list(map(len, plans)) == [2, 3]
    for plan in plans:
        equal = packing.equalize(plan, 3)
        assert len(equal) == 3 and all(equal)
        assert sum(equal, []) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        packing.plan([17], 16)
    with pytest.raises(ValueError):
        packing.equalize(plans[0], 5)


def test_packed_metadata_and_replay_have_one_tail_per_sample():
    raw = fixture()
    samples = data.sample_batches(raw, 16)
    batch = packing.combine(samples, [0, 1])
    assert torch.equal(batch["tokens"], torch.cat(raw["tokens"][:2])[None])
    assert batch["doc_lens"].tolist() == [[5, 8]] and batch["max_doc_lens"] == [8]
    assert batch["weight_versions"] == [["0"], ["1"]] and batch["rewards"] == [0.0, 1.0]
    assert batch["total_lengths"] == [5, 8] and batch["max_seq_lens"] is None
    model = nn.Module()
    model.blocks = nn.ModuleDict({"0": nn.Module()})
    model.blocks["0"].routed_experts_router = nn.Identity()
    routes = data.router_routes(model, batch)["blocks.0.routed_experts_router"]
    assert routes.shape == (1, 13, 2)
    assert routes[0, 4].tolist() == routes[0, 12].tolist() == [0, 1]
    assert routes[0, 5].tolist() == [2, 2]
    raw["rollout_routed_experts"][1] = torch.zeros(8, 1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="per sample"):
        data.router_routes(model, packing.combine(data.sample_batches(raw, 16), [0, 1]))


def test_options_compile_to_concatenated_loss_layout():
    config = RunConfig(
        CoreConfig(sequence_packing=True, max_sequence_length=16), {"hf_checkpoint": "/hf", "global_batch_size": 4}
    )
    argv = config.arguments()
    assert argv[argv.index("--qkv-format") + 1] == "thd"
    with pytest.raises(ValueError, match="requires qkv_format"):
        RunConfig(config.core, {**config.miles, "qkv_format": "bshd"}).validate()
    with pytest.raises(ValueError, match="requires sequence_packing"):
        RunConfig(CoreConfig(packing_max_tokens=8192), config.miles).validate()
    with pytest.raises(ValueError, match="must cover"):
        RunConfig(
            CoreConfig(sequence_packing=True, packing_max_tokens=8, max_sequence_length=16), config.miles
        ).validate()


def test_researcher_trainer_section_accepts_packing(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text("""schema_version = 1
name = "packing"
[model]
source = "/model"
format = "hf"
[output]
root = "/output"
[data]
prompt_data = "/data/train.jsonl"
reward_config = "/data/rewards.json"
[trainer]
sequence_packing = true
packing_max_tokens = 8192
""")
    spec = RunSpec.load(path)
    # plan is intentionally CPU safe; the compiled core fields preserve the knobs.
    core = spec.compile().core
    assert core.sequence_packing and core.packing_max_tokens == 8192
