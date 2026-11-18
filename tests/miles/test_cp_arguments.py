"""Exercise native argument parsing before reserving GPUs for CP qualification."""

import sys

import pytest
from miles.utils import arguments as miles_arguments

from open_instruct.miles.configuration import arguments
from open_instruct.miles.configuration.config import CoreConfig, RunConfig


@pytest.mark.parametrize("cp", [1, 2])
def test_native_core_parser_preserves_replay_diagnostics_and_cp_layout(monkeypatch, cp):
    config = RunConfig(
        CoreConfig(
            attention_backend="flash_4",
            expert_parallel_size=2,
            sequence_packing=True,
            router_aux_loss_weight=0,
            router_z_loss_weight=0,
            replay_diagnostics=True,
        ),
        dict(
            hf_checkpoint="fixture",
            actor_num_gpus_per_node=2,
            global_batch_size=4,
            rollout_batch_size=1,
            n_samples_per_prompt=4,
            context_parallel_size=cp,
            use_rollout_routing_replay=True,
            use_miles_router=True,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["cp-arguments", *config.arguments()])
    args = arguments.load_core_args(miles_arguments.get_miles_extra_args_provider())
    assert args.context_parallel_size == cp
    assert args.use_rollout_routing_replay
    assert args.olmo_core.replay_diagnostics
    assert args.allgather_cp == (cp > 1)
