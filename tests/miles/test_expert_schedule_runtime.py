"""Pinned MILES producer conversion must preserve rewards and advantages by identity."""

import sys
from copy import deepcopy

import pytest
import torch
from miles.backends.training_utils.loss_hub import advantages
from miles.ray.rollout import rollout_data_conversion, train_data_conversion
from miles.utils import arguments
from miles.utils.types import Sample

from open_instruct.miles import arguments as core_arguments
from open_instruct.miles import expert_schedule
from open_instruct.miles.config import EXPERT_SCHEDULE
from open_instruct.test_miles_expert_schedule import configuration, hook_args, sample_groups


def test_hook_through_real_reward_normalization_and_partition(tmp_path):
    args = hook_args(tmp_path)
    args.reward_key = None
    args.advantage_estimator = "grpo"
    args.rewards_normalization = True
    args.grpo_std_normalization = True
    args.use_dynamic_global_batch_size = False
    args.disable_rollout_trim_samples = False
    args.balance_data = False
    args.multi_lora = False
    original = sample_groups(36, Sample)
    planned = deepcopy(original)
    expert_schedule.reorder_samples(args, planned)
    results = []
    for groups in [original, planned]:
        flat, metadata = rollout_data_conversion.postprocess_rollout_data(args, groups, {"dp_size": 4})
        assert len(flat) == 32
        converted = train_data_conversion.convert_samples_to_train_data(args, flat, metadata, None, None)
        shards = train_data_conversion.split_train_data_by_dp_raw(args, converted, dp_size=4)
        by_id = {}
        for shard in shards:
            shard = train_data_conversion.process_rollout_data_shard(args, shard)
            masks = [torch.tensor(mask) for mask in shard["loss_masks"]]
            adv, _ = advantages.compute_advantages(
                args,
                kl=[torch.zeros_like(m, dtype=torch.float32) for m in masks],
                rewards=shard["rewards"],
                log_probs=None,
                loss_masks=masks,
                total_lengths=shard["total_lengths"],
                response_lengths=shard["response_lengths"],
            )
            for i, index in enumerate(shard["sample_indices"]):
                by_id[index] = (
                    shard["rewards"][i],
                    adv[i].tolist(),
                    shard["loss_masks"][i],
                    shard["weight_versions"][i],
                )
        results.append(by_id)
    assert results[0] == results[1]
    assert any(values[0] != 0 for values in results[0].values())


@pytest.mark.parametrize("native", [False, True])
def test_native_parser_injects_managed_schedule(monkeypatch, native):
    config = configuration()
    argv = ["test", *config.arguments()]
    if native:
        index = argv.index("--rollout-sample-filter-path")
        del argv[index : index + 2]
    monkeypatch.setattr(sys, "argv", argv)
    parsed = core_arguments.load_core_args(arguments.get_miles_extra_args_provider())
    assert parsed.olmo_core.expert_balanced_packing
    assert parsed.rollout_sample_filter_path == EXPERT_SCHEDULE
