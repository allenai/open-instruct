"""Attach Core training options to the native MILES argument parser. The runtime
uses this bridge to share MILES optimizer and rollout settings while selecting the
Core backend and validating its additional controls. This module runs inside the
pinned runtime; CPU-side planning uses the configuration classes and parser snapshot.
"""

import json
import sys

from miles.backends.fsdp_utils.arguments import load_fsdp_args

from open_instruct.miles.configuration import constraints
from open_instruct.miles.configuration.config import CoreConfig, RunConfig


def load_core_args(extra_args_provider):
    def add_arguments(parser):
        parser = extra_args_provider(parser)
        parser.add_argument("--olmo-core-config", required=True)
        return parser

    args = load_fsdp_args(extra_args_provider=add_arguments)
    core = CoreConfig(**json.loads(args.olmo_core_config))
    supplied = {token.partition("=")[0] for token in sys.argv[1:] if token.startswith("--")}
    fields = constraints.native_values(args, supplied)
    config = RunConfig(core, fields)
    config.validate()
    args.dynamic_sampling_filter_path = config.resolved_miles().get("dynamic_sampling_filter_path")
    args.rollout_sample_filter_path = config.resolved_miles().get("rollout_sample_filter_path")
    args.core_records_factory = "open_instruct.miles.datasets.recording.create_recorder"
    args.olmo_core = core
    args.compress_ratios = None
    if args.fully_async:
        args.max_weight_staleness = core.max_policy_lag
        args.custom_async_data_buffer_path = (
            "miles.backends.core_utils.rollout.async_buffer.RefreshPolicyDataBuffer"
            if core.publication_mode == "refresh"
            else "miles.backends.core_utils.rollout.async_buffer.HomogeneousPolicyDataBuffer"
        )
    args.data_source_path = "open_instruct.miles.rollout.data_source.DashboardDrainingRolloutDataSource"
    return args
