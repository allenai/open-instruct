"""Parse shared MILES optimizer settings and explicit Core backend options."""

import json

from miles.backends.fsdp_utils.arguments import load_fsdp_args

from open_instruct.miles.config import CoreConfig, RunConfig


def load_core_args(extra_args_provider):
    def add_arguments(parser):
        parser = extra_args_provider(parser)
        parser.add_argument("--olmo-core-config", required=True)
        return parser

    args = load_fsdp_args(extra_args_provider=add_arguments)
    core = CoreConfig(**json.loads(args.olmo_core_config))
    fields = {
        key: value
        for key, value in vars(args).items()
        if key
        in (
            "hf_checkpoint",
            "actor_num_nodes",
            "actor_num_gpus_per_node",
            "global_batch_size",
            "micro_batch_size",
            "offload_train",
            "qkv_format",
            "use_dynamic_batch_size",
            "context_parallel_size",
            "use_critic",
            "multi_lora",
            "indep_dp",
            "use_opd",
            "fully_async",
        )
        and value is not None
    }
    RunConfig(core, fields).validate()
    args.olmo_core = core
    args.compress_ratios = None
    if args.fully_async:
        args.max_weight_staleness = core.max_policy_lag
        args.custom_async_data_buffer_path = "open_instruct.miles.async_buffer.HomogeneousPolicyDataBuffer"
    args.data_source_path = "open_instruct.miles.data_source.DashboardDrainingRolloutDataSource"
    return args
