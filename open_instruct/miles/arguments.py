"""Parse shared MILES optimizer settings and explicit Core backend options."""

import json
import sys

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
    # Validate the same backend constraints for direct native CLI callers, including
    # options that used to be accepted and silently ignored by this adapter.
    for key in (
        "rollout_batch_size",
        "n_samples_per_prompt",
        "offload",
        "fsdp_cpu_offload",
        "stream_optimizer_state_to_disk",
        "check_weight_update_selector",
        "ref_update_interval",
        "colocate",
        "offload_rollout",
        "update_weights_interval",
        "use_routing_replay",
        "use_rollout_routing_replay",
        "use_miles_router",
        "optimizer",
        "fp16",
        "keep_fp32_master",
        "async_save",
        "no_save_optim",
        "reset_optimizer_states",
        "override_lr_scheduler",
        "use_checkpoint_lr_scheduler",
        "compute_advantages_and_returns",
        "skip_actor_forward_only",
        "keep_old_actor",
        "dp_replicate_size",
        "deterministic_mode",
        "lora_train_only",
        "lora_rank",
        "save_hf",
        "debug_disable_optimizer",
        "debug_skip_weight_update",
        "debug_rollout_only",
        "update_weight_transfer_mode",
        "colocated_weight_update_pipeline_depth",
    ):
        value = getattr(args, key, None)
        if value is not None:
            fields[key] = value
    supplied = {token.partition("=")[0] for token in sys.argv[1:] if token.startswith("--")}
    for key in (
        "max_weight_staleness",
        "data_source_path",
        "custom_async_data_buffer_path",
        "gradient_checkpointing",
        "attn_implementation",
        "warmup_ratio",
        "max_tokens_per_gpu",
    ):
        if "--" + key.replace("_", "-") in supplied:
            fields[key] = getattr(args, key)
    RunConfig(core, fields).validate()
    args.olmo_core = core
    args.compress_ratios = None
    if args.fully_async:
        args.max_weight_staleness = core.max_policy_lag
        args.custom_async_data_buffer_path = "open_instruct.miles.async_buffer.HomogeneousPolicyDataBuffer"
    args.data_source_path = "open_instruct.miles.data_source.DashboardDrainingRolloutDataSource"
    return args
