"""Share the Core adapter's configuration constraints with validation and docs.
These CPU-only definitions distinguish invalid inputs from missing implementation
or runtime qualification. All categories remain pre-flight errors; support gaps
invite contributions rather than implying that the requested feature is inherently
wrong. Conditional checks stay in the validators, while their option names and
shared explanations live here so documentation never needs to parse Python code.
"""

import json
from dataclasses import dataclass
from enum import Enum

from open_instruct.miles.errors import InputError


class Category(str, Enum):
    INVALID = "Invalid configuration"
    NOT_IMPLEMENTED = "Not implemented"
    NOT_VALIDATED = "Not validated"


def error(message, category=Category.INVALID):
    """Explain a blocking constraint without changing its enforcement severity."""
    return InputError(Constraint(message, category).describe())


@dataclass(frozen=True)
class Constraint:
    message: str
    category: Category = Category.INVALID

    def describe(self):
        text = f"{self.category.value}: {self.message}"
        if self.category == Category.NOT_IMPLEMENTED:
            text += " An implementation with tests is welcome; please contribute support for this configuration."
        elif self.category == Category.NOT_VALIDATED:
            text += " Help validate this configuration with a focused run and share the results."
        return text

    def error(self):
        return InputError(self.describe())


# Alternatives to these native settings are not implemented by this adapter.
REQUIRED_VALUES = {
    "optimizer": "adam",
    "fp16": False,
    "keep_fp32_master": True,
    "async_save": False,
    "no_save_optim": False,
    "reset_optimizer_states": False,
    "override_lr_scheduler": False,
    "use_checkpoint_lr_scheduler": True,
    "compute_advantages_and_returns": True,
    "skip_actor_forward_only": False,
    "keep_old_actor": False,
    "dp_replicate_size": 1,
    "deterministic_mode": False,
    "lora_train_only": False,
    "debug_disable_optimizer": False,
    "debug_skip_weight_update": False,
    "update_weights_interval": 1,
    "debug_rollout_only": False,
    "update_weight_transfer_mode": "broadcast",
    "colocated_weight_update_pipeline_depth": 1,
}

REPLACEMENTS = {
    "gradient_checkpointing": "core.activation_checkpointing",
    "attn_implementation": "core.attention_backend",
    "warmup_ratio": "miles.lr_warmup_fraction",
    "max_tokens_per_gpu": "micro_batch_size=1 (Core dynamic batching is not implemented)",
}

ADAPTER_OWNED_OPTIONS = (
    "train_backend",
    "custom_config_path",
    "config",
    "olmo_core_config",
    "data_source_path",
    "custom_async_data_buffer_path",
)

TRAINER_PARALLEL_OPTIONS = ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")
UNIMPLEMENTED_OPTIONS = ("use_critic", "multi_lora", "indep_dp", "use_opd", "use_routing_replay")
OFFLOAD_OPTIONS = ("offload", "fsdp_cpu_offload", "optimizer_cpu_offload", "stream_optimizer_state_to_disk")

# These rules need predicates or Core values, so their checks remain in RunConfig.
NATIVE_CONSTRAINTS = {
    "save_hf": Constraint(
        "Core RL does not implement miles.save_hf; use miles.eval_hf_dir for snapshot evaluation or export the native checkpoint separately",
        Category.NOT_IMPLEMENTED,
    ),
    "lora_rank": Constraint(
        "Core RL does not implement LoRA; miles.lora_rank must be nonpositive", Category.NOT_IMPLEMENTED
    ),
    "max_weight_staleness": Constraint("miles.max_weight_staleness must equal core.max_policy_lag (optimizer steps)"),
    "micro_batch_size": Constraint(
        "Use micro_batch_size=1; Core accumulates unpadded response samples", Category.NOT_IMPLEMENTED
    ),
    "use_dynamic_batch_size": Constraint(
        "Use micro_batch_size=1; Core accumulates unpadded response samples", Category.NOT_IMPLEMENTED
    ),
    "offload_train": Constraint(
        "Core trainer offload has not been qualified; set offload_train=false", Category.NOT_VALIDATED
    ),
    "qkv_format": Constraint("Use thd with sequence_packing; bshd otherwise"),
    "check_weight_update_selector": Constraint(
        "Core serving checks currently require check_weight_update_selector=all", Category.NOT_IMPLEMENTED
    ),
    "ref_update_interval": Constraint("Core RL requires a fixed reference policy", Category.NOT_IMPLEMENTED),
    "fully_async": Constraint(
        "Async Core training requires resident disaggregated rollout engines; set miles.colocate=false and miles.offload_rollout=false.",
        Category.NOT_IMPLEMENTED,
    ),
    "use_rollout_routing_replay": Constraint(
        "The pinned SGLang router strips expert-ID requests; rollout replay requires use_miles_router"
    ),
}

# Structured migration errors are distinct from native backend restrictions.
UNSUPPORTED_RUN_FIELDS = {
    "megatron_checkpoint": Constraint("use model.source/model.format or miles.load for a native Core RL resume"),
    "output_dir": Constraint("use output.root"),
    "hf_checkpoint": Constraint("use model.source; preparation supplies miles.hf_checkpoint"),
    "trainer_backend": Constraint(
        "Core selects its native model backend; omit the Megatron optimized/compatibility switch"
    ),
    "recompute_modules": Constraint(
        "Core supports block activation_checkpointing, not Megatron selective modules", Category.NOT_IMPLEMENTED
    ),
    "accumulate_allreduce_grads_in_fp32": Constraint(
        "Core owns reduction precision; this Megatron switch has no Core equivalent"
    ),
    "save_retain_interval": Constraint(
        "use core.checkpoint_keep_last/core.checkpoint_keep_every for native retention"
    ),
    "save_tokens_per_expert_interval": Constraint(
        "tokens-per-expert checkpoint capture is not implemented", Category.NOT_IMPLEMENTED
    ),
    "capture_generation_samples": Constraint("use output.rollout_sample_rate to capture whole prompt groups"),
    "generation_samples_per_rollout": Constraint("use output.rollout_sample_rate to capture whole prompt groups"),
    "rollout_recovery_max_attempts": Constraint(
        "Core does not yet implement the baseline driver retry budget", Category.NOT_IMPLEMENTED
    ),
    "rollout_recovery_mem_fraction_static": Constraint(
        "Core does not implement recovery-time memory overrides", Category.NOT_IMPLEMENTED
    ),
    "rollout_stage_timeout": Constraint(
        "Core does not yet implement the baseline per-stage deadline", Category.NOT_IMPLEMENTED
    ),
    "rollout_health_diagnostics": Constraint(
        "use dedicated recovery probes; the baseline diagnostic wrapper is not installed", Category.NOT_IMPLEMENTED
    ),
    "rollout_test_fault": Constraint("use a dedicated fault-injection qualification, not an ordinary run"),
    "inference_ep_diagnostics": Constraint("use the separate inference-EP diagnostics"),
    "determinism_probe_samples": Constraint("use retained-input diagnostic scripts"),
    "determinism_probe_forward_trace": Constraint("use retained-input diagnostic scripts"),
    "determinism_probe_cross_gpu": Constraint("use retained-input diagnostic scripts"),
    "determinism_probe_retune_kda": Constraint("use retained-input diagnostic scripts"),
    "determinism_probe_l2norm_inputs": Constraint("use retained-input diagnostic scripts"),
    "weight_export_mode": Constraint("Core exports native HF tensors; use core.stream_moe_export"),
    "colocated_live_weight_export": Constraint(
        "Core already owns live IPC export; there is no Megatron patch selector"
    ),
    "hardware_profile": Constraint(
        "choose explicit Core/serving settings and launch.cluster; automatic hardware policy is not implemented",
        Category.NOT_IMPLEMENTED,
    ),
    "code_service_mode": Constraint("provision the verifier service externally and pass its environment"),
    "code_service_workers": Constraint("provision the verifier service externally"),
    "code_service_source_revision": Constraint("record externally provisioned service provenance"),
    "start_code_service": Constraint("per-run code-service provisioning is not implemented", Category.NOT_IMPLEMENTED),
    "code_service_source_root": Constraint(
        "per-run code-service provisioning is not implemented", Category.NOT_IMPLEMENTED
    ),
    "code_service_python": Constraint(
        "per-run code-service provisioning is not implemented", Category.NOT_IMPLEMENTED
    ),
    "code_service_host": Constraint("per-run code-service provisioning is not implemented", Category.NOT_IMPLEMENTED),
    "code_service_port": Constraint("per-run code-service provisioning is not implemented", Category.NOT_IMPLEMENTED),
    "code_service_log": Constraint("per-run code-service provisioning is not implemented", Category.NOT_IMPLEMENTED),
    "fla_prewarm": Constraint(
        "use compiler_cache.enabled; generic FLA prewarming is not implemented", Category.NOT_IMPLEMENTED
    ),
    "fla_prewarm_sequence_length": Constraint("generic FLA prewarming is not implemented", Category.NOT_IMPLEMENTED),
    "miles_train_script": Constraint("this workflow owns the Core driver"),
    "python_path": Constraint("install code in the pinned runtime image; the launcher owns PYTHONPATH"),
    "skip_cuda_check": Constraint("plan is CPU-safe; validate checks the installed runtime"),
    "validate_miles_args": Constraint("use the validate command"),
    "no_start_ray": Constraint("the launcher owns Ray startup"),
    "dataset_profile": Constraint("choose data.tasks, data.recipe or data.rl_manifest"),
    "rl_manifest": Constraint("use data.rl_manifest"),
}

# Native parser defaults are inspected for ordinary options. These options are
# inspected only when supplied explicitly, preserving existing CLI behavior.
EXPLICIT_NATIVE_OPTIONS = (
    "max_weight_staleness",
    "data_source_path",
    "custom_async_data_buffer_path",
    "gradient_checkpointing",
    "attn_implementation",
    "warmup_ratio",
    "max_tokens_per_gpu",
)

# Additional inputs needed by conditional validation, beyond REQUIRED_VALUES.
NATIVE_VALIDATION_OPTIONS = (
    frozenset(REQUIRED_VALUES)
    | (frozenset(NATIVE_CONSTRAINTS) - frozenset(EXPLICIT_NATIVE_OPTIONS))
    | frozenset(UNIMPLEMENTED_OPTIONS)
    # Keep the existing native forwarding boundary; optimizer_cpu_offload is
    # checked on the low-level config path but was not forwarded by this parser.
    | frozenset(name for name in OFFLOAD_OPTIONS if name != "optimizer_cpu_offload")
    | frozenset(
        (
            "actor_num_gpus_per_node",
            "actor_num_nodes",
            "advantage_estimator",
            "balance_data",
            "calculate_per_token_loss",
            "colocate",
            "context_parallel_size",
            "custom_convert_samples_to_train_data_path",
            "custom_generate_function_path",
            "custom_loss_function_path",
            "custom_reward_post_process_path",
            "dynamic_sampling_filter_path",
            "eval_num_gpus",
            "global_batch_size",
            "group_rm",
            "hf_checkpoint",
            "kl_coef",
            "load_debug_rollout_data",
            "loss_type",
            "n_samples_per_prompt",
            "normalize_advantages",
            "offload_rollout",
            "partial_rollout",
            "prefill_num_servers",
            "recompute_rollout_log_probs",
            "rewards_normalization",
            "rollout_all_samples_process_path",
            "rollout_batch_size",
            "rollout_external",
            "rollout_function_path",
            "rollout_num_gpus_per_engine",
            "rollout_sample_filter_path",
            "rollout_temperature",
            "rollout_top_k",
            "rollout_top_p",
            "router_pd_disaggregation",
            "sglang_config",
            "sglang_cuda_graph_backend_decode",
            "sglang_cuda_graph_backend_prefill",
            "sglang_disaggregation_mode",
            "sglang_dp_size",
            "sglang_ep_size",
            "sglang_pp_size",
            "sglang_speculative_algorithm",
            "use_dynamic_global_batch_size",
            "use_fault_tolerance",
            "use_miles_router",
            "use_rollout_indexer_replay",
            "use_rollout_logprobs",
            "use_tis",
        )
    )
)


def native_values(args, supplied_flags):
    """Select native arguments for the same validator used by CPU planning."""
    fields = {
        name: value for name in sorted(NATIVE_VALIDATION_OPTIONS) if (value := getattr(args, name, None)) is not None
    }
    for name in EXPLICIT_NATIVE_OPTIONS:
        if "--" + name.replace("_", "-") in supplied_flags:
            fields[name] = getattr(args, name)
    return fields


# Closed structured schemas, also consumed by documentation coverage checks.
STRUCTURED_FIELDS = {
    "data": {"eval_prompt_data", "prompt_data", "recipe", "reward_config", "rl_manifest", "seed", "shuffle", "tasks"},
    "launch": {
        "auto_resume",
        "budget",
        "cluster",
        "coordination",
        "env",
        "gpus_per_replica",
        "max_retries",
        "min_runtime",
        "priority",
        "secrets",
        "shared_memory",
        "timeout",
        "weka_mounts",
        "workspace",
    },
    "model": {"source", "format", "hf_template", "reference_hf"},
    "output": {"export_hf", "hf_dir", "rollout_sample_rate", "root"},
    "conversion": {"hf_output"},
    "compiler_cache": {
        "diagnostics",
        "enabled",
        "max_storage_bytes",
        "publish_interval_seconds",
        "restore",
        "shared_root",
    },
    "selection": {"table", "sha256"},
    "records": {"enabled", "responses", "response_sample_rate", "root"},
    "judging": {"bindings"},
    "judges.NAME": {
        "backend",
        "chat_template",
        "context_extension",
        "endpoint",
        "gpus",
        "max_concurrent_calls",
        "max_context_length",
        "mode",
        "model",
        "prepared_dir",
        "revision",
        "tensor_parallel_size",
        "timeout",
    },
    "rubrics.NAME": {"temperature", "profile", "max_response_tokens"},
    "judging.bindings.VERIFIER": {"rubric", "judge"},
}

# Map special run-file controls to their compiler branch. Ordinary aliases use
# FIELD_MAP; these need transformations or deferred multi-field resolution.
SPECIAL_CONTROLS = {
    "gpus": "gpus",
    "placement_mode": "deferred",
    "max_context_length": "deferred",
    "save_checkpoints": "deferred",
    "off_policy_correction": "deferred",
    "policy_drift_action": "deferred",
    "radix_cache": "radix_cache",
    "disable_radix_cache": "radix_cache",
    "trainer_diagnostics": "trainer_diagnostics",
    "recompute_mode": "recompute_mode",
    "trainer_flash_attention_version": "trainer_flash_attention_version",
    "dynamic_batching": "dynamic_batching",
}


def native_descriptions():
    """Render the same fixed rules and conditional explanations used by validation."""
    descriptions = {
        name: Constraint(f"Required value: {json.dumps(expected)}", Category.NOT_IMPLEMENTED).describe()
        for name, expected in REQUIRED_VALUES.items()
    }
    descriptions.update(
        {name: Constraint(f"Rejected; use {replacement}").describe() for name, replacement in REPLACEMENTS.items()}
    )
    descriptions.update(
        {name: Constraint("Adapter-owned; cannot override").describe() for name in ADAPTER_OWNED_OPTIONS}
    )
    descriptions.update({name: rule.describe() for name, rule in NATIVE_CONSTRAINTS.items()})
    descriptions.update(
        {
            name: Constraint(
                "Required value: 1; trainer TP/PP/CP > 1 is not implemented", Category.NOT_IMPLEMENTED
            ).describe()
            for name in TRAINER_PARALLEL_OPTIONS
        }
    )
    descriptions.update(
        {
            name: Constraint("Required value: false; no Core implementation", Category.NOT_IMPLEMENTED).describe()
            for name in (*UNIMPLEMENTED_OPTIONS, *OFFLOAD_OPTIONS)
        }
    )
    return descriptions
