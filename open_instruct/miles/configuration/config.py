"""Compile low-level MILES and Core configuration into a runnable argument list.
The dataclasses validate supported combinations and attach the Open Instruct
reward, recording and data-source hooks that the runtime needs. Their plan output
also explains requested capacities and potential bottlenecks, allowing configuration
review without loading the GPU training stack.
"""

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

from open_instruct.miles.configuration import async_capacity, constraints, graph_config, throughput, validation
from open_instruct.miles.configuration import options as cli_options
from open_instruct.miles.errors import InputError

ZERO_STD_FILTER = "miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
FINISHED_STD_FILTER = "open_instruct.miles.rewards.truncation.filter_finished_reward_variance"
EXCLUDE_TRUNCATED = "open_instruct.miles.rewards.truncation.exclude_truncated"
EXPERT_SCHEDULE = "miles.backends.core_utils.expert_schedule.reorder_samples"
RECORD_RESPONSE_MODES = ("off", "all", "sample")


@dataclasses.dataclass(frozen=True)
class CoreConfig:
    reward_zero_truncated: bool = False
    reward_final_answer_only: bool = False
    max_run_seconds: float | None = None
    filter_zero_std_groups: bool = True
    max_train_rollout_logprob_abs_diff: float | None = None
    diagnostic_interval: int = 0
    pipeline_observation_interval: float = 0.0
    replay_diagnostics: bool = False
    stream_moe_export: bool = True
    weight_sync_mode: str = "flattened"
    publication_mode: str = "barrier"
    engine_drain_timeout: float = 180.0
    engine_update_timeout: float = 180.0
    refresh_request_timeout: float = 1800.0
    snapshot_capacity: int = 2
    row_specialization: str = "static"
    compiler_cache: bool = True
    compiler_cache_root: str | None = None
    compiler_cache_restore: bool = True
    compiler_cache_diagnostics: bool = False
    compiler_cache_max_storage_bytes: int = 8 * 1024**3
    compiler_cache_publish_interval_seconds: float = 600.0
    checkpoint_profile: bool = False
    checkpoint_thread_count: int | None = None
    checkpoint_process_count: int | None = None
    checkpoint_compact_storage: bool = True
    checkpoint_dedup_save_to_lowest_rank: bool = False
    checkpoint_constant_memory_planning: bool = True
    # Committed native checkpoints to retain after each commit: the newest
    # checkpoint_keep_last plus every checkpoint_keep_every-th completed update.
    # None keeps every checkpoint.
    checkpoint_keep_last: int | None = None
    checkpoint_keep_every: int | None = None
    model_config: str | None = None
    reward_config: str | None = None
    expert_parallel_size: int = 1
    attention_backend: str = "torch"
    fp32_lm_head: bool = False
    activation_checkpointing: bool = True
    compile_model: bool = False
    compile_optimizer: bool = False
    use_reduce_scatter: bool = False
    max_sequence_length: int = 8192
    sequence_packing: bool = False
    expert_balanced_packing: bool = False
    expert_balance_layer_stride: int = 1
    expert_balance_search_proposals: int = 1024
    expert_balance_search_seconds: float = 0.25
    packing_max_tokens: int | None = None
    max_policy_lag: int = 0
    router_aux_loss_grouping: str = "pack"
    router_aux_loss_reduction: str = "token"
    router_z_loss_reduction: str = "token"
    router_aux_count_source: str = "executed"
    router_aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 1e-5
    # The standalone pre-update scoring pass is skipped when the recipe makes it
    # redundant (see scoring_pass); these control the override and the periodic
    # standalone-versus-training check that guards the skipped path.
    scoring_pass_required: bool = False
    scoring_check_interval: int = 50
    scoring_check_tolerance: float = 1e-3
    # Publication layout for routed experts: per-expert HF slices, or one stacked
    # tensor per layer and projection in the serving engine's fused layout.
    expert_publication: str = "per_expert"
    # Append every scored training group, kept or filtered, to this shared store.
    records_root: str | None = None
    records_responses: str = "off"
    records_response_sample_rate: float | None = None
    # Frozen prompt-exclusion table from `records select`, pinned by SHA-256.
    selection_table: str | None = None
    selection_sha256: str | None = None

    def __post_init__(self):
        if self.max_run_seconds is not None:
            validation.number(self.max_run_seconds, "core.max_run_seconds", exclusive_min=True)
        validation.choice(self.publication_mode, "core.publication_mode", ("barrier", "engine_drain", "refresh"))
        validation.integer(self.snapshot_capacity, "core.snapshot_capacity", minimum=1)
        for name in ("engine_drain_timeout", "engine_update_timeout", "refresh_request_timeout"):
            validation.number(getattr(self, name), f"core.{name}")
            if getattr(self, name) <= 0:
                raise constraints.error(f"core.{name} must be positive")
        validation.integer(self.compiler_cache_max_storage_bytes, "core.compiler_cache_max_storage_bytes", minimum=0)
        validation.number(
            self.compiler_cache_publish_interval_seconds,
            "core.compiler_cache_publish_interval_seconds",
            exclusive_min=True,
        )
        if self.compiler_cache_root is not None:
            if not isinstance(self.compiler_cache_root, str) or not self.compiler_cache_root:
                raise constraints.error("core.compiler_cache_root must be a nonempty absolute path or unset")
            try:
                validation.compiler_cache_root(Path(self.compiler_cache_root))
            except ValueError as error:
                raise constraints.error(f"core.compiler_cache_root: {error}") from error
        if self.records_root is not None:
            validation.text(self.records_root, "core.records_root")
            if not Path(self.records_root).is_absolute():
                raise constraints.error("core.records_root must be an absolute path or unset")
        validation.choice(self.records_responses, "core.records_responses", RECORD_RESPONSE_MODES)
        if self.selection_table is not None:
            validation.text(self.selection_table, "core.selection_table")
            if not Path(self.selection_table).is_absolute():
                raise constraints.error("core.selection_table must be an absolute path or unset")
        if self.selection_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", str(self.selection_sha256)):
            raise constraints.error("core.selection_sha256 must be a lowercase 64-character SHA-256 hex digest")
        if self.records_response_sample_rate is not None:
            validation.number(
                self.records_response_sample_rate, "core.records_response_sample_rate", maximum=1, exclusive_min=True
            )
        validation.choice(
            self.attention_backend, "core.attention_backend", ("torch", "flash_2", "flash_3", "flash_4", "te")
        )
        if self.row_specialization not in ("static", "dynamic"):
            raise constraints.error("core.row_specialization must be static or dynamic")
        if self.expert_publication not in ("per_expert", "fused"):
            raise constraints.error("core.expert_publication must be per_expert or fused")
        for name in ("diagnostic_interval", "scoring_check_interval"):
            validation.integer(getattr(self, name), f"core.{name}", minimum=0)
        validation.number(self.pipeline_observation_interval, "core.pipeline_observation_interval")
        if self.pipeline_observation_interval < 0:
            raise constraints.error("core.pipeline_observation_interval must be nonnegative (0 disables observation)")
        limit = self.max_train_rollout_logprob_abs_diff
        if limit is not None:
            validation.number(limit, "core.max_train_rollout_logprob_abs_diff")
        validation.number(self.scoring_check_tolerance, "core.scoring_check_tolerance")
        for name in ("model_config", "reward_config"):
            if getattr(self, name) is not None:
                validation.text(getattr(self, name), f"core.{name}")
        if self.weight_sync_mode not in ("flattened", "per_tensor"):
            raise constraints.error("core.weight_sync_mode must be flattened or per_tensor")
        for name in (
            "reward_zero_truncated",
            "reward_final_answer_only",
            "filter_zero_std_groups",
            "compiler_cache",
            "compiler_cache_restore",
            "compiler_cache_diagnostics",
            "replay_diagnostics",
            "stream_moe_export",
            "activation_checkpointing",
            "fp32_lm_head",
            "compile_model",
            "compile_optimizer",
            "use_reduce_scatter",
            "checkpoint_profile",
            "checkpoint_compact_storage",
            "checkpoint_dedup_save_to_lowest_rank",
            "checkpoint_constant_memory_planning",
            "scoring_pass_required",
            "sequence_packing",
            "expert_balanced_packing",
        ):
            validation.boolean(getattr(self, name), f"core.{name}")
        for name in (
            "checkpoint_thread_count",
            "checkpoint_process_count",
            "checkpoint_keep_last",
            "checkpoint_keep_every",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise constraints.error(f"core.{name} must be a positive integer or unset")
        for name in ("expert_parallel_size", "max_sequence_length", "expert_balance_layer_stride"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise constraints.error(f"core.{name} must be a positive integer")
        validation.integer(self.expert_balance_search_proposals, "core.expert_balance_search_proposals", minimum=0)
        validation.number(self.expert_balance_search_seconds, "core.expert_balance_search_seconds")
        if self.packing_max_tokens is not None:
            validation.integer(self.packing_max_tokens, "core.packing_max_tokens", minimum=1)
        if type(self.max_policy_lag) is not int or self.max_policy_lag < 0:
            raise constraints.error("core.max_policy_lag must be a nonnegative integer")
        validation.choice(self.router_aux_loss_grouping, "core.router_aux_loss_grouping", ("pack", "sequence"))
        validation.choice(
            self.router_aux_count_source, "core.router_aux_count_source", ("executed", "router_selected")
        )
        for name in ("router_aux_loss_reduction", "router_z_loss_reduction"):
            validation.choice(getattr(self, name), f"core.{name}", ("token", "response"))
        if self.compile_model and (
            self.router_aux_loss_grouping,
            self.router_aux_loss_reduction,
            self.router_z_loss_reduction,
        ) != ("pack", "token", "token"):
            raise constraints.error("Document router objectives currently require core.compile_model=false")
        for name in ("router_aux_loss_weight", "router_z_loss_weight"):
            value = getattr(self, name)
            validation.number(value, f"core.{name}")

    def checkpoint_save_options(self):
        """Writer policy, kept separate from model geometry and resume compatibility."""
        return {
            field.name.removeprefix("checkpoint_"): getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name.startswith("checkpoint_")
            and field.name not in {"checkpoint_keep_last", "checkpoint_keep_every"}
            and getattr(self, field.name) is not None
        }


@dataclasses.dataclass(frozen=True)
class ScoringPass:
    """Whether every update runs the standalone pre-update scoring pass."""

    standalone: bool
    reason: str
    optimizer_steps_per_collection: int | None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def scoring_pass(core: CoreConfig, options: dict[str, Any]) -> ScoringPass:
    """Decide from the static recipe whether the standalone scoring pass carries information.

    The pass produces the old log-probabilities for the PPO ratio. With exactly one
    optimizer step per collection and no KL term in the advantages, those values are
    the training forward's own log-probabilities at unchanged weights, so the trainer
    reads them there instead. Rollout log-probabilities as the anchor make the pass
    diagnostic-only as well. Model-level conditions (dropout) are checked by the trainer,
    which can see the loaded configuration.
    """
    samples = options.get("global_batch_size")
    collection = options.get("rollout_batch_size", 0) * options.get("n_samples_per_prompt", 1)
    steps = collection // samples if collection and samples else None
    if core.scoring_pass_required:
        return ScoringPass(True, "core.scoring_pass_required", steps)
    if steps is None:
        return ScoringPass(True, "unknown collection size; one optimizer step per collection is unproven", steps)
    if steps != 1:
        return ScoringPass(True, f"{steps} optimizer steps per collection need the pre-update anchor", steps)
    if options.get("kl_coef", 0) != 0:
        return ScoringPass(True, "kl_coef needs actor log-probabilities before advantages", steps)
    anchor = (
        "rollout log-probabilities"
        if options.get("use_rollout_logprobs", False)
        else "the training forward at unchanged weights"
    )
    return ScoringPass(
        False, f"one optimizer step per collection with zero KL; old log-probabilities are {anchor}", steps
    )


@dataclasses.dataclass(frozen=True)
class RunConfig:
    core: CoreConfig
    miles: dict[str, Any]

    def resolved_miles(self) -> dict[str, Any]:
        """Resolve the default group filter for plans, native argv and direct CLI callers."""
        options = cli_options.normalize_options(self.miles)
        if self.core.fp32_lm_head:
            if options.get("sglang_enable_fp32_lm_head") is False:
                raise constraints.error("core.fp32_lm_head conflicts with miles.sglang_enable_fp32_lm_head=false")
            options["sglang_enable_fp32_lm_head"] = True
        path = options.get("dynamic_sampling_filter_path")
        builtin_filter = (
            FINISHED_STD_FILTER
            if options.get("custom_reward_post_process_path") == EXCLUDE_TRUNCATED
            else ZERO_STD_FILTER
        )
        if self.core.filter_zero_std_groups:
            if path not in (None, ZERO_STD_FILTER, builtin_filter):
                raise constraints.error(
                    "core.filter_zero_std_groups conflicts with miles.dynamic_sampling_filter_path; "
                    "disable the built-in filter before selecting a custom filter"
                )
            if options.get("n_samples_per_prompt") == 1:
                raise constraints.error(
                    "filter_zero_std_groups requires samples_per_prompt > 1; disable it for single samples"
                )
            options["dynamic_sampling_filter_path"] = builtin_filter
        elif path in (ZERO_STD_FILTER, FINISHED_STD_FILTER):
            raise constraints.error(
                "core.filter_zero_std_groups=false conflicts with the explicit zero-std filter path"
            )
        hook = options.get("rollout_sample_filter_path")
        if self.core.expert_balanced_packing:
            if hook not in (None, EXPERT_SCHEDULE):
                raise constraints.error("expert_balanced_packing conflicts with miles.rollout_sample_filter_path")
            options["rollout_sample_filter_path"] = EXPERT_SCHEDULE
        elif hook == EXPERT_SCHEDULE:
            raise constraints.error("The expert schedule hook requires core.expert_balanced_packing=true")
        return options

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "RunConfig":
        return cls.from_dict(validation.read_document(path), overrides)

    @classmethod
    def from_dict(cls, data, overrides=None):
        validation.fields(data, "configuration", {"core", "miles"})
        data = {key: dict(validation.mapping(value, f"[{key}]")) for key, value in data.items()}
        for override in overrides or []:
            key, separator, raw = validation.text(override, "Override").partition("=")
            parts = key.split(".")
            if not separator or len(parts) != 2 or parts[0] not in ("core", "miles"):
                raise constraints.error("Overrides must be core.KEY=TOML_VALUE or miles.KEY=TOML_VALUE")
            value = validation.override_value(key, raw)
            data.setdefault(parts[0], {})[parts[1]] = value
        validation.fields(data.get("core", {}), "[core]", {f.name for f in dataclasses.fields(CoreConfig)})
        miles = cli_options.normalize_options(data.get("miles", {}))
        if "max_weight_staleness" in miles:
            lag = miles["max_weight_staleness"]
            validation.integer(lag, "miles.max_weight_staleness", minimum=0)
            core = data.setdefault("core", {})
            if "max_policy_lag" in core and core["max_policy_lag"] != lag:
                raise constraints.error(
                    "Conflicting settings for policy lag: core.max_policy_lag and miles.max_weight_staleness"
                )
            core.setdefault("max_policy_lag", lag)
        config = cls(CoreConfig(**data.get("core", {})), dict(data.get("miles", {})))
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.core, CoreConfig):
            raise constraints.error(
                "RunConfig.core must be a CoreConfig instance; use CoreConfig(...) for Core settings."
            )
        validation.mapping(self.miles, "[miles]")
        # Normalize aliases before checking the backend contract.
        for name in self.miles:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise constraints.error(f"Invalid MILES option name: {name!r}; use underscores")
        for key in constraints.ADAPTER_OWNED_OPTIONS:
            if key in self.miles:
                raise constraints.error(f"miles.{key} is managed by the Core backend")
        cli_options.encode_options(self.miles)
        if (self.core.selection_table is None) != (self.core.selection_sha256 is None):
            raise constraints.error("core.selection_table and core.selection_sha256 must be set together")
        if (self.core.records_responses == "sample") != (self.core.records_response_sample_rate is not None):
            raise constraints.error(
                'core.records_response_sample_rate is required with, and only with, records_responses="sample"'
            )
        if self.core.sequence_packing and self.core.attention_backend == "torch":
            raise constraints.error(
                'core.sequence_packing=true is incompatible with core.attention_backend="torch": '
                "Torch attention does not support packed document boundaries. "
                'Select a packing-capable backend (for example core.attention_backend="flash_4" '
                "on supported hardware), or disable sequence_packing."
            )
        if self.core.packing_max_tokens is not None:
            if not self.core.sequence_packing:
                raise constraints.error("core.packing_max_tokens requires sequence_packing=true")
            if self.core.packing_max_tokens < self.core.max_sequence_length:
                raise constraints.error("packing_max_tokens must cover max_sequence_length; samples are never split")
        options = self.resolved_miles()
        if self.core.reward_zero_truncated and options.get("custom_reward_post_process_path"):
            raise constraints.error("Zero-truncation rewards cannot be combined with reward exclusion/postprocessing")
        validation.runtime_values(options)
        if self.core.records_root is not None and not options.get("fully_async", False):
            # The completed buffer that records groups exists only in fully async runs.
            raise constraints.error(
                "core.records_root (records.enabled) requires fully_async=true; synchronous runs record nothing"
            )
        prompt_limit = options.get("rollout_max_prompt_len")
        context_limit = options.get("rollout_max_context_len")
        if prompt_limit is not None and context_limit is not None and prompt_limit >= context_limit:
            raise constraints.error("rollout_max_prompt_len must be smaller than rollout_max_context_len")
        for name, expected in constraints.REQUIRED_VALUES.items():
            if name in options and options[name] != expected:
                raise constraints.error(
                    f"Core RL requires miles.{name}={expected!r}; this alternative is not implemented",
                    constraints.Category.NOT_IMPLEMENTED,
                )
        for name, replacement in constraints.REPLACEMENTS.items():
            if name in options:
                raise constraints.error(f"Core RL does not consume miles.{name}; use {replacement}")
        if options.get("lora_rank", 0) > 0:
            raise constraints.NATIVE_CONSTRAINTS["lora_rank"].error()
        if options.get("save_hf") is not None:
            raise constraints.NATIVE_CONSTRAINTS["save_hf"].error()
        if "max_weight_staleness" in options and options["max_weight_staleness"] != self.core.max_policy_lag:
            raise constraints.NATIVE_CONSTRAINTS["max_weight_staleness"].error()
        if not options.get("hf_checkpoint"):
            raise constraints.error("miles.hf_checkpoint is required for the serving architecture/tokenizer")
        nodes = options.get("actor_num_nodes", 1)
        gpus = options.get("actor_num_gpus_per_node", 1)
        if any(type(n) is not int or n < 1 for n in (nodes, gpus)):
            raise constraints.error("Trainer nodes and GPUs per node must be positive integers")
        world = nodes * gpus
        if self.core.expert_balanced_packing:
            if not 1 < self.core.expert_parallel_size < world:
                raise constraints.error("expert_balanced_packing requires world > expert_parallel_size > 1")
            if not self.core.sequence_packing:
                raise constraints.error("expert_balanced_packing requires sequence_packing=true")
            if self.core.router_aux_loss_weight != 0:
                raise constraints.error("expert_balanced_packing requires router_aux_loss_weight=0")
            if not options.get("use_rollout_routing_replay", False):
                raise constraints.error("expert_balanced_packing requires use_rollout_routing_replay=true")
            if options.get("balance_data", False):
                raise constraints.error(
                    "core.expert_balanced_packing and miles.balance_data are mutually exclusive: "
                    "the expert packing planner requires stride partitioning, but balance_data changes "
                    "sample-to-rank assignment. Enable only one."
                )
            for name in (
                "use_dynamic_global_batch_size",
                "multi_lora",
                "partial_rollout",
                "custom_reward_post_process_path",
                "custom_convert_samples_to_train_data_path",
            ):
                if options.get(name):
                    raise constraints.error(
                        f"expert_balanced_packing does not support miles.{name}", constraints.Category.NOT_IMPLEMENTED
                    )
            if self.core.model_config:
                raise constraints.error(
                    "expert_balanced_packing requires the model layout from the HF configuration",
                    constraints.Category.NOT_IMPLEMENTED,
                )
        if world % self.core.expert_parallel_size:
            raise constraints.error(
                f"Trainer world size {world} ({nodes} nodes × {gpus} GPUs) must be divisible by "
                f"core.expert_parallel_size={self.core.expert_parallel_size}; choose a divisor of {world}."
            )
        samples = options.get("global_batch_size")
        if type(samples) is not int or samples < world or samples % world:
            raise constraints.error(
                f"miles.global_batch_size={samples!r} must be a positive multiple of trainer world size {world}; "
                f"use {world}, {2 * world}, or another multiple."
            )
        for name in constraints.OFFLOAD_OPTIONS:
            if options.get(name, False):
                raise constraints.error(
                    f"Core RL has no implementation for miles.{name}", constraints.Category.NOT_IMPLEMENTED
                )
        for name in ("rollout_batch_size", "n_samples_per_prompt", "num_rollout"):
            if name in options and (type(options[name]) is not int or options[name] < 1):
                raise constraints.error(f"miles.{name} must be a positive integer")
        collection = options.get("rollout_batch_size", 0) * options.get("n_samples_per_prompt", 1)
        if collection and collection % samples:
            raise constraints.error(
                f"Rollout collection has {collection} samples (rollout_batch_size × n_samples_per_prompt); "
                f"miles.global_batch_size={samples} must divide that count to form complete optimizer batches."
            )
        if collection and self.core.max_policy_lag < collection // samples - 1:
            raise constraints.error(
                f"This collection takes {collection // samples} optimizer steps; "
                f"set core.max_policy_lag >= {collection // samples - 1} or increase miles.global_batch_size."
            )
        if options.get("check_weight_update_selector", "all") != "all":
            raise constraints.NATIVE_CONSTRAINTS["check_weight_update_selector"].error()
        if options.get("ref_update_interval") is not None:
            raise constraints.NATIVE_CONSTRAINTS["ref_update_interval"].error()
        if options.get("offload_train", False):
            raise constraints.NATIVE_CONSTRAINTS["offload_train"].error()
        layout = "thd" if self.core.sequence_packing else "bshd"
        if options.get("qkv_format", layout) != layout:
            raise constraints.error(f"Core sequence_packing={self.core.sequence_packing} requires qkv_format={layout}")
        if options.get("micro_batch_size", 1) != 1 or options.get("use_dynamic_batch_size", False):
            raise constraints.NATIVE_CONSTRAINTS["micro_batch_size"].error()
        for name in constraints.TRAINER_PARALLEL_OPTIONS:
            if options.get(name, 1) != 1:
                raise constraints.error(f"Core RL does not yet support {name}>1", constraints.Category.NOT_IMPLEMENTED)
        for name in constraints.UNIMPLEMENTED_OPTIONS:
            if options.get(name, False):
                raise constraints.error(
                    f"Core RL has no implementation for miles.{name}", constraints.Category.NOT_IMPLEMENTED
                )
        if options.get("use_rollout_routing_replay", False) and not options.get("use_miles_router", False):
            raise constraints.NATIVE_CONSTRAINTS["use_rollout_routing_replay"].error()
        if self.core.replay_diagnostics and not options.get("use_rollout_routing_replay", False):
            raise constraints.error(
                "core.replay_diagnostics requires rollout routing replay; set miles.use_rollout_routing_replay=true and miles.use_miles_router=true, or disable replay_diagnostics."
            )
        if options.get("fully_async", False):
            if options.get("colocate", False) or options.get("offload_rollout", False):
                raise constraints.NATIVE_CONSTRAINTS["fully_async"].error()
            if options.get("update_weights_interval", 1) != 1:
                raise constraints.error("Bounded async publishes every collected batch")
        if self.core.publication_mode == "engine_drain":
            self._validate_engine_drain(options, collection, samples)
        if self.core.publication_mode == "refresh":
            self._validate_refresh(options, collection, samples)
        if options.get("fully_async", False) and self.core.max_policy_lag == 0:
            raise constraints.error("Async training requires an explicit positive core.max_policy_lag")

    def _validate_engine_drain(self, options, collection, samples):
        if not options.get("fully_async", False):
            raise constraints.error("engine_drain requires miles.fully_async=true")
        if options.get("rollout_num_gpus_per_engine", 1) != 1:
            raise constraints.error(
                "engine_drain currently supports TP1 serving only", constraints.Category.NOT_IMPLEMENTED
            )
        if collection != samples:
            raise constraints.error(
                "engine_drain currently requires one optimizer step per rollout collection",
                constraints.Category.NOT_IMPLEMENTED,
            )
        if self.core.weight_sync_mode != "flattened":
            raise constraints.error(
                "engine_drain requires core.weight_sync_mode=flattened", constraints.Category.NOT_IMPLEMENTED
            )
        if self.core.diagnostic_interval != 0:
            raise constraints.error(
                "engine_drain requires diagnostic_interval=0; full publication audits run at startup",
                constraints.Category.NOT_IMPLEMENTED,
            )
        for name in ("partial_rollout", "use_fault_tolerance", "rollout_external"):
            if options.get(name, False):
                raise constraints.error(
                    f"engine_drain does not yet support miles.{name}; disable it or use publication_mode=barrier",
                    constraints.Category.NOT_IMPLEMENTED,
                )
        for name in ("sglang_dp_size", "sglang_ep_size", "sglang_pp_size"):
            if options.get(name, 1) != 1:
                raise constraints.error(f"engine_drain requires miles.{name}=1", constraints.Category.NOT_IMPLEMENTED)
        if options.get("router_pd_disaggregation", False) or options.get("prefill_num_servers") is not None:
            raise constraints.error(
                "engine_drain does not support serving prefill/decode disaggregation",
                constraints.Category.NOT_IMPLEMENTED,
            )
        if options.get("sglang_disaggregation_mode", "null") not in (None, "null"):
            raise constraints.error(
                "engine_drain requires ordinary TP1 engines, not a prefill/decode serving role",
                constraints.Category.NOT_IMPLEMENTED,
            )
        if options.get("eval_num_gpus", 0) > 0:
            raise constraints.error(
                "engine_drain currently supports blocking shared-engine evaluation only",
                constraints.Category.NOT_IMPLEMENTED,
            )
        if options.get("dynamic_sampling_filter_path") not in (None, ZERO_STD_FILTER, FINISHED_STD_FILTER):
            raise constraints.error(
                "engine_drain supports only the built-in zero-std dynamic sampling filter",
                constraints.Category.NOT_IMPLEMENTED,
            )
        for name in (
            "custom_generate_function_path",
            "sglang_config",
            "load_debug_rollout_data",
            "rollout_function_path",
            "eval_function_path",
            "rollout_sample_filter_path",
            "rollout_router_url",
        ):
            if name == "rollout_sample_filter_path" and self.core.expert_balanced_packing:
                continue  # resolved_miles admits only our managed hook.
            if options.get(name):
                raise constraints.error(
                    f"engine_drain requires the managed single-turn producer; remove miles.{name}",
                    constraints.Category.NOT_IMPLEMENTED,
                )

    def _validate_refresh(self, options, collection, samples):
        # The first qualification shares the independent publisher's resident
        # TP1/single-turn restrictions, without its immutable snapshots.
        try:
            self._validate_engine_drain(options, collection, samples)
        except InputError as error:
            raise InputError(str(error).replace("engine_drain", "refresh")) from error
        if not options.get("use_miles_router", False):
            raise constraints.error("refresh requires miles.use_miles_router=true to retain policy-span metadata")
        if not options.get("use_tis", False) or options.get("use_rollout_logprobs", False):
            raise constraints.error("refresh requires miles.use_tis=true and miles.use_rollout_logprobs=false")
        if options.get("advantage_estimator", "grpo") != "grpo":
            raise constraints.error(
                "refresh currently supports the token-level grpo objective only", constraints.Category.NOT_IMPLEMENTED
            )
        if options.get("sglang_speculative_algorithm") or options.get("use_rollout_indexer_replay", False):
            raise constraints.error(
                "refresh does not yet support speculative decoding or indexer replay",
                constraints.Category.NOT_IMPLEMENTED,
            )
        # Full decode graphs retain model/state buffer addresses across in-place
        # publication. Prefill and compiler-driven graph modes remain excluded.
        graphs = graph_config.explicit_settings(options)
        if graphs["decode"].get("backend") not in ("disabled", "full"):
            raise constraints.error(
                "refresh requires miles.sglang_cuda_graph_backend_decode=disabled or full",
                constraints.Category.NOT_VALIDATED,
            )
        if graphs["prefill"].get("backend") != "disabled":
            raise constraints.error(
                "refresh requires miles.sglang_cuda_graph_backend_prefill=disabled, including JSON overrides",
                constraints.Category.NOT_VALIDATED,
            )
        if options.get("rollout_temperature", 1.0) != 1.0 or options.get("rollout_top_p", 1.0) != 1.0:
            raise constraints.error(
                "refresh qualification requires rollout_temperature=1 and rollout_top_p=1",
                constraints.Category.NOT_VALIDATED,
            )
        if options.get("rollout_top_k", -1) != -1:
            raise constraints.error(
                "refresh qualification requires rollout_top_k=-1", constraints.Category.NOT_VALIDATED
            )

    def arguments(self) -> list[str]:
        """Compile without importing CUDA, MILES, Core, or downloading models."""
        self.validate()
        core = dataclasses.asdict(self.core)
        if not self.core.expert_balanced_packing:
            # Preserve the pre-feature native argv when scheduling is disabled.
            core.pop("expert_balanced_packing")
            core.pop("expert_balance_layer_stride")
            core.pop("expert_balance_search_proposals")
            core.pop("expert_balance_search_seconds")
        options = {
            "train_backend": "olmo_core",
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 1,
            "micro_batch_size": 1,
            "qkv_format": "thd" if self.core.sequence_packing else "bshd",
            "offload_train": False,
            "data_pad_size_multiplier": 1,
            **self.resolved_miles(),
            "olmo_core_config": json.dumps(core, sort_keys=True),
        }
        return cli_options.encode_options(options)

    def plan(self) -> dict[str, Any]:
        """Describe explicit settings; installed-runtime and model checks belong to validate."""
        argv = self.arguments()
        options = self.resolved_miles()
        world = options.get("actor_num_nodes", 1) * options.get("actor_num_gpus_per_node", 1)
        collection = options.get("rollout_batch_size", 0) * options.get("n_samples_per_prompt", 1)
        return {
            "argv": argv,
            "async_capacity": async_capacity.report(options, self.core.max_policy_lag),
            "throughput": throughput.report(options, self.core),
            "runtime_validated": False,
            "core": dataclasses.asdict(self.core),
            "miles": options,
            "shape": {
                "placement": "colocated (resident trainer)" if options.get("colocate", False) else "disaggregated",
                "trainer_gpus": world,
                "rollout_gpus": options.get("rollout_num_gpus"),
                "fully_async": options.get("fully_async", False),
                "max_policy_lag_optimizer_steps": self.core.max_policy_lag,
                "samples_per_collection": collection or None,
                "samples_per_optimizer_step": options["global_batch_size"],
                "optimizer_steps_per_collection": collection // options["global_batch_size"] if collection else None,
            },
            "scoring_pass": {
                **scoring_pass(self.core, options).as_dict(),
                "check_interval": self.core.scoring_check_interval,
                "check_tolerance": self.core.scoring_check_tolerance,
            },
        }
