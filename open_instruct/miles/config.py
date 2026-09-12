"""CPU-safe configuration compilation for the MILES runtime."""

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

import tomllib

from open_instruct.miles import compiler_cache as cache
from open_instruct.miles import options as cli_options


@dataclasses.dataclass(frozen=True)
class CoreConfig:
    max_train_rollout_logprob_abs_diff: float | None = None
    diagnostic_interval: int = 0
    replay_diagnostics: bool = False
    stream_moe_export: bool = True
    weight_sync_mode: str = "flattened"
    row_specialization: str = "static"
    compiler_cache: bool = True
    compiler_cache_root: str | None = None
    compiler_cache_restore: bool = True
    compiler_cache_diagnostics: bool = False
    checkpoint_profile: bool = False
    checkpoint_thread_count: int | None = None
    checkpoint_process_count: int | None = None
    checkpoint_compact_storage: bool = True
    checkpoint_dedup_save_to_lowest_rank: bool = False
    checkpoint_constant_memory_planning: bool = True
    model_config: str | None = None
    reward_config: str | None = None
    expert_parallel_size: int = 1
    attention_backend: str = "torch"
    activation_checkpointing: bool = True
    max_sequence_length: int = 8192
    max_policy_lag: int = 0
    router_aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 1e-5
    # The standalone pre-update scoring pass is skipped when the recipe makes it
    # redundant (see scoring_pass); these control the override and the periodic
    # standalone-versus-training check that guards the skipped path.
    scoring_pass_required: bool = False
    scoring_check_interval: int = 50
    scoring_check_tolerance: float = 1e-3

    def __post_init__(self):
        if self.compiler_cache_root is not None:
            if not isinstance(self.compiler_cache_root, str) or not self.compiler_cache_root:
                raise ValueError("core.compiler_cache_root must be a nonempty absolute path or unset")
            cache.validate_shared_root(Path(self.compiler_cache_root))
        if self.row_specialization not in ("static", "dynamic"):
            raise ValueError("core.row_specialization must be static or dynamic")
        for name in ("diagnostic_interval", "scoring_check_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"core.{name} must be a nonnegative integer")
        limit = self.max_train_rollout_logprob_abs_diff
        if limit is not None and (not math.isfinite(limit) or limit < 0):
            raise ValueError("core.max_train_rollout_logprob_abs_diff must be finite and nonnegative")
        tolerance = self.scoring_check_tolerance
        if type(tolerance) is bool or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance):
            raise ValueError("core.scoring_check_tolerance must be a finite number")
        if tolerance < 0:
            raise ValueError("core.scoring_check_tolerance must be nonnegative")
        if self.weight_sync_mode not in ("flattened", "per_tensor"):
            raise ValueError("core.weight_sync_mode must be flattened or per_tensor")
        for name in (
            "compiler_cache",
            "compiler_cache_restore",
            "compiler_cache_diagnostics",
            "replay_diagnostics",
            "stream_moe_export",
            "activation_checkpointing",
            "checkpoint_profile",
            "checkpoint_compact_storage",
            "checkpoint_dedup_save_to_lowest_rank",
            "checkpoint_constant_memory_planning",
            "scoring_pass_required",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"core.{name} must be a boolean")
        for name in ("checkpoint_thread_count", "checkpoint_process_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"core.{name} must be a positive integer or unset")
        for name in ("expert_parallel_size", "max_sequence_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"core.{name} must be a positive integer")
        if type(self.max_policy_lag) is not int or self.max_policy_lag < 0:
            raise ValueError("core.max_policy_lag must be a nonnegative integer")
        for name in ("router_aux_loss_weight", "router_z_loss_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"core.{name} must be finite and nonnegative")

    def checkpoint_save_options(self):
        """Writer policy, kept separate from model geometry and resume compatibility."""
        return {
            field.name.removeprefix("checkpoint_"): getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name.startswith("checkpoint_") and getattr(self, field.name) is not None
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


def stochastic_fields(model_config: dict[str, Any]) -> list[str]:
    """Model settings that make a forward pass non-repeatable, so no single old log-probability exists."""
    return sorted(name for name, value in model_config.items() if "dropout" in name and value)


def scoring_check_due(core: CoreConfig, checks_done: int, completed_steps: int) -> bool:
    """Check the first update of every process (including after resume), then on the interval."""
    interval = core.scoring_check_interval
    return checks_done == 0 or (interval > 0 and completed_steps % interval == 0)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    core: CoreConfig
    miles: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> "RunConfig":
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
        for override in overrides or []:
            key, separator, raw = override.partition("=")
            parts = key.split(".")
            if not separator or len(parts) != 2 or parts[0] not in ("core", "miles"):
                raise ValueError("Overrides must be core.KEY=TOML_VALUE or miles.KEY=TOML_VALUE")
            try:
                value = tomllib.loads("value=" + raw)["value"]
            except tomllib.TOMLDecodeError as error:
                raise ValueError(f"Invalid TOML value in override {key}; quote strings") from error
            data.setdefault(parts[0], {})[parts[1]] = value
        unknown = set(data) - {"core", "miles"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        config = cls(CoreConfig(**data.get("core", {})), dict(data.get("miles", {})))
        config.validate()
        return config

    def validate(self) -> None:
        # Normalize aliases before checking the backend contract.
        for name in self.miles:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"Invalid MILES option name: {name!r}; use underscores")
        for key in (
            "train_backend",
            "custom_config_path",
            "config",
            "olmo_core_config",
            "data_source_path",
            "custom_async_data_buffer_path",
        ):
            if key in self.miles:
                raise ValueError(f"miles.{key} is managed by the Core backend")
        cli_options.encode_options(self.miles)
        options = cli_options.normalize_options(self.miles)
        for name, expected in {
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
        }.items():
            if name in options and options[name] != expected:
                raise ValueError(f"Core RL requires miles.{name}={expected!r}; this alternative is not implemented")
        for name, replacement in {
            "gradient_checkpointing": "core.activation_checkpointing",
            "attn_implementation": "core.attention_backend",
            "warmup_ratio": "miles.lr_warmup_fraction",
            "max_tokens_per_gpu": "micro_batch_size=1 (Core dynamic batching is not implemented)",
        }.items():
            if name in options:
                raise ValueError(f"Core RL does not consume miles.{name}; use {replacement}")
        if options.get("lora_rank", 0) > 0:
            raise ValueError("Core RL does not implement LoRA; miles.lora_rank must be nonpositive")
        if options.get("save_hf") is not None:
            raise ValueError(
                "Core RL does not implement miles.save_hf; use miles.eval_hf_dir for snapshot evaluation "
                "or export the native checkpoint separately"
            )
        if "max_weight_staleness" in options and options["max_weight_staleness"] != self.core.max_policy_lag:
            raise ValueError("miles.max_weight_staleness must equal core.max_policy_lag (optimizer steps)")
        if not options.get("hf_checkpoint"):
            raise ValueError("miles.hf_checkpoint is required for the serving architecture/tokenizer")
        nodes = options.get("actor_num_nodes", 1)
        gpus = options.get("actor_num_gpus_per_node", 1)
        if any(type(n) is not int or n < 1 for n in (nodes, gpus)):
            raise ValueError("Trainer nodes and GPUs per node must be positive integers")
        world = nodes * gpus
        if world % self.core.expert_parallel_size:
            raise ValueError("Trainer world size must be divisible by core.expert_parallel_size")
        samples = options.get("global_batch_size")
        if type(samples) is not int or samples < world or samples % world:
            raise ValueError("miles.global_batch_size must be a positive multiple of trainer world size")
        for name in ("offload", "fsdp_cpu_offload", "optimizer_cpu_offload", "stream_optimizer_state_to_disk"):
            if options.get(name, False):
                raise ValueError(f"Core RL has no implementation for miles.{name}")
        for name in ("rollout_batch_size", "n_samples_per_prompt", "num_rollout"):
            if name in options and (type(options[name]) is not int or options[name] < 1):
                raise ValueError(f"miles.{name} must be a positive integer")
        collection = options.get("rollout_batch_size", 0) * options.get("n_samples_per_prompt", 1)
        if collection and (collection % samples or self.core.max_policy_lag < collection // samples - 1):
            raise ValueError("Rollout collection needs complete optimizer batches and a sufficient max_policy_lag")
        if options.get("check_weight_update_selector", "all") != "all":
            raise ValueError("Core serving checks currently require check_weight_update_selector=all")
        if options.get("ref_update_interval") is not None:
            raise ValueError("Core RL requires a fixed reference policy")
        if options.get("offload_train", False):
            raise ValueError("Core trainer offload has not been qualified; set offload_train=false")
        if options.get("qkv_format", "bshd") != "bshd":
            raise ValueError("The Core backend currently uses bshd layout")
        if options.get("micro_batch_size", 1) != 1 or options.get("use_dynamic_batch_size", False):
            raise ValueError("Use micro_batch_size=1; Core accumulates unpadded response samples")
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
            if options.get(name, 1) != 1:
                raise ValueError(f"Core RL does not yet support {name}>1")
        for name in ("use_critic", "multi_lora", "indep_dp", "use_opd", "use_routing_replay"):
            if options.get(name, False):
                raise ValueError(f"Core RL has no implementation for miles.{name}")
        if options.get("use_rollout_routing_replay", False) and not options.get("use_miles_router", False):
            raise ValueError(
                "The pinned SGLang router strips expert-ID requests; rollout replay requires use_miles_router"
            )
        if self.core.replay_diagnostics and not options.get("use_rollout_routing_replay", False):
            raise ValueError("core.replay_diagnostics requires rollout routing replay")
        if options.get("fully_async", False):
            if options.get("colocate", False) or options.get("offload_rollout", False):
                raise ValueError("Async Core training requires resident disaggregated rollout engines")
            if options.get("update_weights_interval", 1) != 1:
                raise ValueError("Bounded async publishes every collected batch")
        if options.get("fully_async", False) and self.core.max_policy_lag == 0:
            raise ValueError("Async training requires an explicit positive core.max_policy_lag")

    def arguments(self) -> list[str]:
        """Compile without importing CUDA, MILES, Core, or downloading models."""
        self.validate()
        options = {
            "train_backend": "olmo_core",
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 1,
            "micro_batch_size": 1,
            "qkv_format": "bshd",
            "offload_train": False,
            "data_pad_size_multiplier": 1,
            **cli_options.normalize_options(self.miles),
            "olmo_core_config": json.dumps(dataclasses.asdict(self.core), sort_keys=True),
        }
        return cli_options.encode_options(options)

    def plan(self) -> dict[str, Any]:
        """Describe explicit settings; installed-runtime and model checks belong to validate."""
        argv = self.arguments()
        options = cli_options.normalize_options(self.miles)
        world = options.get("actor_num_nodes", 1) * options.get("actor_num_gpus_per_node", 1)
        collection = options.get("rollout_batch_size", 0) * options.get("n_samples_per_prompt", 1)
        return {
            "argv": argv,
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
