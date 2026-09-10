"""CPU-safe configuration compilation for the MILES runtime."""

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any

import tomllib


@dataclasses.dataclass(frozen=True)
class CoreConfig:
    max_train_rollout_logprob_abs_diff: float | None = None
    stream_moe_export: bool = True
    weight_sync_mode: str = "flattened"
    model_config: str | None = None
    reward_config: str | None = None
    expert_parallel_size: int = 1
    attention_backend: str = "flash_2"
    activation_checkpointing: bool = True
    max_sequence_length: int = 8192
    max_policy_lag: int = 0
    router_aux_loss_weight: float = 0.01
    router_z_loss_weight: float = 1e-5

    def __post_init__(self):
        limit = self.max_train_rollout_logprob_abs_diff
        if limit is not None and (not math.isfinite(limit) or limit < 0):
            raise ValueError("core.max_train_rollout_logprob_abs_diff must be finite and nonnegative")
        if self.weight_sync_mode not in ("flattened", "per_tensor"):
            raise ValueError("core.weight_sync_mode must be flattened or per_tensor")
        if type(self.stream_moe_export) is not bool:
            raise ValueError("core.stream_moe_export must be a boolean")
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


@dataclasses.dataclass(frozen=True)
class RunConfig:
    core: CoreConfig
    miles: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with Path(path).open("rb") as stream:
            data = tomllib.load(stream)
        unknown = set(data) - {"core", "miles"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        config = cls(CoreConfig(**data.get("core", {})), dict(data.get("miles", {})))
        config.validate()
        return config

    def validate(self) -> None:
        # All other keys are validated against the pinned MILES parser by `validate`.
        for name in self.miles:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"Invalid MILES option name: {name!r}; use underscores")
        for key in ("train_backend", "custom_config_path", "config", "olmo_core_config"):
            if key in self.miles:
                raise ValueError(f"miles.{key} is managed by the Core backend")
        if not self.miles.get("hf_checkpoint"):
            raise ValueError("miles.hf_checkpoint is required for the serving architecture/tokenizer")
        nodes = self.miles.get("actor_num_nodes", 1)
        gpus = self.miles.get("actor_num_gpus_per_node", 1)
        if any(type(n) is not int or n < 1 for n in (nodes, gpus)):
            raise ValueError("Trainer nodes and GPUs per node must be positive integers")
        world = nodes * gpus
        if world % self.core.expert_parallel_size:
            raise ValueError("Trainer world size must be divisible by core.expert_parallel_size")
        samples = self.miles.get("global_batch_size")
        if type(samples) is not int or samples < world or samples % world:
            raise ValueError("miles.global_batch_size must be a positive multiple of trainer world size")
        for name in ("offload", "fsdp_cpu_offload", "optimizer_cpu_offload", "stream_optimizer_state_to_disk"):
            if self.miles.get(name, False):
                raise ValueError(f"Core RL has no implementation for miles.{name}")
        collection = self.miles.get("rollout_batch_size", 0) * self.miles.get("n_samples_per_prompt", 1)
        if collection and (collection % samples or self.core.max_policy_lag < collection // samples - 1):
            raise ValueError("Rollout collection needs complete optimizer batches and a sufficient max_policy_lag")
        if self.miles.get("ref_update_interval") is not None:
            raise ValueError("Core RL requires a fixed reference policy")
        if self.miles.get("offload_train", False):
            raise ValueError("Core trainer offload has not been qualified; set offload_train=false")
        if self.miles.get("qkv_format", "bshd") != "bshd":
            raise ValueError("The Core backend currently uses bshd layout")
        if self.miles.get("micro_batch_size", 1) != 1 or self.miles.get("use_dynamic_batch_size", False):
            raise ValueError("Use micro_batch_size=1; Core accumulates unpadded response samples")
        for name in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size"):
            if self.miles.get(name, 1) != 1:
                raise ValueError(f"Core RL does not yet support {name}>1")
        for name in ("use_critic", "multi_lora", "indep_dp", "use_opd", "use_routing_replay"):
            if self.miles.get(name, False):
                raise ValueError(f"Core RL has no implementation for miles.{name}")
        if self.miles.get("fully_async", False):
            if self.miles.get("colocate", False) or self.miles.get("offload_rollout", False):
                raise ValueError("Async Core training requires resident disaggregated rollout engines")
            if self.miles.get("update_weights_interval", 1) != 1:
                raise ValueError("Bounded async publishes every collected batch")
        if self.miles.get("fully_async", False) and self.core.max_policy_lag == 0:
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
            **self.miles,
            "olmo_core_config": json.dumps(dataclasses.asdict(self.core), sort_keys=True),
        }
        result = []
        for name, value in options.items():
            flag = "--" + name.replace("_", "-")
            if name == "rollout_global_dataset" and isinstance(value, bool):
                if not value:
                    result.append("--disable-rollout-global-dataset")
            elif isinstance(value, bool):
                # Explicit false values need BooleanOptionalAction support; validated by argparse.
                result.append(flag if value else "--no-" + name.replace("_", "-"))
            elif isinstance(value, (str, int, float)):
                result.extend([flag, str(value)])
            elif isinstance(value, list) and value and all(isinstance(item, (str, int, float)) for item in value):
                result.extend([flag, *map(str, value)])
            else:
                raise ValueError(f"Unsupported value for miles.{name}: {value!r}")
        return result
