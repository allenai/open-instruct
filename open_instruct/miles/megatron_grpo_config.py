"""CPU-only contract for teacher-free Megatron verifier GRPO qualification."""

import copy
import dataclasses
import re
from pathlib import Path

from open_instruct.miles import dppo_math, megatron_grpo_assets, opd_config, options, run_spec, validation
from open_instruct.miles.errors import InputError

# Megatron profile -> the HF ``model_type`` its learner checkpoint must declare.
ARCHITECTURES = {"qwen3-1.7B": "qwen3", "qwen3.5-2B": "qwen3_5"}
# ``ppo``: Miles' clipped policy loss. ``dppo``: Open Instruct's DPPO trust-region mask
# (open_instruct.miles.dppo_loss), which needs the rollout engine's behavior log-probs.
POLICY_LOSSES = ("ppo", "dppo")
DPPO_LOSS = "open_instruct.miles.dppo_loss.policy_loss"
ASYNC_ROLLOUT = "open_instruct.miles.opd_async.VerifierAsyncRollout"
DEFAULTS = {
    "model": {"source": "", "architecture": "qwen3-1.7B", "native_checkpoint": {}},
    "training": {**opd_config.DEFAULTS["training"], "algorithm": "grpo"},
    "trainer": dict(opd_config.DEFAULTS["trainer"]),
    "inference": {**opd_config.DEFAULTS["inference"], "max_prompt_length": 2048, "max_context_length": 4096},
    "optimizer": {**opd_config.DEFAULTS["optimizer"], "adam_beta2": 0.95, "adam_eps": 1e-8, "clip_grad": 1.0},
    "objective": {
        "eps_clip": 0.2,
        "eps_clip_high": 0.28,
        "eps_clip_c": 3.0,
        "std_normalization": False,
        "policy_loss": "ppo",
        # Score the PPO/DPPO ratio against the rollout engine's sampled-token log-probs (Open
        # Instruct's --use_vllm_logprobs) instead of the trainer's pre-update forward pass.
        "use_rollout_logprobs": False,
        "dppo_divergence_type": "tv",
        "dppo_threshold": 0.1,
    },
    "output": dict(opd_config.DEFAULTS["output"]),
    "tracking": {**opd_config.DEFAULTS["tracking"], "wandb_project": "rl-backend-comparison"},
}
OWNED = {
    **opd_config.OWNED_NATIVE_OPTIONS,
    "tensor_model_parallel_size": "trainer.tensor_parallel_size",
    "pipeline_model_parallel_size": "one-node GRPO topology",
    "context_parallel_size": "one-node GRPO topology",
    "expert_model_parallel_size": "dense Qwen3 topology",
    "expert_tensor_parallel_size": "dense Qwen3 topology",
    "micro_batch_size": "one-sample microbatches",
    "eps_clip": "objective.eps_clip",
    "eps_clip_high": "objective.eps_clip_high",
    "eps_clip_c": "objective.eps_clip_c",
    "grpo_std_normalization": "objective.std_normalization",
    "rewards_normalization": "group-centered GRPO rewards",
    "normalize_advantages": "no cross-group normalization",
    "compute_advantages_and_returns": "native GRPO reward-to-advantage computation",
    "kl_coef": "teacher/reference-free GRPO",
    "use_kl_loss": "teacher/reference-free GRPO",
    "kl_loss_coef": "teacher/reference-free GRPO",
    "entropy_coef": "zero entropy bonus",
    "adam_eps": "optimizer.adam_eps",
    "clip_grad": "optimizer.clip_grad",
    "skip_actor_forward_only": "pre-update trainer-scored PPO anchor",
    "rollout_max_prompt_len": "inference.max_prompt_length",
    "data_source_path": "native fixed-fanout input source (the async ledger under miles.fully_async)",
    "custom_async_data_buffer_path": "the async verifier route",
    "rollout_function_path": "native synchronous rollout",
    "custom_convert_samples_to_train_data_function_path": "native GRPO data conversion",
    "reward_key": "numeric registered verifier reward",
    "eval_reward_key": "numeric registered verifier reward",
    "update_weights_interval": "one publication per optimizer update",
    "use_tis": "rollout log-probs are the PPO anchor; no importance-sampling correction",
    "use_rollout_logprobs": "objective.use_rollout_logprobs",
    "loss_type": "objective.policy_loss",
    "custom_loss_function_path": "objective.policy_loss",
    "use_opsm": "no additional policy mask",
    "reset_optimizer_states": "persistent optimizer state",
    "custom_advantage_function_path": "native GRPO advantages",
}


@dataclasses.dataclass
class MegatronGRPORunSpec:
    document: dict
    config_path: Path

    @classmethod
    def from_dict(cls, document, *, config_path=None, overrides=None):
        document = copy.deepcopy(document)
        run_spec._apply_overrides(document, overrides)
        validation.fields(document, "run", set(DEFAULTS) | {"schema_version", "name", "data", "launch", "miles"})
        if document.get("schema_version") != 1:
            raise InputError("Megatron GRPO requires schema_version=1")
        name = validation.text(document.get("name"), "name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise InputError("name must contain only letters, digits, '.', '_' and '-'")
        base = Path(config_path or "run.toml").resolve()
        for section, defaults in DEFAULTS.items():
            incoming = document.get(section, {})
            validation.mapping(incoming, section)
            validation.fields(incoming, section, set(defaults))
            document[section] = defaults | incoming
        if document["training"]["algorithm"] != "grpo" or document["trainer"]["backend"] != "megatron":
            raise InputError("This contract requires training.algorithm=grpo and trainer.backend=megatron")
        # Only the immutable local original Qwen3 checkpoint is admitted by this first qualification.
        model = document["model"]
        if not opd_config.is_local(validation.text(model["source"], "model.source")):
            raise InputError("Megatron GRPO qualification requires a pinned local HF model.source")
        model["source"] = run_spec._path(model["source"], base.parent, "model.source")
        validation.choice(model["architecture"], "model.architecture", tuple(ARCHITECTURES))
        training, trainer, inf, opt = (document[k] for k in ("training", "trainer", "inference", "optimizer"))
        validation.choice(training["phase"], "training.phase", ("prepare", "train"))
        validation.boolean(training["resume"], "training.resume")
        validation.integer(training["num_rollouts"], "training.num_rollouts")
        validation.integer(training["save_interval"], "training.save_interval")
        validation.integer(training["eval_interval"], "training.eval_interval", minimum=0)
        validation.integer(training["keep_checkpoints"], "training.keep_checkpoints", minimum=0)
        if training["keep_checkpoints"] != 0:
            raise InputError("GRPO qualification retains all checkpoints; retention deletion is disabled")
        if training["optimizer_steps_per_rollout"] != 1 or type(training["optimizer_steps_per_rollout"]) is not int:
            raise InputError("Initial Megatron GRPO qualification requires one optimizer step per rollout")
        validation.choice(training["loss_aggregation"], "training.loss_aggregation", opd_config.LOSS_AGGREGATIONS)
        for section, keys in (
            (trainer, ("gpus", "tensor_parallel_size")),
            (
                inf,
                (
                    "gpus",
                    "tensor_parallel_size",
                    "rollout_batch_size",
                    "samples_per_prompt",
                    "max_response_length",
                    "max_prompt_length",
                    "max_context_length",
                    "max_running_requests",
                    "eval_samples_per_prompt",
                ),
            ),
        ):
            for key in keys:
                validation.integer(section[key], key)
            if section["gpus"] % section["tensor_parallel_size"]:
                raise InputError("GPU counts must be divisible by tensor_parallel_size")
        if inf["samples_per_prompt"] < 2:
            raise InputError("GRPO requires at least two responses per prompt")
        batch = inf["rollout_batch_size"] * inf["samples_per_prompt"]
        if batch % (trainer["gpus"] // trainer["tensor_parallel_size"]):
            raise InputError("GRPO collection size must divide evenly across trainer data-parallel ranks")
        validation.integer(inf["max_prompt_length"], "inference.max_prompt_length")
        if (
            inf["max_prompt_length"] + max(inf["max_response_length"], inf["eval_max_response_length"])
            > inf["max_context_length"]
        ):
            raise InputError("Context capacity must cover prompt plus response caps")
        validation.integer(inf["eval_max_response_length"], "inference.eval_max_response_length", minimum=0)
        if inf["max_context_length"] <= max(inf["max_response_length"], inf["eval_max_response_length"]):
            raise InputError("max_context_length must exceed both response caps")
        for key in ("temperature", "top_p"):
            validation.number(inf[key], f"inference.{key}", exclusive_min=True)
        for key in ("top_p", "eval_top_p"):
            validation.number(inf[key], f"inference.{key}", exclusive_min=True, maximum=1.0)
        validation.number(inf["eval_temperature"], "inference.eval_temperature")
        for key in ("learning_rate", "adam_eps", "clip_grad"):
            validation.number(opt[key], f"optimizer.{key}", exclusive_min=True)
        for key in ("min_lr", "weight_decay"):
            validation.number(opt[key], f"optimizer.{key}")
        if opt["min_lr"] > opt["learning_rate"]:
            raise InputError("min_lr must not exceed learning_rate")
        for key in ("adam_beta1", "adam_beta2"):
            validation.number(opt[key], f"optimizer.{key}", exclusive_max=True, maximum=1.0)
        validation.integer(opt["lr_warmup_iters"], "optimizer.lr_warmup_iters", minimum=0)
        validation.choice(opt["lr_decay_style"], "optimizer.lr_decay_style", opd_config.LR_DECAY_STYLES)
        for key in ("eps_clip", "eps_clip_high", "eps_clip_c"):
            validation.number(document["objective"][key], f"objective.{key}", exclusive_min=True)
        objective = document["objective"]
        validation.boolean(objective["std_normalization"], "objective.std_normalization")
        validation.boolean(objective["use_rollout_logprobs"], "objective.use_rollout_logprobs")
        validation.choice(objective["policy_loss"], "objective.policy_loss", POLICY_LOSSES)
        validation.number(objective["dppo_threshold"], "objective.dppo_threshold", exclusive_min=True)
        try:
            dppo_math.Settings(objective["dppo_divergence_type"], objective["dppo_threshold"])
        except (TypeError, ValueError) as error:
            raise InputError(f"objective.dppo_*: {error}") from None
        if objective["policy_loss"] == "dppo" and not objective["use_rollout_logprobs"]:
            raise InputError("objective.policy_loss=dppo requires objective.use_rollout_logprobs=true")
        if objective["use_rollout_logprobs"] and inf["top_p"] != 1.0:
            raise InputError("inference.top_p must be 1.0 when the rollout log-probs anchor the policy ratio")
        for key in ("root", "assets"):
            document["output"][key] = run_spec._path(document["output"][key], base.parent, f"output.{key}")
        root, assets = (Path(document["output"][k]) for k in ("root", "assets"))
        if root == assets or root in assets.parents or assets in root.parents:
            raise InputError("output.root and output.assets must be separate non-nested directories")
        megatron_grpo_assets.validate(model["native_checkpoint"], model, trainer, document["output"])
        data = run_spec.RunSpec._data(document.get("data", {}), base.parent)
        if not data.get("prompt_data") or not data.get("eval_prompt_data") or not data.get("reward_config"):
            raise InputError(
                "Megatron GRPO requires pre-rendered prompt_data, eval_prompt_data and a reward_config registry"
            )
        document["data"] = data
        native = options.normalize_options(document.get("miles", {}))
        for key in native:
            if key in OWNED or key.startswith("opd_"):
                raise InputError(f"miles.{key} is owned by {OWNED.get(key, 'teacher-free GRPO')}")
        for key in ("colocate", "offload_train", "offload_rollout", "partial_rollout"):
            if native.get(key, False):
                raise InputError(f"Megatron GRPO does not support miles.{key}")
        if native.get("update_weights_interval", 1) != 1:
            raise InputError("One weight publication per optimizer update is required")
        validation.boolean(native.get("fully_async", False), "miles.fully_async")
        if native.get("fully_async", False):
            # The OPD async producer and restart ledger (opd_async), under the same contract as
            # async OPD: one update per rollout (above), a publication after every update, and
            # behavior log-probs from the engine version that sampled each token.
            validation.integer(native.get("max_weight_staleness"), "miles.max_weight_staleness", minimum=1)
            if not objective["use_rollout_logprobs"]:
                raise InputError("Async Megatron GRPO requires objective.use_rollout_logprobs=true")
            if native.get("pause_generation_mode", "in_place") == "abort":
                raise InputError("Async Megatron GRPO cannot abort generations during weight publication")
            if "async_max_concurrent_samples" in native:
                validation.integer(native["async_max_concurrent_samples"], "miles.async_max_concurrent_samples")
            factor = native.get("async_data_buffer_capacity_factor", 2.0)
            validation.number(factor, "miles.async_data_buffer_capacity_factor", exclusive_min=True)
            if int(factor * inf["rollout_batch_size"]) < 1:
                raise InputError("Async Megatron GRPO completed buffer must hold at least one group")
        elif "max_weight_staleness" in native:
            raise InputError("miles.max_weight_staleness applies only with miles.fully_async=true")
        document["miles"] = native
        launch = run_spec.RunSpec._launch(
            {"auto_resume": False, "shared_memory": "64 GiB"} | document.get("launch", {}), base.parent
        )
        if launch["auto_resume"] and not training["resume"]:
            raise InputError("launch.auto_resume requires training.resume")
        total = trainer["gpus"] + inf["gpus"]
        if total > 8 or launch.get("gpus_per_replica", total) != total:
            raise InputError("Single-node GRPO allocation must equal trainer.gpus + inference.gpus and be <=8")
        if training["phase"] == "prepare" and launch["cluster"] != "ai2/saturn":
            raise InputError("CPU-only WEKA preparation requires ai2/saturn")
        document["launch"] = launch
        tracking = document["tracking"]
        validation.choice(tracking["wandb_mode"], "tracking.wandb_mode", opd_config.WANDB_MODES)
        validation.text(tracking["wandb_project"], "tracking.wandb_project")
        if not isinstance(tracking["wandb_entity"], str):
            raise InputError("tracking.wandb_entity must be a string")
        if tracking["wandb_mode"] == "online" and "WANDB_API_KEY" not in launch["secrets"]:
            raise InputError("Online W&B requires launch.secrets.WANDB_API_KEY")
        return cls(document, base)

    @property
    def name(self):
        return self.document["name"]

    @property
    def output(self):
        return self.document["output"]

    @property
    def launch(self):
        return self.document["launch"]

    @property
    def model(self):
        return {**self.document["model"], "format": "hf"}

    @property
    def conversion(self):
        return {"hf_output": str(Path(self.output["assets"]) / "learner-hf")}

    def to_dict(self):
        return copy.deepcopy(self.document)

    def allocation(self):
        if self.document["training"]["phase"] == "prepare":
            return {"replicas": 1, "gpus_per_replica": 0, "ray_gpus": 0, "roles": {}}
        trainer, student = (self.document[k]["gpus"] for k in ("trainer", "inference"))
        return {
            "replicas": 1,
            "gpus_per_replica": trainer + student,
            "ray_gpus": trainer + student,
            "roles": {"trainer": list(range(trainer)), "student": list(range(trainer, trainer + student))},
        }

    def dppo_settings(self):
        """DPPO settings for the custom loss, forwarded to the trainers as ``OI_DPPO_*`` environment."""
        objective = self.document["objective"]
        if objective["policy_loss"] != "dppo":
            return None
        return dppo_math.Settings(objective["dppo_divergence_type"], objective["dppo_threshold"])

    def plan(self):
        warnings = [
            "Teacher-free verifier GRPO adapter is experimental; runtime mechanics and objective parity must qualify before throughput/learning comparisons."
        ]
        if self.document["model"]["architecture"] != "qwen3-1.7B":
            warnings.append(f"{self.document['model']['architecture']} has not run on the Megatron GRPO route.")
        if self.document["objective"]["policy_loss"] == "dppo":
            warnings.append("The DPPO custom loss is unit-tested against Open Instruct but has not trained.")
        if self.document["miles"].get("fully_async", False):
            warnings.append("Async verifier GRPO reuses the async OPD producer; it has not run for verifier rewards.")
        return {
            "name": self.name,
            "backend": "megatron",
            "algorithm": "grpo",
            "allocation": self.allocation(),
            "runtime_validated": False,
            "spec": self.to_dict(),
            "warnings": warnings,
        }
