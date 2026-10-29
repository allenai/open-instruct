"""CPU-only contract for teacher-free verifier GRPO on a Megatron learner."""

import copy
import dataclasses
import re
from pathlib import Path

from open_instruct.miles.configuration import options, run_spec, validation
from open_instruct.miles.distillation import opd_config
from open_instruct.miles.errors import InputError
from open_instruct.miles.megatron_grpo import checkpoint_reuse, dppo_settings

# Megatron profile -> the HF ``model_type`` its learner checkpoint must declare. Only qwen3.5-2B
# has run on this route; the others need a mechanics run first (``plan`` warns).
ARCHITECTURES = {"qwen3-1.7B": "qwen3", "qwen3.5-2B": "qwen3_5", "qwen3.5-4B": "qwen3_5", "qwen3.5-9B": "qwen3_5"}
POLICY_LOSSES = dppo_settings.POLICY_LOSSES
DPPO_LOSS = dppo_settings.LOSS_FUNCTION
ASYNC_ROLLOUT = "open_instruct.miles.megatron_grpo.grpo_async.VerifierAsyncRollout"
REWARD_HOOK = "open_instruct.miles.megatron_grpo.grpo_hooks.reward"
POST_PROCESS_HOOK = "open_instruct.miles.megatron_grpo.grpo_hooks.post_process"
# Responses keep generating across a weight publication at the pinned MILES (the publication
# never aborts in-flight requests), so the old opt-in switch has no native counterpart.
KEEP_INFLIGHT = "keep_inflight_requests_on_weight_update"
# Set for the native driver process when training.eval_only selects the eval-only entry point.
EVAL_ONLY_ENV = "OI_GRPO_EVAL_ONLY"
DEFAULTS = {
    "model": {"source": "", "architecture": "qwen3-1.7B", "native_checkpoint": {}},
    # eval_only runs the pre-training eval of the starting model, then stops without training.
    "training": {**opd_config.DEFAULTS["training"], "algorithm": "grpo", "eval_only": False},
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
    # The registered-verifier gates of the Core backend (CoreConfig.reward_zero_truncated and
    # reward_final_answer_only). Both off: the whole response is graded, truncated or not.
    # exclude_truncated instead leaves length-truncated responses out of the loss and out of their
    # group's baseline (overlong filtering), keeping their raw reward for metrics.
    "reward": {"zero_truncated": False, "final_answer_only": False, "exclude_truncated": False},
    "output": dict(opd_config.DEFAULTS["output"]),
    "tracking": {**opd_config.DEFAULTS["tracking"], "wandb_project": "rl-backend-comparison"},
}
_TEACHER_FREE = "teacher-free verifier GRPO"
OWNED = {
    **{
        dest: owner
        for dest, owner in opd_config.OWNED_NATIVE_OPTIONS.items()
        # No teacher: the router timeout is an ordinary serving control here.
        if dest != "sglang_router_request_timeout_secs"
    },
    "advantage_estimator": "native GRPO advantages",
    "use_opd": _TEACHER_FREE,
    "opd_type": _TEACHER_FREE,
    "opd_log_prob_top_k": _TEACHER_FREE,
    "opd_kl_coef": _TEACHER_FREE,
    "opd_teacher_urls": _TEACHER_FREE,
    "opd_teacher_key": _TEACHER_FREE,
    "rm_url": "the registered verifier hooks",
    "custom_rm_path": "the registered verifier hooks",
    "custom_reward_post_process_path": "the registered verifier hooks (group centering)",
    "eval_function_path": "the registered verifier hooks",
    "optimizer": "the GRPO route (Adam)",
    "tensor_model_parallel_size": "trainer.tensor_parallel_size",
    "pipeline_model_parallel_size": "one-node GRPO topology",
    "context_parallel_size": "one-node GRPO topology",
    "expert_model_parallel_size": "dense Qwen topology",
    "expert_tensor_parallel_size": "dense Qwen topology",
    "micro_batch_size": "one-sample microbatches",
    "eps_clip": "objective.eps_clip",
    "eps_clip_high": "objective.eps_clip_high",
    "eps_clip_c": "objective.eps_clip_c",
    "grpo_std_normalization": "objective.std_normalization",
    "rewards_normalization": "group-centered GRPO rewards",
    "normalize_advantages": "no cross-group normalization",
    "compute_advantages_and_returns": "native GRPO reward-to-advantage computation",
    "kl_coef": _TEACHER_FREE,
    "use_kl_loss": _TEACHER_FREE,
    "kl_loss_coef": _TEACHER_FREE,
    "entropy_coef": "zero entropy bonus",
    "adam_eps": "optimizer.adam_eps",
    "clip_grad": "optimizer.clip_grad",
    "skip_actor_forward_only": "pre-update trainer-scored PPO anchor",
    "rollout_max_prompt_len": "inference.max_prompt_length",
    "data_source_path": "native fixed-fanout input source (the async ledger under miles.fully_async)",
    "custom_async_data_buffer_path": "the async verifier route",
    "rollout_function_path": "native synchronous rollout (the managed async producer under miles.fully_async)",
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
        model = document["model"]
        if not opd_config.is_local(validation.text(model["source"], "model.source")):
            raise InputError("Megatron GRPO requires a prepared local HF model.source directory")
        model["source"] = run_spec._path(model["source"], base.parent, "model.source")
        validation.choice(model["architecture"], "model.architecture", tuple(ARCHITECTURES))
        training, trainer, inf, opt = (document[k] for k in ("training", "trainer", "inference", "optimizer"))
        validation.choice(training["phase"], "training.phase", ("prepare", "train"))
        validation.boolean(training["resume"], "training.resume")
        validation.boolean(training["eval_only"], "training.eval_only")
        validation.integer(training["num_rollouts"], "training.num_rollouts")
        validation.integer(training["save_interval"], "training.save_interval")
        validation.integer(training["eval_interval"], "training.eval_interval", minimum=0)
        validation.integer(training["keep_checkpoints"], "training.keep_checkpoints", minimum=0)
        if training["keep_checkpoints"] != 0:
            raise InputError("Megatron GRPO retains all checkpoints; training.keep_checkpoints must be 0")
        if training["optimizer_steps_per_rollout"] != 1 or type(training["optimizer_steps_per_rollout"]) is not int:
            raise InputError("Megatron GRPO requires one optimizer step per rollout")
        validation.choice(training["loss_aggregation"], "training.loss_aggregation", opd_config.LOSS_AGGREGATIONS)
        if training["loss_aggregation"] == "domain" and document["objective"]["policy_loss"] != "dppo":
            raise InputError('training.loss_aggregation="domain" requires objective.policy_loss="dppo"')
        for section_name, section, keys in (
            ("trainer", trainer, ("gpus", "tensor_parallel_size")),
            (
                "inference",
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
                validation.integer(section[key], f"{section_name}.{key}")
            if section["gpus"] % section["tensor_parallel_size"]:
                raise InputError(f"{section_name}.gpus must be a multiple of {section_name}.tensor_parallel_size")
        if inf["samples_per_prompt"] < 2:
            raise InputError("GRPO requires at least two responses per prompt (inference.samples_per_prompt)")
        batch = inf["rollout_batch_size"] * inf["samples_per_prompt"]
        if batch % (trainer["gpus"] // trainer["tensor_parallel_size"]):
            raise InputError("GRPO collection size must divide evenly across trainer data-parallel ranks")
        validation.integer(inf["eval_max_response_length"], "inference.eval_max_response_length", minimum=0)
        if (
            inf["max_prompt_length"] + max(inf["max_response_length"], inf["eval_max_response_length"])
            > inf["max_context_length"]
        ):
            raise InputError("inference.max_context_length must cover max_prompt_length plus both response caps")
        validation.number(inf["temperature"], "inference.temperature", exclusive_min=True)
        for key in ("top_p", "eval_top_p"):
            validation.number(inf[key], f"inference.{key}", exclusive_min=True, maximum=1.0)
        validation.number(inf["eval_temperature"], "inference.eval_temperature")
        for key in ("learning_rate", "adam_eps", "clip_grad"):
            validation.number(opt[key], f"optimizer.{key}", exclusive_min=True)
        for key in ("min_lr", "weight_decay"):
            validation.number(opt[key], f"optimizer.{key}")
        if opt["min_lr"] > opt["learning_rate"]:
            raise InputError("optimizer.min_lr must not exceed optimizer.learning_rate")
        for key in ("adam_beta1", "adam_beta2"):
            validation.number(opt[key], f"optimizer.{key}", exclusive_max=True, maximum=1.0)
        validation.integer(opt["lr_warmup_iters"], "optimizer.lr_warmup_iters", minimum=0)
        validation.choice(opt["lr_decay_style"], "optimizer.lr_decay_style", opd_config.LR_DECAY_STYLES)
        objective = document["objective"]
        for key in ("eps_clip", "eps_clip_high", "eps_clip_c"):
            validation.number(objective[key], f"objective.{key}", exclusive_min=True)
        validation.boolean(objective["std_normalization"], "objective.std_normalization")
        validation.boolean(objective["use_rollout_logprobs"], "objective.use_rollout_logprobs")
        validation.choice(objective["policy_loss"], "objective.policy_loss", POLICY_LOSSES)
        validation.number(objective["dppo_threshold"], "objective.dppo_threshold", exclusive_min=True)
        try:
            dppo_settings.Settings(objective["dppo_divergence_type"], objective["dppo_threshold"])
        except (TypeError, ValueError) as error:
            raise InputError(f"objective.dppo_*: {error}") from None
        if objective["policy_loss"] == "dppo" and not objective["use_rollout_logprobs"]:
            raise InputError("objective.policy_loss=dppo requires objective.use_rollout_logprobs=true")
        if objective["use_rollout_logprobs"] and inf["top_p"] != 1.0:
            raise InputError("inference.top_p must be 1.0 when the rollout log-probs anchor the policy ratio")
        for key in ("zero_truncated", "final_answer_only", "exclude_truncated"):
            validation.boolean(document["reward"][key], f"reward.{key}")
        if document["reward"]["zero_truncated"] and document["reward"]["exclude_truncated"]:
            raise InputError("reward.zero_truncated and reward.exclude_truncated are mutually exclusive")
        for key in ("root", "assets"):
            document["output"][key] = run_spec._path(document["output"][key], base.parent, f"output.{key}")
        root, assets = (Path(document["output"][k]) for k in ("root", "assets"))
        if root == assets or root in assets.parents or assets in root.parents:
            raise InputError("output.root and output.assets must be separate, non-nested directories")
        checkpoint_reuse.validate(model["native_checkpoint"], model, trainer, document["output"])
        data = run_spec.RunSpec._data(document.get("data", {}), base.parent)
        if not data.get("prompt_data") or not data.get("eval_prompt_data") or not data.get("reward_config"):
            raise InputError(
                "Megatron GRPO requires pre-rendered data.prompt_data, data.eval_prompt_data name/path pairs and the "
                "data.reward_config verifier registry; scripts/miles/prepare_qwen35_math_prompts.py writes them"
            )
        document["data"] = data
        raw_native = document.get("miles", {})
        validation.mapping(raw_native, "[miles]")
        if any(key.replace("-", "_") == KEEP_INFLIGHT for key in raw_native):
            raise InputError(
                f"miles.{KEEP_INFLIGHT} is not a pinned MILES option: weight publication at the pinned MILES "
                "never aborts in-flight requests, so responses already continue across publications. Remove it."
            )
        native = options.normalize_options(raw_native)
        for key in native:
            if key in OWNED or key.startswith("opd_"):
                raise InputError(
                    f"miles.{key} is set by {OWNED.get(key, _TEACHER_FREE)}; change that control instead of the "
                    "native option"
                )
        for key in ("colocate", "offload_train", "offload_rollout", "partial_rollout"):
            if native.get(key, False):
                raise InputError(f"Megatron GRPO does not support miles.{key}")
        validation.boolean(native.get("fully_async", False), "miles.fully_async")
        if native.get("fully_async", False):
            # The async OPD producer and restart ledger, under the same contract as async OPD: one
            # update per rollout (above), a publication after every update, and behavior log-probs
            # from the engine version that sampled each token.
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
        if training["eval_only"] and not native.get("fully_async", False):
            raise InputError(
                "training.eval_only evaluates with the managed async producer; set miles.fully_async=true"
            )
        document["miles"] = native
        launch = run_spec.RunSpec._launch(
            {"auto_resume": False, "shared_memory": "64 GiB"} | document.get("launch", {}), base.parent
        )
        if launch["auto_resume"] and not training["resume"]:
            raise InputError(
                "launch.auto_resume requires training.resume so a re-queued job continues its checkpoints"
            )
        preparing = training["phase"] == "prepare"
        if preparing and launch["cluster"] not in ("ai2/saturn", "ai2/jupiter"):
            raise InputError("CPU-only Megatron GRPO preparation with WEKA must run on ai2/saturn or ai2/jupiter")
        total = trainer["gpus"] + inf["gpus"]
        if total > 8:
            raise InputError("Megatron GRPO runs in one node; trainer.gpus + inference.gpus must be <= 8")
        if not preparing and launch.get("gpus_per_replica", total) != total:
            raise InputError(
                f"Megatron GRPO allocates trainer.gpus + inference.gpus = {total} GPUs in one task; "
                "launch.gpus_per_replica disagrees"
            )
        document["launch"] = launch
        tracking = document["tracking"]
        validation.choice(tracking["wandb_mode"], "tracking.wandb_mode", opd_config.WANDB_MODES)
        validation.text(tracking["wandb_project"], "tracking.wandb_project")
        if not isinstance(tracking["wandb_entity"], str):
            raise InputError("tracking.wandb_entity must be a string (empty for the default entity)")
        if tracking["wandb_mode"] == "online" and "WANDB_API_KEY" not in launch["secrets"]:
            raise InputError('tracking.wandb_mode="online" requires launch.secrets.WANDB_API_KEY')
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

    @property
    def fully_async(self):
        return bool(self.document["miles"].get("fully_async", False))

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
        return dppo_settings.Settings(objective["dppo_divergence_type"], objective["dppo_threshold"])

    def plan(self):
        warnings = ["Experimental teacher-free verifier GRPO on Megatron; runtime qualification required."]
        if self.document["model"]["architecture"] != "qwen3.5-2B":
            warnings.append(
                f"{self.document['model']['architecture']} has not run on this route at the pinned MILES "
                "(the exercised recipe is qwen3.5-2B)."
            )
        if self.fully_async:
            warnings.append(
                "Responses keep generating across weight publications and may span policy versions; their age "
                "is measured from the oldest version that produced a token."
            )
        unmodeled = sorted(
            key
            for key in self.document["miles"]
            if key not in ("fully_async", "max_weight_staleness") and not key.startswith("async_")
        )
        if unmodeled:
            warnings.append("[miles] passes native options through unqualified: " + ", ".join(unmodeled))
        return {
            "name": self.name,
            "backend": "megatron",
            "algorithm": "grpo",
            "allocation": self.allocation(),
            "runtime_validated": False,
            "spec": self.to_dict(),
            "warnings": warnings,
        }
