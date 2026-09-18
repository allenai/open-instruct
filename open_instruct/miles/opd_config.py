"""CPU-only configuration for the Miles/Megatron sampled-token OPD route."""

import copy
import dataclasses
import re
from pathlib import Path

from open_instruct.miles import run_data, run_spec, validation
from open_instruct.miles.errors import InputError

REVISIONS = {
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    # EOPD (arXiv 2603.07079) replication: Qwen3-Base students, Qwen3-8B teacher.
    "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
    "Qwen/Qwen3-4B-Base": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}
# Megatron architecture profiles: Miles ships scripts/models/<profile>.py; profiles
# under open_instruct/miles/model_profiles/ take precedence at runtime.
PROFILES = ("qwen3.5-2B", "qwen3.5-4B", "qwen3.5-9B", "qwen3-1.7B", "qwen3-4B", "qwen3-8B")
ARCHITECTURES = {
    "Qwen/Qwen3.5-2B": "qwen3.5-2B",
    "Qwen/Qwen3.5-4B": "qwen3.5-4B",
    "Qwen/Qwen3.5-9B": "qwen3.5-9B",
    "Qwen/Qwen3-1.7B-Base": "qwen3-1.7B",
    "Qwen/Qwen3-4B-Base": "qwen3-4B",
    "Qwen/Qwen3-8B": "qwen3-8B",
}
LR_DECAY_STYLES = ("constant", "cosine", "linear")
WANDB_MODES = ("offline", "online", "disabled")
DEFAULTS = {
    # align_eos_with_teacher: give a base-model learner the teacher's end-of-turn token as its
    # eos (Qwen3-Base stops on <|endoftext|>, Qwen3-8B on <|im_end|>) so rollouts stop where the
    # teacher's answers end and the special-token identity check passes; the vocabulary must match.
    "model": {"source": "Qwen/Qwen3.5-4B", "revision": "", "architecture": "", "align_eos_with_teacher": False},
    "teacher": {
        "source": "Qwen/Qwen3.5-9B",
        "revision": "",
        "gpus": 1,
        "concurrency": 4,
        "request_timeout": 180,
        "startup_timeout": 900,
    },
    "training": {
        "algorithm": "opd",
        "phase": "train",
        "num_rollouts": 2,
        "save_interval": 1,
        "eval_interval": 0,
        "resume": False,
        # Optimizer steps per rollout: Miles' global batch is the rollout batch divided by
        # this, so 4 gives the paper-style "batch 128, mini-batch 32" PPO schedule.
        "optimizer_steps_per_rollout": 1,
    },
    "trainer": {"backend": "megatron", "gpus": 2, "tensor_parallel_size": 2},
    "inference": {
        "gpus": 1,
        "tensor_parallel_size": 1,
        "rollout_batch_size": 4,
        "samples_per_prompt": 2,
        "max_response_length": 256,
        "max_context_length": 2048,
        "max_running_requests": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "eval_temperature": 0.0,
        "eval_top_p": 1.0,
        "eval_samples_per_prompt": 1,
        # 0 evaluates with max_response_length; a larger cap needs a matching max_context_length.
        "eval_max_response_length": 0,
    },
    "distillation": {
        "kl_coef": 1.0,
        "log_prob_top_k": 0,
        "task_reward_weight": 0.0,
        "use_rollout_logprobs": False,
        # EOPD (arXiv 2603.07079): add alpha * 1[H_teacher > tau] * FKL over the teacher's top-k
        # to the sampled-token OPD loss. Paper defaults alpha 1.0, tau 0.8, k 16.
        "eopd": False,
        "eopd_alpha": 1.0,
        "eopd_tau": 0.8,
        "eopd_top_k": 16,
    },
    "optimizer": {"learning_rate": 1e-6, "lr_decay_style": "constant", "lr_warmup_iters": 0, "min_lr": 0.0},
    "output": {"root": "", "assets": ""},
    "tracking": {"wandb_mode": "offline", "wandb_project": "open-instruct-opd", "wandb_entity": ""},
}


def is_local(source):
    return source.startswith(("/", ".", "~"))


def _resolve_source(values, role, base):
    """Accept a Hugging Face repository at an immutable revision or a local checkpoint directory."""
    source = validation.text(values["source"], f"{role}.source")
    revision = values["revision"]
    if revision is not None and not isinstance(revision, str):
        raise InputError(f"{role}.revision must be a string commit hash")
    if is_local(source):
        if revision:
            raise InputError(f"Local {role}.source must not specify revision")
        values["source"] = str((base / Path(source).expanduser()).resolve())
        values["revision"] = ""
        return
    if not revision:
        revision = REVISIONS.get(source, "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise InputError(f"Remote {role}.source requires an immutable 40-character {role}.revision")
    values["revision"] = revision


@dataclasses.dataclass
class OPDRunSpec:
    document: dict
    config_path: Path

    @classmethod
    def from_dict(cls, document, *, config_path=None, overrides=None):
        document = copy.deepcopy(document)
        run_spec._apply_overrides(document, overrides)
        validation.fields(document, "run", set(DEFAULTS) | {"schema_version", "name", "data", "launch"})
        if document.get("schema_version") != 1:
            raise InputError("OPD requires schema_version=1")
        name = validation.text(document.get("name"), "name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise InputError("name must contain only letters, digits, '.', '_' and '-'")
        base = Path(config_path or "run.toml").resolve()
        for section, defaults in DEFAULTS.items():
            incoming = document.get(section, {})
            validation.mapping(incoming, section)
            validation.fields(incoming, section, set(defaults))
            document[section] = defaults | incoming
        for role in ("model", "teacher"):
            _resolve_source(document[role], role, base.parent)
        model = document["model"]
        architecture = model["architecture"] or ARCHITECTURES.get(model["source"], "")
        if not architecture:
            raise InputError(
                f"model.architecture must name a Megatron profile ({', '.join(PROFILES)}) for {model['source']}"
            )
        model["architecture"] = validation.choice(architecture, "model.architecture", PROFILES)
        for key in ("root", "assets"):
            document["output"][key] = run_spec._path(document["output"][key], base.parent, f"output.{key}")
        root, assets = (Path(document["output"][key]) for key in ("root", "assets"))
        if root == assets or root in assets.parents or assets in root.parents:
            raise InputError("output.root and output.assets must be separate, non-nested directories")
        for section, keys in {
            "teacher": ("gpus", "concurrency", "request_timeout", "startup_timeout"),
            "trainer": ("gpus", "tensor_parallel_size"),
            "training": ("num_rollouts", "save_interval", "optimizer_steps_per_rollout"),
            "inference": (
                "gpus",
                "tensor_parallel_size",
                "rollout_batch_size",
                "samples_per_prompt",
                "max_response_length",
                "max_context_length",
                "max_running_requests",
                "eval_samples_per_prompt",
            ),
        }.items():
            for key in keys:
                validation.integer(document[section][key], f"{section}.{key}")
        validation.integer(document["training"]["eval_interval"], "training.eval_interval", minimum=0)
        validation.integer(
            document["inference"]["eval_max_response_length"], "inference.eval_max_response_length", minimum=0
        )
        if document["inference"]["eval_max_response_length"] >= document["inference"]["max_context_length"]:
            raise InputError("max_context_length must exceed eval_max_response_length")
        validation.integer(document["optimizer"]["lr_warmup_iters"], "optimizer.lr_warmup_iters", minimum=0)
        validation.number(document["optimizer"]["min_lr"], "optimizer.min_lr")
        if document["optimizer"]["min_lr"] > document["optimizer"]["learning_rate"]:
            raise InputError("optimizer.min_lr must not exceed optimizer.learning_rate")
        validation.choice(document["optimizer"]["lr_decay_style"], "optimizer.lr_decay_style", LR_DECAY_STYLES)
        inference = document["inference"]
        collection = inference["rollout_batch_size"] * inference["samples_per_prompt"]
        if collection % document["training"]["optimizer_steps_per_rollout"]:
            raise InputError(
                "training.optimizer_steps_per_rollout must divide inference.rollout_batch_size * "
                "inference.samples_per_prompt"
            )
        for key in ("top_p", "eval_top_p"):
            validation.number(inference[key], f"inference.{key}", exclusive_min=True, maximum=1.0)
        if inference["top_p"] != 1.0 and document["distillation"]["use_rollout_logprobs"]:
            raise InputError("inference.top_p must be 1.0 when the rollout log-probs are the student side")
        for section, key, expected in (
            ("training", "algorithm", "opd"),
            ("trainer", "backend", "megatron"),
            ("distillation", "log_prob_top_k", 0),
            ("distillation", "task_reward_weight", 0.0),
        ):
            if document[section][key] != expected:
                raise InputError(f"Miles OPD requires {section}.{key}={expected!r}")
        for section, key in (("trainer", "gpus"), ("inference", "gpus")):
            if document[section][key] % document[section]["tensor_parallel_size"]:
                raise InputError(f"{section}.gpus must be a multiple of {section}.tensor_parallel_size")
        validation.choice(document["training"]["phase"], "training.phase", ("prepare", "train"))
        validation.boolean(document["training"]["resume"], "training.resume")
        validation.boolean(document["model"]["align_eos_with_teacher"], "model.align_eos_with_teacher")
        validation.boolean(document["distillation"]["use_rollout_logprobs"], "distillation.use_rollout_logprobs")
        distillation = document["distillation"]
        validation.boolean(distillation["eopd"], "distillation.eopd")
        validation.integer(distillation["eopd_top_k"], "distillation.eopd_top_k")
        validation.number(distillation["eopd_alpha"], "distillation.eopd_alpha", exclusive_min=True)
        validation.number(distillation["eopd_tau"], "distillation.eopd_tau")
        if distillation["eopd"] and not distillation["use_rollout_logprobs"]:
            raise InputError(
                "distillation.eopd requires distillation.use_rollout_logprobs (the paper's SGLang OPD path)"
            )
        for section, key in (
            ("distillation", "kl_coef"),
            ("optimizer", "learning_rate"),
            ("inference", "temperature"),
        ):
            validation.number(document[section][key], f"{section}.{key}", exclusive_min=True)
        validation.number(document["inference"]["eval_temperature"], "inference.eval_temperature")
        if document["inference"]["max_response_length"] >= document["inference"]["max_context_length"]:
            raise InputError("max_context_length must exceed max_response_length")
        tracking = document["tracking"]
        validation.choice(tracking["wandb_mode"], "tracking.wandb_mode", WANDB_MODES)
        validation.text(tracking["wandb_project"], "tracking.wandb_project")
        if not isinstance(tracking["wandb_entity"], str):
            raise InputError("tracking.wandb_entity must be a string (empty for the default entity)")
        document["data"] = data = run_spec.RunSpec._data(document.get("data", {}), base.parent)
        if "tasks" in data:
            if len(data["tasks"]) != 1 or not data["tasks"][0].get("eval_count"):
                raise InputError("OPD data.tasks must hold one registered task with a held-out eval_count")
            validation.choice(data["tasks"][0]["task"], "data.tasks[0].task", sorted(run_data.TASKS))
        elif "prompt_data" in data:
            if not data.get("eval_prompt_data") or "reward_config" not in data:
                raise InputError(
                    "OPD data.prompt_data (pre-rendered prompts) requires data.eval_prompt_data name/path pairs and "
                    "the data.reward_config verifier registry; scripts/miles/prepare_qwen35_math_prompts.py writes them"
                )
        else:
            raise InputError("OPD data must select tasks or pre-rendered prompt_data")
        document["launch"] = run_spec.RunSpec._launch(
            {"auto_resume": False, "shared_memory": "64 GiB"} | document.get("launch", {}), base.parent
        )
        if document["launch"]["auto_resume"] and not document["training"]["resume"]:
            raise InputError(
                "launch.auto_resume requires training.resume so a re-queued job continues its checkpoints"
            )
        preparing = document["training"]["phase"] == "prepare"
        if preparing and document["launch"]["cluster"] != "ai2/saturn":
            raise InputError("CPU-only OPD preparation with WEKA must run on ai2/saturn")
        total = sum(document[section]["gpus"] for section in ("trainer", "inference", "teacher"))
        if not preparing and document["launch"].get("gpus_per_replica", total) != total:
            raise InputError(
                f"OPD training allocates trainer.gpus + inference.gpus + teacher.gpus = {total} GPUs in one task; "
                f"launch.gpus_per_replica disagrees"
            )
        if total > 8:
            raise InputError("Miles OPD runs in one node; trainer.gpus + inference.gpus + teacher.gpus must be <= 8")
        if tracking["wandb_mode"] == "online" and "WANDB_API_KEY" not in document["launch"]["secrets"]:
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

    def to_dict(self):
        return copy.deepcopy(self.document)

    def allocation(self):
        if self.document["training"]["phase"] == "prepare":
            return {"replicas": 1, "gpus_per_replica": 0, "ray_gpus": 0, "roles": {}}
        trainer, student, teacher = (self.document[section]["gpus"] for section in ("trainer", "inference", "teacher"))
        return {
            "replicas": 1,
            "gpus_per_replica": trainer + student + teacher,
            "ray_gpus": trainer + student,
            "roles": {
                "trainer": list(range(trainer)),
                "student": list(range(trainer, trainer + student)),
                "teacher": list(range(trainer + student, trainer + student + teacher)),
            },
        }

    def plan(self):
        exercised = self.allocation()["roles"] == {"trainer": [0, 1], "student": [2], "teacher": [3]}
        warnings = ["Experimental sampled-token OPD; runtime qualification required."]
        if not exercised:
            warnings.append("Only the 2 trainer / 1 student / 1 teacher GPU topology has been exercised.")
        return {
            "name": self.name,
            "backend": "megatron",
            "algorithm": "opd",
            "allocation": self.allocation(),
            "runtime_validated": False,
            "spec": self.to_dict(),
            "warnings": warnings,
        }
