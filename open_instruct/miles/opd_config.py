"""CPU-only configuration for the bounded Miles/Megatron OPD prototype."""

import copy
import dataclasses
import re
from pathlib import Path

from open_instruct.miles import run_spec, validation
from open_instruct.miles.errors import InputError

REVISIONS = {
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}
DEFAULTS = {
    "model": {"source": "Qwen/Qwen3.5-4B", "revision": REVISIONS["Qwen/Qwen3.5-4B"]},
    "teacher": {
        "source": "Qwen/Qwen3.5-9B",
        "revision": REVISIONS["Qwen/Qwen3.5-9B"],
        "gpus": 1,
        "concurrency": 4,
        "request_timeout": 180,
        "startup_timeout": 900,
    },
    "training": {"algorithm": "opd", "phase": "train", "num_rollouts": 2, "save_interval": 1, "resume": False},
    "trainer": {"backend": "megatron", "gpus": 2, "tensor_parallel_size": 2},
    "inference": {
        "gpus": 1,
        "rollout_batch_size": 4,
        "samples_per_prompt": 2,
        "max_response_length": 256,
        "max_context_length": 2048,
        "temperature": 1.0,
    },
    "distillation": {"kl_coef": 1.0, "log_prob_top_k": 0, "task_reward_weight": 0.0},
    "optimizer": {"learning_rate": 1e-6},
    "output": {"root": "", "assets": ""},
    "tracking": {"wandb_mode": "offline"},
}


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
            values = document[role]
            expected = "Qwen/Qwen3.5-4B" if role == "model" else "Qwen/Qwen3.5-9B"
            validation.choice(values["source"], f"{role}.source", (expected,))
            if not re.fullmatch(r"[0-9a-f]{40}", values["revision"]):
                raise InputError(f"{role}.revision must be an immutable HF commit")
        for key in ("root", "assets"):
            document["output"][key] = run_spec._path(document["output"][key], base.parent, f"output.{key}")
        root, assets = (Path(document["output"][key]) for key in ("root", "assets"))
        if root == assets or root in assets.parents or assets in root.parents:
            raise InputError("output.root and output.assets must be separate, non-nested directories")
        for section, keys in {
            "teacher": ("gpus", "concurrency", "request_timeout", "startup_timeout"),
            "trainer": ("gpus", "tensor_parallel_size"),
            "training": ("num_rollouts", "save_interval"),
            "inference": (
                "gpus",
                "rollout_batch_size",
                "samples_per_prompt",
                "max_response_length",
                "max_context_length",
            ),
        }.items():
            for key in keys:
                validation.integer(document[section][key], f"{section}.{key}")
        for section, key, expected in (
            ("training", "algorithm", "opd"),
            ("training", "save_interval", 1),
            ("trainer", "backend", "megatron"),
            ("teacher", "gpus", 1),
            ("trainer", "gpus", 2),
            ("trainer", "tensor_parallel_size", 2),
            ("inference", "gpus", 1),
            ("distillation", "log_prob_top_k", 0),
            ("distillation", "task_reward_weight", 0.0),
            ("tracking", "wandb_mode", "offline"),
        ):
            if document[section][key] != expected:
                raise InputError(f"Prototype requires {section}.{key}={expected!r}")
        validation.choice(document["training"]["phase"], "training.phase", ("prepare", "train"))
        validation.boolean(document["training"]["resume"], "training.resume")
        if document["training"]["resume"]:
            raise InputError("Resume is not yet qualified for the OPD prototype; use a fresh run")
        for section, key in (
            ("distillation", "kl_coef"),
            ("optimizer", "learning_rate"),
            ("inference", "temperature"),
        ):
            validation.number(document[section][key], f"{section}.{key}", exclusive_min=True)
        if document["inference"]["max_response_length"] >= document["inference"]["max_context_length"]:
            raise InputError("max_context_length must exceed max_response_length")
        document["data"] = run_spec.RunSpec._data(document.get("data", {}), base.parent)
        tasks = document["data"].get("tasks", [])
        if len(tasks) != 1 or tasks[0]["task"] != "gsm8k" or not tasks[0].get("eval_count"):
            raise InputError("OPD prototype requires one GSM8K task with held-out eval_count")
        document["launch"] = run_spec.RunSpec._launch(
            {"auto_resume": False, "shared_memory": "64 GiB"} | document.get("launch", {}), base.parent
        )
        if document["launch"]["auto_resume"]:
            raise InputError("Automatic restart is not supported for the OPD prototype")
        preparing = document["training"]["phase"] == "prepare"
        if preparing and document["launch"]["cluster"] != "ai2/saturn":
            raise InputError("CPU-only OPD preparation with WEKA must run on ai2/saturn")
        if not preparing and document["launch"].get("gpus_per_replica", 4) != 4:
            raise InputError("OPD training allocates four GPUs in one task")
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
        preparing = self.document["training"]["phase"] == "prepare"
        return {
            "replicas": 1,
            "gpus_per_replica": 0 if preparing else 4,
            "ray_gpus": 0 if preparing else 3,
            "roles": {} if preparing else {"trainer": [0, 1], "student": [2], "teacher": [3]},
        }

    def plan(self):
        return {
            "name": self.name,
            "backend": "megatron",
            "algorithm": "opd",
            "allocation": self.allocation(),
            "runtime_validated": False,
            "spec": self.to_dict(),
            "warnings": ["Experimental 4B <- 9B sampled-token OPD; runtime qualification required."],
        }
