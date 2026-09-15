"""CPU-safe configuration for OPD through the registered OLMo-core backend."""

import copy
import re
from pathlib import Path

from open_instruct.miles import validation
from open_instruct.miles.errors import InputError

ENV = "OI_MILES_OPD_TEACHER"


def parse(document, base):
    algorithm = document.get("training", {}).get("algorithm", "grpo")
    validation.choice(algorithm, "training.algorithm", ("grpo", "opd"))
    validation.choice(document.get("trainer", {}).get("backend", "olmo_core"), "trainer.backend", ("olmo_core",))
    teacher = copy.deepcopy(document.get("teacher", {}))
    distillation = copy.deepcopy(document.get("distillation", {}))
    if algorithm != "opd":
        if teacher or distillation:
            raise InputError("teacher/distillation require training.algorithm=opd")
        return {}, {}
    validation.mapping(teacher, "teacher")
    defaults = dict(
        gpus=1,
        tensor_parallel_size=1,
        concurrency=4,
        request_timeout=180,
        startup_timeout=1800,
        max_context_length=8192,
        chat_template_kwargs={},
        attention_backend="triton",
    )
    validation.fields(teacher, "teacher", set(defaults) | {"source", "revision"})
    teacher = defaults | teacher
    source = validation.text(teacher.get("source"), "teacher.source")
    if source.startswith(("/", ".", "~")):
        teacher["source"] = str((base / Path(source).expanduser()).resolve())
        if teacher.get("revision"):
            raise InputError("Local teacher.source must not specify revision")
    elif not re.fullmatch(r"[0-9a-f]{40}", teacher.get("revision", "")):
        raise InputError("Remote teacher.source requires an immutable 40-character revision")
    for key in (
        "gpus",
        "tensor_parallel_size",
        "concurrency",
        "request_timeout",
        "startup_timeout",
        "max_context_length",
    ):
        validation.integer(teacher[key], f"teacher.{key}")
    if teacher["gpus"] != teacher["tensor_parallel_size"]:
        raise InputError("teacher.gpus must equal teacher.tensor_parallel_size for one teacher engine")
    validation.mapping(teacher["chat_template_kwargs"], "teacher.chat_template_kwargs")
    validation.text(teacher["attention_backend"], "teacher.attention_backend")
    validation.fields(distillation, "distillation", {"alignment", "kl_coef"})
    distillation = {"alignment": "shared_token_ids", "kl_coef": 1.0} | distillation
    validation.choice(distillation["alignment"], "distillation.alignment", ("shared_token_ids", "exact_text_spans"))
    validation.number(distillation["kl_coef"], "distillation.kl_coef")
    if distillation["kl_coef"] <= 0:
        raise InputError("distillation.kl_coef must be positive")
    return teacher, distillation


def compile_options(spec, miles, core):
    if not spec.teacher:
        return
    if miles.get("fully_async") or core.get("publication_mode", "barrier") != "barrier":
        raise InputError("Core OPD currently requires synchronous barrier publication")
    if miles.get("use_rollout_logprobs"):
        raise InputError("Core OPD requires pre-update trainer scores; use_rollout_logprobs=false")
    required = {
        "normalize_advantages": False,
        "kl_coef": 0.0,
        "use_opd": True,
        "opd_type": "sglang",
        "opd_kl_coef": spec.distillation["kl_coef"],
        "opd_log_prob_top_k": 0,
        "custom_rm_path": "open_instruct.miles.core_opd_hooks.reward",
        "custom_reward_post_process_path": "open_instruct.miles.core_opd_hooks.post_process",
        "eval_function_path": "open_instruct.miles.core_opd_hooks.evaluate",
    }
    for key, value in required.items():
        if key in miles and miles[key] != value and key != "custom_rm_path":
            raise InputError(f"Core OPD owns miles.{key}")
        if key == "custom_rm_path" and miles.get(key) not in (
            None,
            value,
            "open_instruct.miles.rewards.registered_reward",
        ):
            raise InputError("Core OPD supplies its teacher scoring reward")
        miles[key] = value
