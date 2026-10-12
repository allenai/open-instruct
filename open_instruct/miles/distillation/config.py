"""Configure a frozen scoring teacher without changing the Core trainer."""

import copy
from pathlib import Path

from open_instruct.miles.configuration import validation
from open_instruct.miles.errors import InputError

ENV = "OI_MILES_OPD_TEACHER"
HOOK = "open_instruct.miles.distillation.hooks"


def parse(document, base):
    teacher = copy.deepcopy(document.get("teacher", {}))
    if not teacher:
        return {}
    defaults = dict(
        gpus=1,
        tensor_parallel_size=1,
        concurrency=4,
        request_timeout=600,
        startup_timeout=1800,
        max_context_length=65536,
        kl_coef=1.0,
        objective="full",
        pressure_clip=None,
        calibrate_initial=False,
    )
    validation.mapping(teacher, "teacher")
    validation.fields(teacher, "teacher", set(defaults) | {"source"})
    teacher = defaults | teacher
    source = validation.text(teacher.get("source"), "teacher.source")
    teacher["source"] = str((base / Path(source).expanduser()).resolve())
    for name in set(defaults) - {"kl_coef", "objective", "pressure_clip", "calibrate_initial"}:
        validation.integer(teacher[name], f"teacher.{name}")
    if not isinstance(teacher["calibrate_initial"], bool):
        raise InputError("teacher.calibrate_initial must be boolean")
    validation.number(teacher["kl_coef"], "teacher.kl_coef", exclusive_min=True)
    if teacher["objective"] not in {"full", "cancel_answer_positive"}:
        raise InputError("teacher.objective must be full or cancel_answer_positive")
    if teacher["pressure_clip"] is not None:
        validation.number(teacher["pressure_clip"], "teacher.pressure_clip", exclusive_min=True)
        if teacher["objective"] != "cancel_answer_positive":
            raise InputError("pressure_clip belongs to cancel_answer_positive")
    if teacher["gpus"] != teacher["tensor_parallel_size"]:
        raise InputError("teacher.gpus must equal teacher.tensor_parallel_size")
    return teacher


def compile_options(spec, miles, core):
    if not spec.teacher:
        if miles.get("use_opd"):
            raise InputError("Core OPD requires a configured frozen teacher")
        return
    if miles.get("dynamic_sampling_filter_path") or core.get("filter_zero_std_groups"):
        raise InputError("Distillation retains zero-reward groups; disable zero-std filtering")
    if spec.launch["auto_resume"]:
        raise InputError("Frozen-teacher pilot requires explicit restart; auto_resume=false")
    required = {
        "normalize_advantages": False,
        "kl_coef": 0.0,
        "use_opd": True,
        "opd_type": "sglang",
        "opd_kl_coef": spec.teacher["kl_coef"],
        "opd_log_prob_top_k": 0,
        "custom_rm_path": HOOK + ".reward",
        "custom_reward_post_process_path": HOOK + ".post_process",
        "eval_function_path": HOOK + ".evaluate",
    }
    for key, value in required.items():
        if (
            key in miles
            and miles[key] != value
            and (key != "custom_rm_path" or miles[key] != "open_instruct.miles.rewards.rewards.registered_reward")
        ):
            raise InputError(f"Frozen-context OPD owns miles.{key}")
        miles[key] = value
    core["scoring_pass_required"] = True


def validate(miles, core):
    if not miles.get("use_opd"):
        return
    if miles.get("opd_type") != "sglang" or miles.get("opd_log_prob_top_k", 0):
        raise InputError("Core OPD supports external SGLang sampled-token scoring only")
    if miles.get("fully_async") or core.publication_mode != "barrier" or miles.get("use_rollout_logprobs"):
        raise InputError("Core OPD validation requires synchronous publication and trainer-scored probabilities")
    if not core.scoring_pass_required or miles.get("normalize_advantages") or miles.get("kl_coef", 0):
        raise InputError("Pure OPD requires pre-update scores, unnormalized advantages, and zero reference KL")
    if (
        miles.get("custom_rm_path") != HOOK + ".reward"
        or miles.get("custom_reward_post_process_path") != HOOK + ".post_process"
    ):
        raise InputError("Core OPD requires aligned teacher scoring and zero task-reward advantages")
    if miles.get("rollout_temperature", 1.0) != 1.0 or miles.get("rollout_top_p", 1.0) != 1.0:
        raise InputError("OPD validation requires temperature=1 and top_p=1 for raw-probability sampling")
