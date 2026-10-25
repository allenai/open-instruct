"""Explicit, bounded fixed-policy inference measurements without trainer GPUs."""

from open_instruct.miles.configuration import validation
from open_instruct.miles.errors import InputError


def parse(value):
    defaults = {
        "warmup_seconds": 120.0,
        "measure_seconds": 300.0,
        "collection_timeout": 900.0,
        "drain_timeout": 120.0,
        "capture_samples": 8,
        "reward_mode": "none",
    }
    validation.fields(value, "benchmark", set(defaults))
    result = defaults | value
    for key in ("warmup_seconds", "measure_seconds", "collection_timeout", "drain_timeout"):
        validation.number(result[key], f"benchmark.{key}", exclusive_min=key != "warmup_seconds")
    validation.integer(result["capture_samples"], "benchmark.capture_samples", minimum=0)
    if result["reward_mode"] not in ("none", "configured"):
        raise InputError("benchmark.reward_mode must be none or configured")
    return result


def validate(spec, config):
    if not spec.benchmark:
        return
    miles = config.miles
    if not miles["fully_async"] or miles["colocate"] or miles["rollout_num_gpus_per_engine"] != 1:
        raise InputError("Inference benchmarks require fully_async, disaggregated placement and TP1")
    if miles["rollout_num_gpus"] > 8:
        raise InputError("Inference benchmarks currently support at most eight GPUs on one node")
    if config.core.publication_mode == "engine_drain":
        raise InputError("Inference benchmarks support barrier or refresh request paths, not engine_drain")
    if spec.judges["judging"]["bindings"] or spec.evaluation["mode"] == "background":
        raise InputError("Inference benchmarks do not launch managed judges or background evaluation")
    if spec.output["export_hf"] or miles.get("save_interval") or miles.get("eval_interval"):
        raise InputError("Inference benchmarks require save_checkpoints=false, no eval_interval and export_hf=false")
    if spec.launch["auto_resume"]:
        raise InputError(
            "Inference benchmarks require launch.auto_resume=false; each attempt is an independent measurement"
        )
    if spec.benchmark["reward_mode"] == "none" and (
        config.core.filter_zero_std_groups or miles.get("dynamic_sampling_filter_path")
    ):
        raise InputError("benchmark.reward_mode=none requires filter_zero_std_groups=false and no reward filter")
    if miles.get("load") or miles.get("load_debug_rollout_data") or config.core.expert_balanced_packing:
        raise InputError(
            "Inference benchmarks cannot resume training, replay saved rollouts or prepare trainer packing"
        )
    if miles.get("sglang_weight_version", "0") != "0":
        raise InputError('Inference benchmarks use the fixed checkpoint version sglang_weight_version="0"')
