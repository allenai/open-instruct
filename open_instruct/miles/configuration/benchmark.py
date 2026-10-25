"""Explicit, bounded fixed-policy inference measurements without trainer GPUs."""

import re

from open_instruct.miles.configuration import validation
from open_instruct.miles.errors import InputError

# Keep a panel's model, precision, sampling and output contract fixed. Native
# SGLang options include much more than scheduling (including dtype and TP).
SERVING_CONTROLS = {
    "sglang_server_concurrency",
    "sglang_mem_fraction_static",
    "sglang_max_running_requests",
    "sglang_max_queued_requests",
    "sglang_max_total_tokens",
    "sglang_chunked_prefill_size",
    "sglang_max_prefill_tokens",
    "sglang_schedule_policy",
    "sglang_schedule_conservativeness",
    "sglang_disable_radix_cache",
    "sglang_disable_overlap_schedule",
    "sglang_scheduler_recv_interval",
    "sglang_enable_mixed_chunk",
    "sglang_cuda_graph_backend_decode",
    "sglang_cuda_graph_backend_prefill",
    "sglang_cuda_graph_max_bs_decode",
    "sglang_cuda_graph_max_bs_prefill",
    "sglang_cuda_graph_bs_decode",
    "sglang_cuda_graph_bs_prefill",
    "sglang_cuda_graph_tc_compiler",
    "sglang_max_mamba_cache_size",
    "sglang_mamba_max_states_per_path",
    "sglang_mamba_radix_cache_strategy",
    "sglang_mamba_track_interval",
}
QUEUE_CONTROLS = {
    "inference.radix_cache",
    "inference.mamba_radix_cache_strategy",
    "inference.enable_mixed_chunk",
    "miles.miles_router_max_connections",
    "async.async_max_concurrent_samples",
    "async.async_data_buffer_capacity_factor",
    "async.rollout_submission_granularity",
}


def parse(value):
    defaults = {
        "warmup_seconds": 120.0,
        "measure_seconds": 300.0,
        "collection_timeout": 900.0,
        "drain_timeout": 120.0,
        "capture_samples": 8,
        "reward_mode": "none",
        "variants": [],
        "replay_reference": None,
        "replay_repeats": 2,
        "replay_copies": 1,
        "replay_background": False,
        "score_reference": False,
    }
    validation.fields(value, "benchmark", set(defaults))
    result = defaults | value
    for key in ("warmup_seconds", "measure_seconds", "collection_timeout", "drain_timeout"):
        validation.number(result[key], f"benchmark.{key}", exclusive_min=key != "warmup_seconds")
    validation.integer(result["capture_samples"], "benchmark.capture_samples", minimum=0)
    validation.integer(result["replay_repeats"], "benchmark.replay_repeats", minimum=1)
    validation.integer(result["replay_copies"], "benchmark.replay_copies", minimum=1)
    validation.boolean(result["score_reference"], "benchmark.score_reference")
    validation.boolean(result["replay_background"], "benchmark.replay_background")
    if result["replay_reference"] is not None:
        validation.text(result["replay_reference"], "benchmark.replay_reference")
        if not result["replay_reference"].startswith("/"):
            raise InputError("benchmark.replay_reference must be an absolute completed benchmark directory")
    if result["reward_mode"] not in ("none", "configured"):
        raise InputError("benchmark.reward_mode must be none or configured")
    variants = result["variants"]
    if not isinstance(variants, list):
        raise InputError("benchmark.variants must be an array of name/overrides tables")
    names = set()
    for variant in variants:
        validation.mapping(variant, "benchmark variant")
        validation.fields(variant, "benchmark variant", {"name", "overrides"})
        name = variant.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise InputError("Benchmark variant names must be unique letters/digits/hyphens/underscores")
        names.add(name)
        overrides = variant.get("overrides", [])
        if not isinstance(overrides, list):
            raise InputError("Benchmark variant overrides must be a list of SECTION.KEY=VALUE strings")
        for override in overrides:
            if not isinstance(override, str) or "=" not in override:
                raise InputError("Benchmark variant overrides must be SECTION.KEY=VALUE strings")
            key = override.partition("=")[0]
            section, _, option = key.partition(".")
            if key not in QUEUE_CONTROLS and not (section in {"inference", "miles"} and option in SERVING_CONTROLS):
                raise InputError(f"Benchmark variants may change serving/queue controls only, not {key}")
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
    if spec.benchmark["score_reference"] and (
        not spec.benchmark["replay_reference"] or miles["rollout_num_gpus"] != 1 or spec.benchmark["variants"]
    ):
        raise InputError("Core reference scoring requires replay_reference, one GPU and no variants")
    if (
        spec.benchmark["replay_reference"]
        and not spec.benchmark["score_reference"]
        and not miles.get("sglang_enable_custom_logit_processor")
    ):
        raise InputError("Diagnostic replay requires miles.sglang_enable_custom_logit_processor=true")
