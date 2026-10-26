"""Measured serving starting points, selected without importing a GPU runtime.

Only checkpoint metadata is read. Unknown shapes keep the generic defaults;
these profiles describe a measured workload, not a universal memory estimator.
"""

import hashlib
import json
from pathlib import Path

from open_instruct.miles.errors import InputError
from open_instruct.miles.infrastructure import gpu_hardware

MEASURED_CONTEXT = 34816
MEASURED_RESPONSE = 32768
MINIMUM_MEMORY_MIB = {"b300": 270000, "h100": 80000}
# Conservative lower bounds for MODEL_SHAPE only: weights rounded down, plus
# attention KV and FP32 recurrent matrices; convolution/workspace/graphs add more.
WEIGHTS_LOWER_BOUND_MIB = 23 * 1024
KV_BYTES_PER_TOKEN = 4096
RECURRENT_MATRIX_MIB_PER_SLOT = 14

PROFILES = {"olmo35-b300-v1": "b300", "olmo35-h100-v1": "h100"}
PROFILE_KEYS = (
    "sglang_disable_radix_cache",
    "sglang_mamba_radix_cache_strategy",
    "sglang_max_running_requests",
    "sglang_server_concurrency",
    "sglang_cuda_graph_backend_decode",
    "sglang_cuda_graph_max_bs_decode",
    "sglang_cuda_graph_backend_prefill",
    "sglang_max_total_tokens",
    "sglang_max_mamba_cache_size",
    "sglang_mem_fraction_static",
    "sglang_chunked_prefill_size",
)
# Parameter and cache geometry of the measured checkpoint. Do not match only
# model_type: that would apply these large pools to much larger Olmo models.
MODEL_SHAPE = {
    "model_type": "olmo3moe",
    "hidden_size": 1024,
    "num_hidden_layers": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "attention_hidden_size": 1024,
    "vocab_size": 100278,
    "tie_word_embeddings": False,
    "n_routed_experts": 512,
    "num_experts_per_tok": 16,
    "moe_intermediate_size": 1024,
    "latent_moe_dim": 512,
    "dense_layers_indices": [0],
    "dense_mlp_intermediate_size": 8192,
    "dense_layers_use_shared_expert": True,
    "shared_expert_intermediate_size": 1024,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 8,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 256,
    "linear_conv_kernel_dim": 4,
    "number_of_conv_states": 3,
    "layer_types": ["full_attention" if i in (7, 15) else "linear_attention" for i in range(16)],
}


def model_metadata(model, prepared):
    """Read config.json when available, including after remote preparation."""
    source = prepared.get("hf_checkpoint")
    if source is None:
        source = model["source"] if model["format"] == "hf" else model.get("hf_template")
    path = Path(source) / "config.json" if source else None
    if path is None or not path.is_file():
        return None, None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise InputError(f"Cannot read inference profile model metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise InputError(f"Inference profile model metadata must be an object: {path}")
    return value, str(path)


def resolve(selector, cluster, model, options):
    if not isinstance(selector, str) or selector not in ("auto", "generic", *PROFILES):
        raise InputError(f"inference.profile must be auto, generic or one of {sorted(PROFILES)}")
    report = {
        "requested": selector,
        "selected": "generic",
        "status": "fallback",
        "expected_gpu": gpu_hardware.CLUSTERS.get(cluster, {}).get("gpu"),
        "defaults": {},
        "qualification": "Starting point only; runtime, workload and async-refresh qualification remain required.",
    }

    def fallback(reason):
        if selector in PROFILES and report["status"] != "pending_model_metadata":
            raise InputError(f"inference.profile={selector}: {reason}")
        report["reason"] = reason
        return report

    if selector == "generic":
        return fallback("Generic defaults explicitly selected")
    if options.get("colocate") or options.get("rollout_num_gpus_per_engine", 1) != 1:
        return fallback("Measured profiles require dedicated TP1 inference engines")
    if any(options.get(name) for name in ("rollout_external_engine_addrs", "custom_inference_engine_provider_path")):
        return fallback("External engine settings are owned by their provider")
    if any(options.get(name, 1) != 1 for name in ("sglang_dp_size", "sglang_ep_size", "sglang_pp_size")):
        return fallback("Measured profiles require ordinary single-rank engines")
    gpu = PROFILES.get(selector, gpu_hardware.CLUSTERS.get(cluster, {}).get("gpu"))
    if gpu not in PROFILES.values():
        return fallback("No measured hardware mapping for this cluster")
    if selector in PROFILES and gpu_hardware.CLUSTERS.get(cluster, {}).get("gpu") not in (None, gpu):
        return fallback("Named profile conflicts with the cluster GPU type")
    if options.get("sglang_quantization"):
        return fallback("Quantized inference was not measured")
    if options.get("sglang_dtype", "auto") not in ("auto", "bfloat16"):
        return fallback("Serving dtype differs from the measured BF16 configuration")
    if (
        options.get("rollout_max_context_len") != MEASURED_CONTEXT
        or options.get("rollout_max_response_len") != MEASURED_RESPONSE
    ):
        return fallback(
            f"Automatic profiles currently cover the measured {MEASURED_CONTEXT:,} context / {MEASURED_RESPONSE:,} response caps"
        )
    if model is None:
        report["status"] = "pending_model_metadata"
        return fallback("Model config.json is unavailable here; resolve again after model preparation")
    shape = {key: model.get(key) for key in MODEL_SHAPE}
    report["model_shape_sha256"] = hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()
    if shape != MODEL_SHAPE or model.get("dtype", model.get("torch_dtype")) != "bfloat16":
        return fallback("Model geometry or weight dtype differs from the measured checkpoint")
    if model.get("quantization_config"):
        return fallback("Quantized inference was not measured")
    cache_off = options.get("sglang_disable_radix_cache", gpu != "b300")
    measured_running = 512 if gpu == "b300" or cache_off else 320
    running = options.get("sglang_max_running_requests", measured_running)
    if running > measured_running and any(
        key not in options for key in ("sglang_max_total_tokens", "sglang_max_mamba_cache_size")
    ):
        raise InputError(
            f"Admission {running} exceeds the {measured_running}-request {gpu.upper()} "
            f"cache-{'off' if cache_off else 'on'} profile. Automatic pool extrapolation is not qualified; "
            "lower admission or explicitly size both sglang_max_total_tokens and sglang_max_mamba_cache_size "
            "and validate graph/workspace memory on the target GPU."
        )
    kv_contexts = running if gpu == "b300" else min(running, 192 if cache_off else 128)
    state_multiplier = (2 if gpu == "b300" else 1) if cache_off else 6
    report.update(
        selected=f"olmo35-{gpu}-v1",
        status="candidate",
        expected_gpu=gpu,
        reason="Matched model geometry, dedicated TP1 placement, context caps and GPU family",
        defaults={
            "sglang_disable_radix_cache": cache_off,
            **({"sglang_mamba_radix_cache_strategy": "extra_buffer"} if not cache_off else {}),
            "sglang_max_running_requests": running,
            "sglang_server_concurrency": running,
            "sglang_cuda_graph_backend_decode": "full",
            "sglang_cuda_graph_max_bs_decode": running,
            "sglang_cuda_graph_backend_prefill": "disabled",
            "sglang_max_total_tokens": kv_contexts * MEASURED_CONTEXT,
            "sglang_max_mamba_cache_size": running * state_multiplier,
            "sglang_mem_fraction_static": 0.7 if gpu == "b300" else 0.9,
            "sglang_chunked_prefill_size": 8192,
        },
    )
    validate_pool_budget(report, {**report["defaults"], **options})
    return report


def validate_pool_budget(report, options):
    """Reject impossible matched pools; passing is not a full memory-fit proof."""
    lower_bound = (
        WEIGHTS_LOWER_BOUND_MIB
        + options["sglang_max_total_tokens"] * KV_BYTES_PER_TOKEN / (1024 * 1024)
        + options["sglang_max_mamba_cache_size"] * RECURRENT_MATRIX_MIB_PER_SLOT
    )
    budget = MINIMUM_MEMORY_MIB[report["expected_gpu"]] * options["sglang_mem_fraction_static"]
    report["memory_lower_bound_mib"] = lower_bound
    report["profile_memory_budget_mib"] = budget
    if lower_bound > budget:
        raise InputError(
            f"Requested profile pools need at least {lower_bound / 1024:.2f} GiB for weights/KV/recurrent matrices, "
            f"above this profile's {budget / 1024:.2f} GiB static-memory budget before graphs/workspaces. "
            "Reduce explicit pools/admission or choose generic and qualify the different hardware/memory budget."
        )


def advisories(report, reference, options):
    """Explain departures from measured defaults without changing user choices.

    The reference preserves the selected cache variant and model/workload/topology
    eligibility, while ignoring other serving overrides. Fleet coverage is deliberately separate from per-engine fit.
    """
    warnings = []

    def warn(code, message, **details):
        warnings.append({"code": code, "message": message, **details})

    if report.get("explicit_selection") and report["requested"] != "generic" and report["status"] != "candidate":
        warn(
            "inference_profile_pending"
            if report["status"] == "pending_model_metadata"
            else "inference_profile_fallback",
            f"Inference profile {report['requested']}: {report['reason']}. "
            "Generic settings are provisional starting points, not a memory-fit or throughput guarantee. "
            "Inspect resolved-plan.json after preparation and qualify the intended workload before scaling.",
        )
    if report["requested"] == "generic" and reference["status"] == "pending_model_metadata":
        warn(
            "inference_profile_pending",
            "Generic inference settings were explicitly selected, but checkpoint metadata is unavailable; "
            "a matching measured profile cannot yet be checked. Inspect resolved-plan.json after preparation.",
        )
    if reference["status"] != "candidate":
        return warnings
    differences = {
        key: {"configured": options.get(key), "profile_default": value}
        for key, value in reference["defaults"].items()
        if options.get(key) != value
    }
    if report["requested"] == "generic":
        warn(
            "inference_profile_bypassed",
            f"Measured profile {reference['selected']} matches this model, hardware and engine topology, "
            "but inference.profile=generic bypasses it. Compare the recorded setting differences; "
            "use inference.profile=auto or the named profile to adopt its defaults, or retain generic deliberately.",
            differences=differences,
        )
    elif differences:
        description = "; ".join(
            f"{key}={values['configured']!r} (profile default {values['profile_default']!r})"
            for key, values in differences.items()
        )
        warn(
            "inference_profile_modified",
            f"Resolved settings differ from cache-variant defaults for {reference['selected']}: {description}. "
            "Overrides remain in effect. The selected cache variant is the comparison baseline; "
            "other derived capacities are arithmetic, not measured optima. See docs/miles/throughput-profiles.md "
            "and validate throughput, memory and async refresh for the resulting configuration.",
            differences=differences,
        )
    gpus = options.get("rollout_num_gpus", 1)
    engines = gpus // options.get("rollout_num_gpus_per_engine", 1)
    if engines > 1:
        warn(
            "inference_fleet_extrapolation",
            f"This plan requests {engines} inference engines. Profile {reference['selected']} selects "
            "per-engine settings; it does not certify this fleet's routing, producer capacity or weight-publication "
            "latency. The B300 study includes a separate two-engine throughput check, but no multi-engine "
            "fully-async refresh qualification; the updated-runtime async trainer-consumption check used one B300 "
            "engine. H100 deployment/replay checks have a separate scope. Inspect async_capacity warnings and "
            "measure occupancy, policy age/discards and publication acknowledgments on the intended placement "
            "before scaling. See docs/miles/throughput-profiles.md.",
            requested_engines=engines,
        )
    return warnings


def validate_hardware(report, devices):
    """One startup check per node, not a token-path runtime check."""
    if not report or report["status"] != "candidate":
        return
    expected = report["expected_gpu"]
    minimum_mib = MINIMUM_MEMORY_MIB[expected]
    if not devices or any(
        expected not in name.lower() or memory is None or memory < minimum_mib for name, memory in devices
    ):
        raise InputError(
            f"Inference profile {report['selected']} expects {expected.upper()} GPUs with at least "
            f"{minimum_mib} MiB; found {devices}. Select inference.profile=generic or the correct profile."
        )


def check_hardware(report, device_ids=None, *, node=None):
    """Check visible devices once before starting a matched serving profile."""
    if not report or report["status"] != "candidate":
        return []
    inventory = gpu_hardware.inventory(device_ids)
    devices = (
        gpu_hardware.rollout_inventory(node, inventory)["hardware"]
        if node is not None
        else [(device["name"], device["memory_mib"]) for device in inventory]
    )
    validate_hardware(report, devices)
    return devices


def check_cluster_hardware(report, directory, layout):
    """Validate every rollout replica's startup inventory after model preparation.

    Inventory is collected independently of checkpoint availability. The fresh
    rendezvous directory is passed only by the managed cluster launcher.
    """
    if not report or report["status"] != "candidate":
        return []
    inventories = []
    expected = sorted(node["rollout_gpus"] for node in layout["nodes"])
    for rank in range(layout["replicas"]):
        path = Path(directory) / f"inference-profile-{rank}.json"
        try:
            record = json.loads(path.read_text())
            count = record["rollout_gpus"]
            devices = record["hardware"]
            if type(count) is not int or count < 0 or len(devices) != count:
                raise ValueError("Rollout device inventory is incomplete")
            if count:
                validate_hardware(report, devices)
            inventories.append(record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise InputError(f"Cannot qualify rollout hardware from {path}: {error}") from error
    if sorted(record["rollout_gpus"] for record in inventories) != expected:
        raise InputError("Rollout hardware inventory does not match the planned replica placement")
    return inventories
