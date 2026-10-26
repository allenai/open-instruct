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


_COMMON = {
    "sglang_max_running_requests": 512,
    "sglang_server_concurrency": 512,
    "sglang_cuda_graph_backend_decode": "full",
    "sglang_cuda_graph_max_bs_decode": 512,
    "sglang_cuda_graph_backend_prefill": "disabled",
    "sglang_chunked_prefill_size": 8192,
}
PRESETS = {
    "olmo35-b300-v1": _COMMON
    | {
        "sglang_disable_radix_cache": False,
        "sglang_mamba_radix_cache_strategy": "extra_buffer",
        "sglang_max_total_tokens": 512 * MEASURED_CONTEXT,
        "sglang_max_mamba_cache_size": 3072,
        "sglang_mem_fraction_static": 0.7,
    },
    "olmo35-h100-v1": _COMMON
    | {
        "sglang_disable_radix_cache": True,
        "sglang_max_total_tokens": 192 * MEASURED_CONTEXT,
        "sglang_max_mamba_cache_size": 512,
        "sglang_mem_fraction_static": 0.9,
    },
}


def editable_toml(name):
    """Render the complete preset as an editable generic configuration."""
    lines = ["[inference]", 'profile = "generic"']
    lines.extend(f"{key} = {json.dumps(value)}" for key, value in PRESETS[name].items())
    return "\n".join(lines)


def resolve(selector, cluster, model, options):
    """Apply a whole preset, or retain generic settings with an optional hint."""
    if not isinstance(selector, str) or selector not in ("auto", "generic", *PROFILES):
        raise InputError(f"inference.profile must be auto, generic or one of {sorted(PROFILES)}")
    report = {
        "requested": selector,
        "selected": "generic",
        "status": "generic" if selector == "generic" else "fallback",
        "expected_gpu": gpu_hardware.CLUSTERS.get(cluster, {}).get("gpu"),
        "defaults": {},
        "qualification": "Starting point only; runtime, workload and async-refresh qualification remain required.",
    }

    def no_match(reason):
        if selector in PROFILES:
            raise InputError(f"inference.profile={selector}: {reason}")
        report["reason"] = reason
        return report

    if options.get("colocate") or options.get("rollout_num_gpus_per_engine", 1) != 1:
        return no_match("Measured presets require dedicated TP1 inference engines")
    if any(options.get(name) for name in ("rollout_external_engine_addrs", "custom_inference_engine_provider_path")):
        return no_match("External engine settings are owned by their provider")
    if any(options.get(name, 1) != 1 for name in ("sglang_dp_size", "sglang_ep_size", "sglang_pp_size")):
        return no_match("Measured presets require ordinary single-rank engines")
    gpu = PROFILES.get(selector, gpu_hardware.CLUSTERS.get(cluster, {}).get("gpu"))
    if gpu not in PROFILES.values():
        return no_match("No measured hardware mapping for this cluster")
    if selector in PROFILES and gpu_hardware.CLUSTERS.get(cluster, {}).get("gpu") not in (None, gpu):
        return no_match("Named preset conflicts with the cluster GPU type")
    if options.get("sglang_quantization"):
        return no_match("Quantized inference was not measured")
    if options.get("sglang_dtype", "auto") not in ("auto", "bfloat16"):
        return no_match("Serving dtype differs from the measured BF16 configuration")
    if (
        options.get("rollout_max_context_len") != MEASURED_CONTEXT
        or options.get("rollout_max_response_len") != MEASURED_RESPONSE
    ):
        return no_match(f"Measured presets cover {MEASURED_CONTEXT:,} context / {MEASURED_RESPONSE:,} response caps")
    if model is not None:
        shape = {key: model.get(key) for key in MODEL_SHAPE}
        report["model_shape_sha256"] = hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()
        if shape != MODEL_SHAPE or model.get("dtype", model.get("torch_dtype")) != "bfloat16":
            return no_match("Model geometry or weight dtype differs from the measured checkpoint")
        if model.get("quantization_config"):
            return no_match("Quantized inference was not measured")
    name = f"olmo35-{gpu}-v1"
    if selector == "generic":
        report["reason"] = "Generic settings retained"
        if model is not None:
            report["available_profile"] = name
        return report
    conflicts = sorted(set(PROFILE_KEYS).intersection(options))
    if conflicts:
        values = ", ".join(f"{key}={options[key]!r}" for key in conflicts)
        raise InputError(
            f"inference.profile={selector} applies {name} as a whole preset. "
            f"Explicit preset-owned settings conflict: {values}. Equal values also conflict.\n"
            "Either remove these settings (including native aliases) and keep the preset, "
            "or select generic and manage the settings explicitly. "
            "Merge this complete editable preset block into your inference section, then change it as needed:\n\n"
            + editable_toml(name)
        )
    if model is None:
        report.update(
            status="pending_model_metadata",
            reason="Model config.json is unavailable here; resolve again after model preparation",
        )
        return report
    report.update(
        selected=name,
        status="candidate",
        expected_gpu=gpu,
        reason="Matched model geometry, dedicated TP1 placement, context caps and GPU family",
        defaults=dict(PRESETS[name]),
    )
    return report


def advisories(report, options):
    """Expose preset availability, missing matches and fleet evidence boundaries."""
    if report["requested"] == "generic":
        available = report.get("available_profile")
        return (
            [
                {
                    "code": "inference_profile_available",
                    "message": f"Measured preset {available} matches; generic settings are unchanged. Select auto or that name to use it whole.",
                }
            ]
            if available
            else []
        )
    if report["status"] != "candidate":
        return [
            {
                "code": "inference_profile_pending"
                if report["status"] == "pending_model_metadata"
                else "inference_profile_fallback",
                "message": f"Inference profile {report['requested']}: {report['reason']}. Generic settings are provisional; inspect resolved-plan.json after preparation.",
            }
        ]
    engines = options.get("rollout_num_gpus", 1) // options.get("rollout_num_gpus_per_engine", 1)
    if engines <= 1:
        return []
    return [
        {
            "code": "inference_fleet_extrapolation",
            "message": f"This plan requests {engines} inference engines. Preset {report['selected']} covers per-engine settings, "
            "not this fleet's routing, producer capacity or publication latency. The B300 two-engine throughput check "
            "does not qualify multi-engine fully async refresh. Inspect async_capacity and validate occupancy, policy age/discards "
            "and publication acknowledgments on the intended placement. See docs/miles/throughput-profiles.md.",
            "requested_engines": engines,
        }
    ]


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
