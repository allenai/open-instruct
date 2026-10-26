"""Measured serving starting points, selected without importing a GPU runtime.

Named presets guard a measured workload; generic settings remain user-owned.
The read-only advisor recommends presets without selecting or resizing settings.
"""

import json
from pathlib import Path

from open_instruct.miles.configuration import validation
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
        # An HF template describes conversion, not the source Core checkpoint.
        source = model["source"] if model["format"] == "hf" else None
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


def incompatibility(name, cluster, model, options):
    """Guard a preset's known model/hardware/topology contract; never select settings."""
    gpu = PROFILES[name]
    cluster_gpu = gpu_hardware.CLUSTERS.get(cluster, {}).get("gpu")
    if cluster_gpu not in (None, gpu):
        alternative = next((key for key, value in PROFILES.items() if value == cluster_gpu), "generic")
        return f"Preset expects {gpu.upper()}, but {cluster} uses {cluster_gpu.upper()}; choose {alternative} or change the cluster"
    if options.get("colocate") or options.get("rollout_num_gpus_per_engine", 1) != 1:
        return "Measured presets require dedicated TP1 inference engines"
    if any(options.get(key) for key in ("rollout_external_engine_addrs", "custom_inference_engine_provider_path")):
        return "External engine settings are owned by their provider"
    if any(options.get(key, 1) != 1 for key in ("sglang_dp_size", "sglang_ep_size", "sglang_pp_size")):
        return "Measured presets require ordinary single-rank engines"
    if options.get("sglang_quantization"):
        return "Quantized inference was not measured"
    if options.get("sglang_dtype", "auto") not in ("auto", "bfloat16"):
        return "Serving dtype differs from the measured BF16 configuration"
    if (
        options.get("rollout_max_context_len") != MEASURED_CONTEXT
        or options.get("rollout_max_response_len") != MEASURED_RESPONSE
    ):
        return f"Measured presets cover {MEASURED_CONTEXT:,} context / {MEASURED_RESPONSE:,} response caps"
    try:
        # Apply the same capacity rules as generic configuration. The advisor
        # checks the whole proposed preset, never derives it from overrides.
        validation.inference_capacity({**options, **PRESETS[name]})
    except InputError as error:
        return str(error)
    if model is not None:
        shape = {key: model.get(key) for key in MODEL_SHAPE}
        if shape != MODEL_SHAPE or model.get("dtype", model.get("torch_dtype")) != "bfloat16":
            return "Model geometry or weight dtype differs from the measured checkpoint; use generic for this model"
        if model.get("quantization_config"):
            return "Quantized inference was not measured"
    return None


def resolve(selector, cluster, model, options, *, require_model_metadata=False):
    """Apply only an explicit named preset. Generic never changes serving settings."""
    if not isinstance(selector, str) or selector not in ("generic", *PROFILES):
        raise InputError(
            f"inference.profile must be generic (or omitted) or one of {sorted(PROFILES)}; auto was removed. Planning offers recommendations without changing settings."
        )
    report = {
        "requested": selector,
        "selected": selector,
        "defaults": {},
        "model_metadata_available": model is not None,
    }
    if selector == "generic":
        return report
    reason = incompatibility(selector, cluster, model, options)
    if reason:
        raise InputError(f"inference.profile={selector}: {reason}")
    conflicts = sorted(set(PROFILE_KEYS).intersection(options))
    if conflicts:
        values = ", ".join(f"{key}={options[key]!r}" for key in conflicts)
        raise InputError(
            f"inference.profile={selector} is a whole preset. Explicit preset-owned settings conflict: {values}. Equal values also conflict.\n"
            "Either remove these settings (including native aliases) and keep the preset, or select generic and manage them explicitly. "
            "Merge this editable preset block into your inference section, then change it as needed:\n\n"
            + editable_toml(selector)
        )
    if model is None and require_model_metadata:
        raise InputError(
            f"inference.profile={selector}: prepared checkpoint config.json is unavailable; cannot verify model geometry. Fix model preparation or select generic explicitly."
        )
    report.update(expected_gpu=PROFILES[selector], defaults=dict(PRESETS[selector]))
    return report


def advise(report, cluster, model, options, explicit_keys):
    """Read-only suggestions and evidence notes, independent of configuration selection."""
    warnings, tips = [], []
    selected = report["selected"]
    if selected == "generic":
        cluster_gpu = gpu_hardware.CLUSTERS.get(cluster, {}).get("gpu")
        for name, preset in PRESETS.items():
            if (
                model is not None
                and cluster_gpu == PROFILES[name]
                and not incompatibility(name, cluster, model, options)
            ):
                tips.append(
                    {
                        "code": "inference_profile_available",
                        "message": f'Measured preset {name} matches. To adopt it whole, add inference.profile = "{name}" and remove its explicit serving keys. Your current settings are unchanged.',
                    }
                )
            pool_keys = ("sglang_max_total_tokens", "sglang_max_mamba_cache_size")
            if (
                cluster_gpu
                and cluster_gpu != PROFILES[name]
                and all(key in explicit_keys and options.get(key) == preset[key] for key in pool_keys)
            ):
                tips.append(
                    {
                        "code": "inference_pool_hardware_mismatch",
                        "message": f"Your explicit KV/recurrent pools resemble {name} ({PROFILES[name].upper()}), but {cluster} uses {cluster_gpu.upper()}. Review GPU memory and the matching hardware's preset before running; settings are unchanged.",
                    }
                )
        if model is None:
            warnings.append(
                {
                    "code": "inference_metadata_unavailable",
                    "message": "Model metadata is unavailable here; preset recommendations will be checked after preparation. Generic settings are unchanged.",
                }
            )
    else:
        if model is None:
            warnings.append(
                {
                    "code": "inference_model_guard_deferred",
                    "message": f"Preset {selected} is selected; model geometry must be verified after preparation before training.",
                }
            )
        engines = options.get("rollout_num_gpus", 1) // options.get("rollout_num_gpus_per_engine", 1)
        if engines > 1:
            warnings.append(
                {
                    "code": "inference_fleet_extrapolation",
                    "message": f"This plan requests {engines} inference engines. Preset {selected} covers per-engine settings, not fleet routing, producer capacity or publication latency. The B300 two-engine throughput check does not qualify multi-engine fully async refresh; validate the intended placement. See docs/miles/throughput-profiles.md.",
                    "requested_engines": engines,
                }
            )
    return warnings, tips


def validate_hardware(report, devices):
    """One startup check per node, not a token-path runtime check."""
    if not report or report.get("selected") not in PROFILES:
        return
    expected = report["expected_gpu"]
    minimum_mib = MINIMUM_MEMORY_MIB[expected]
    if not devices or any(
        not isinstance(name, str)
        or expected not in name.lower()
        or not isinstance(memory, (int, float))
        or memory < minimum_mib
        for name, memory in devices
    ):
        raise InputError(
            f"Inference profile {report['selected']} expects {expected.upper()} GPUs with at least "
            f"{minimum_mib} MiB; found {devices}. Select inference.profile=generic or the correct profile."
        )


def check_hardware(report, *, node):
    """Check only this node's rollout slice before starting a named preset."""
    if not report or report.get("selected") not in PROFILES:
        return []
    inventory = gpu_hardware.inventory()
    devices = gpu_hardware.rollout_inventory(node, inventory)["hardware"]
    validate_hardware(report, devices)
    return devices


def check_cluster_hardware(report, directory, layout):
    """Validate every rollout replica's startup inventory after model preparation.

    Inventory is collected independently of checkpoint availability. The fresh
    rendezvous directory is passed only by the managed cluster launcher.
    """
    if not report or report.get("selected") not in PROFILES:
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
        except InputError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise InputError(f"Cannot qualify rollout hardware from {path}: {error}") from error
    if sorted(record["rollout_gpus"] for record in inventories) != expected:
        raise InputError("Rollout hardware inventory does not match the planned replica placement")
    return inventories
