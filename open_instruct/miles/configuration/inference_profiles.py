"""Measured serving starting points, selected without importing a GPU runtime.

Only checkpoint metadata is read. Unknown shapes keep the generic defaults;
these profiles describe a measured workload, not a universal memory estimator.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from open_instruct.miles.errors import InputError

CLUSTER_GPUS = {"ai2/holmes": "b300", "ai2/jupiter": "h100"}
PROFILES = {"olmo35-b300-v1": "b300", "olmo35-h100-v1": "h100"}
PROFILE_KEYS = (
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
        "expected_gpu": CLUSTER_GPUS.get(cluster),
        "defaults": {},
        "qualification": "Starting point only; runtime, workload and async-refresh qualification remain required.",
    }

    def fallback(reason):
        if selector in PROFILES and model is not None:
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
    gpu = PROFILES.get(selector, CLUSTER_GPUS.get(cluster))
    if gpu is None:
        return fallback("No measured hardware mapping for this cluster")
    if selector in PROFILES and CLUSTER_GPUS.get(cluster) not in (None, gpu):
        return fallback("Named profile conflicts with the cluster GPU type")
    if model is None:
        report["status"] = "pending_model_metadata"
        return fallback("Model config.json is unavailable here; resolve again after model preparation")
    shape = {key: model.get(key) for key in MODEL_SHAPE}
    report["model_shape_sha256"] = hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()
    if shape != MODEL_SHAPE or model.get("dtype", model.get("torch_dtype")) != "bfloat16":
        return fallback("Model geometry or weight dtype differs from the measured checkpoint")
    if model.get("quantization_config") or options.get("sglang_quantization"):
        return fallback("Quantized inference was not measured")
    if options.get("sglang_dtype", "auto") not in ("auto", "bfloat16"):
        return fallback("Serving dtype differs from the measured BF16 configuration")
    if options.get("rollout_max_context_len") != 34816 or options.get("rollout_max_response_len") != 32768:
        return fallback("Automatic profiles currently cover the measured 34,816 context / 32,768 response caps")
    cache_off = options.get("sglang_disable_radix_cache", True)
    running = options.get("sglang_max_running_requests", 512 if gpu == "b300" or cache_off else 320)
    kv_contexts = running if gpu == "b300" else min(running, 192 if cache_off else 128)
    state_multiplier = (2 if gpu == "b300" else 1) if cache_off else 6
    report.update(
        selected=f"olmo35-{gpu}-v1",
        status="candidate",
        expected_gpu=gpu,
        reason="Matched model geometry, dedicated TP1 placement, context caps and GPU family",
        defaults={
            "sglang_max_running_requests": running,
            "sglang_server_concurrency": running,
            "sglang_cuda_graph_backend_decode": "full",
            "sglang_cuda_graph_max_bs_decode": running,
            "sglang_cuda_graph_backend_prefill": "disabled",
            "sglang_max_total_tokens": kv_contexts * 34816,
            "sglang_max_mamba_cache_size": running * state_multiplier,
            "sglang_mem_fraction_static": 0.7 if gpu == "b300" else 0.9,
            "sglang_chunked_prefill_size": 8192,
        },
    )
    return report


def validate_hardware(report, devices):
    """One startup check per node, not a token-path runtime check."""
    if report["status"] != "candidate":
        return
    expected = report["expected_gpu"]
    minimum_mib = 270000 if expected == "b300" else 80000
    if not devices or any(expected not in name.lower() or memory < minimum_mib for name, memory in devices):
        raise InputError(
            f"Inference profile {report['selected']} expects {expected.upper()} GPUs with at least "
            f"{minimum_mib} MiB; found {devices}. Select inference.profile=generic or the correct profile."
        )


def check_hardware(report, device_ids=None):
    """Check visible devices once before starting a matched serving profile."""
    if not report or report["status"] != "candidate":
        return []
    command = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    mask = ",".join(device_ids) if device_ids is not None else os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask:
        command.append(f"--id={mask}")
    try:
        output = subprocess.check_output(command, text=True, timeout=15)
        devices = [
            (name.strip(), float(memory)) for name, memory in (line.rsplit(",", 1) for line in output.splitlines())
        ]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise InputError(f"Cannot verify hardware for inference profile {report['selected']}: {error}") from error
    validate_hardware(report, devices)
    return devices
