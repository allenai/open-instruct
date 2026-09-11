"""Audit native -> streaming HF and HF -> Core -> streaming HF on CPU.

Reads source checkpoints without modifying them or loading optimizer moments.
This checks architecture/weight interchange, not forward or cached-generation parity.
"""

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from olmo_core.config import DType
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_model_and_optim_state
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.config import get_hf_config
from olmo_core.nn.hf.convert_checkpoint import (
    _load_ddp_optimizer_model_state,
    _normalize_legacy_latent_moe_config,
    load_config,
)
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from olmo_core.nn.transformer.config import TransformerConfig
from scripts.miles.checkpoint_weights import SafeTensorState, compare_stream

# Execution backend, dtype and tokenizer metadata may differ across export. These
# fields instead describe the mathematical architecture consumed by the adapter.
ARCHITECTURE_FIELDS = (
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "layer_types",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "latent_moe_dim",
    "latent_moe_bias",
    "latent_moe_up_proj_input_norm",
    "dense_layers_indices",
    "dense_mlp_intermediate_size",
    "embed_norm",
    "use_peri_ln",
    "rms_norm_eps",
    "use_head_qk_norm",
    "qk_norm_per_head_gains",
    "scalable_softmax",
    "use_rope",
    "attention_gate_type",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_allow_neg_eigval",
    "linear_conv_kernel_dim",
    "gating_function",
    "normalize_expert_weights",
    "restore_weight_scale",
    "original_num_experts_per_tok",
    "embed_scale",
    "attention_dropout",
    "attention_gate_full_precision",
    "linear_norm_eps",
    "tie_word_embeddings",
    "global_load_balancing",
)


def check_architecture(actual, expected):
    differences = {
        key: [getattr(actual, key, None), getattr(expected, key, None)]
        for key in ARCHITECTURE_FIELDS
        if getattr(actual, key, None) != getattr(expected, key, None)
    }
    if (
        "sliding_attention" in (getattr(expected, "layer_types", None) or [])
        and actual.sliding_window != expected.sliding_window
    ):
        differences["sliding_window"] = [actual.sliding_window, expected.sliding_window]
    if getattr(expected, "use_rope", False):
        for key in ("rope_theta", "rope_parameters"):
            if getattr(actual, key, None) != getattr(expected, key, None):
                differences[key] = [getattr(actual, key, None), getattr(expected, key, None)]
    if differences:
        raise ValueError(f"Architecture mismatch: {differences}")


def validate_native_parameter_sources(model, checkpoint):
    """Reject ambiguous DDP master/model copies before allocating model storage."""
    keys = set(get_checkpoint_metadata(checkpoint).state_dict_metadata)
    if not any(key.endswith(".main") for key in keys):
        return {"layout": "conventional_model"}
    selected = {}
    for name, _ in model.named_parameters():
        candidates = (
            f"model.{name}",
            f"model.module.{name}",
            name,
            f"module.{name}",
            f"{name}.main",
            f"module.{name}.main",
        )
        matches = [key for key in candidates if key in keys]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous native parameter copies for {name}: {matches}")
        if not matches:
            raise ValueError(f"Missing native parameter source: {name}")
        selected[name] = matches[0]
    return {
        "layout": "ddp_optimizer",
        "parameters": len(selected),
        "master_parameter_sources": sum(key.endswith(".main") for key in selected.values()),
        "selected_parameter_keys_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
    }


def validate(native_path, hf_path):
    started = time.monotonic()
    experiment = load_config(native_path)
    config_dict = experiment["model"]
    _normalize_legacy_latent_moe_config(config_dict)
    config = TransformerConfig.from_dict(config_dict)
    model = config.build(init_device="meta")
    hf = Olmo3MoeConfig.from_dict(json.loads((hf_path / "config.json").read_text()))
    native_hf = get_hf_config(model)
    check_architecture(native_hf, hf)
    if hf.vocab_size != experiment["dataset"]["tokenizer"]["vocab_size"]:
        raise ValueError("HF vocabulary differs from the native tokenizer vocabulary")
    checkpoint = native_path / "model_and_optim"
    parameter_sources = validate_native_parameter_sources(model, checkpoint)
    model.to_empty(device="cpu")
    with TemporaryDirectory(prefix="hero-model-load-") as work:
        loaded = _load_ddp_optimizer_model_state(checkpoint, model, work_dir=work, return_state_dict=False)
        if loaded is None:
            load_model_and_optim_state(checkpoint, model, work_dir=work)
    with SafeTensorState(hf_path) as reference:
        native_result = compare_stream(
            olmo3.iter_olmo3_moe_hf_state(model, hf),
            reference,
            native_vocab_size=config.vocab_size,
            hf_vocab_size=hf.vocab_size,
        )
    del model
    gc.collect()
    print(json.dumps({"stage": "native_to_hf", **native_result}), flush=True)

    adapter_config = olmo3.build_olmo3_moe_config_from_hf_config(
        hf, dtype=DType.bfloat16, attention_backend=AttentionBackendName.torch
    )
    adapter = adapter_config.build(init_device="meta")
    adapter_hf = get_hf_config(adapter)
    check_architecture(adapter_hf, hf)
    adapter.to_empty(device="cpu")
    with SafeTensorState(hf_path) as reference:
        olmo3.load_olmo3_moe_hf_state(adapter, hf, reference)
        roundtrip = compare_stream(olmo3.iter_olmo3_moe_hf_state(adapter, hf), reference)
    return {
        "valid": True,
        "scope": "CPU architecture and exhaustive weight conversion; no forward/generation qualification",
        "native_checkpoint": str(native_path),
        "reference_hf_checkpoint": str(hf_path),
        "native_config_sha256": hashlib.sha256((native_path / "config.json").read_bytes()).hexdigest(),
        "hf_config_sha256": hashlib.sha256((hf_path / "config.json").read_bytes()).hexdigest(),
        "architecture": {key: getattr(hf, key, None) for key in ARCHITECTURE_FIELDS},
        "dense_layers_use_shared_expert": {
            "native": native_hf.dense_layers_use_shared_expert,
            "reference_hf": hf.dense_layers_use_shared_expert,
            "adapter": adapter_hf.dense_layers_use_shared_expert,
        },
        "native_parameter_sources": parameter_sources,
        "native_to_hf": native_result,
        "hf_core_hf": roundtrip,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--hf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = validate(args.native, args.hf)
    except Exception as exc:
        args.report.write_text(json.dumps({"valid": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
        raise
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
