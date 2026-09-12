"""Standard Core dense transformer construction, interchange, and checkpoints."""

import json
import math
from pathlib import Path
from typing import Any

from olmo_core import config as core_config
from olmo_core.distributed import checkpoint as core_checkpoint
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn import attention, feed_forward, layer_norm, lm_head, transformer
from olmo_core.nn.hf import convert
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module import transformer as train_transformer
from olmo_core.train.train_module.transformer import config as train_config
from torch import distributed as dist
from torch.distributed.checkpoint import state_dict as distributed_state
from torch.distributed.tensor import DTensor

from open_instruct.miles import validation
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.errors import InputError


def _rope_scaling(model_type, rope):
    kind = rope.get("rope_type", rope.get("type", "default"))
    if kind == "default":
        return None
    if model_type != "olmo3" or kind != "yarn":
        raise InputError(
            "Scaled RoPE requires an explicit matching core.model_config; automatic support covers Olmo 3 YaRN"
        )
    validation.fields(
        rope,
        "Olmo 3 YaRN",
        {
            "rope_type",
            "type",
            "rope_theta",
            "factor",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "attention_factor",
        },
    )
    factor = validation.number(rope.get("factor"), "HF rope_scaling.factor", minimum=1)
    original = validation.integer(
        rope.get("original_max_position_embeddings"), "HF rope_scaling.original_max_position_embeddings"
    )
    betas = {}
    for name, default in (("beta_fast", 32), ("beta_slow", 1)):
        value = validation.number(rope.get(name, default), f"HF rope_scaling.{name}", exclusive_min=True)
        if int(value) != value:
            raise InputError(f"HF rope_scaling.{name} must be an integer for Core YaRN")
        betas[name] = int(value)
    if betas["beta_fast"] <= betas["beta_slow"]:
        raise InputError("HF rope_scaling.beta_fast must exceed beta_slow")
    scaling = YaRNRoPEScalingConfig(factor=factor, old_context_len=original, **betas)
    expected = scaling.get_attention_rescale_factor()
    supplied = validation.number(
        rope.get("attention_factor", expected), "HF rope_scaling.attention_factor", exclusive_min=True
    )
    if not math.isclose(supplied, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise InputError(
            f"Core YaRN derives attention_factor={expected}; HF specifies {supplied}. "
            "A custom attention rescale factor requires explicit implementation and parity validation."
        )
    return scaling


def model_config_from_hf(hf: Any, options: Any) -> transformer.TransformerConfig:
    dtype = core_config.DType.bfloat16
    backend = attention.AttentionBackendName(options.attention_backend)
    if hf.model_type not in ("llama", "qwen2", "qwen3", "olmo2", "olmo3"):
        raise ValueError(f"No verified Core HF factory for {hf.model_type}; supply core.model_config")
    rope = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
    scaling = _rope_scaling(hf.model_type, rope)
    is_olmo = hf.model_type in ("olmo2", "olmo3")
    has_qk_norm = is_olmo or hf.model_type == "qwen3"
    config = transformer.TransformerConfig.llama_like(
        d_model=hf.hidden_size,
        vocab_size=hf.vocab_size,
        n_layers=hf.num_hidden_layers,
        n_heads=hf.num_attention_heads,
        n_kv_heads=hf.num_key_value_heads,
        head_dim=getattr(hf, "head_dim", None),
        dtype=dtype,
        qk_norm=has_qk_norm,
        use_head_qk_norm=hf.model_type == "qwen3",
        layer_norm_eps=hf.rms_norm_eps,
        layer_norm_name=layer_norm.LayerNormType.qwen_rms
        if hf.model_type == "qwen3"
        else layer_norm.LayerNormType.rms,
        rope_theta=rope.get("rope_theta", getattr(hf, "rope_theta", 10000)),
        rope_full_precision=is_olmo,
        attn_backend=backend,
        block_name=transformer.TransformerBlockType.reordered_norm
        if is_olmo
        else transformer.TransformerBlockType.default,
        feed_forward=feed_forward.FeedForwardConfig(hidden_size=hf.intermediate_size, bias=False, dtype=dtype),
        tie_word_embeddings=hf.tie_word_embeddings,
    )
    assert isinstance(config.block.sequence_mixer, attention.AttentionConfig)
    # Qwen2 uses QKV biases, but no output-projection bias. Set them separately.
    if hf.model_type == "qwen2" or getattr(hf, "attention_bias", False):
        config.block.sequence_mixer.qkv_bias = True
    layers = getattr(hf, "layer_types", None)
    if layers and (len(layers) != hf.num_hidden_layers or set(layers) - {"sliding_attention", "full_attention"}):
        raise InputError("HF layer_types must name full_attention or sliding_attention for every layer")
    if layers and "sliding_attention" in layers:
        config.block.sequence_mixer.sliding_window = attention.SlidingWindowAttentionConfig(
            pattern=[hf.sliding_window if kind == "sliding_attention" else -1 for kind in layers],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        )
    config.lm_head.loss_implementation = lm_head.LMLossImplementation.default
    if scaling is not None:
        # Released Olmo 3 applies YaRN to global layers only. Sliding layers keep
        # ordinary RoPE, even when the HF descriptor stores one scaling dict.
        config = config.with_rope_scaling(scaling, full_attn_layers_only=True)
    return config


def validate_training_options(args):
    """Reject MoE-only operations before model storage or checkpoint allocation."""
    if getattr(args, "use_rollout_routing_replay", False):
        raise ValueError("Router replay requires an OLMoDDP MoE model; standard dense trainers do not support replay")
    if args.olmo_core.checkpoint_save_options() != CoreConfig().checkpoint_save_options():
        raise ValueError("Checkpoint writer overrides currently require the OLMoDDP MoE trainer")
    if args.olmo_core.expert_parallel_size != 1:
        raise ValueError("Expert parallelism requires an OLMoDDP MoE model")


def replay_context(module, batch):
    raise ValueError("Router replay is not supported by the standard dense trainer")


def build_train_module(args, *, common, optim, hf_config, hf_state):
    validate_training_options(args)
    module = train_transformer.TransformerTrainModule(
        **common,
        optim=AdamWConfig(**optim),
        dp_config=train_config.TransformerDataParallelConfig(name=DataParallelType.fsdp)
        if dist.get_world_size() > 1
        else None,
        ac_config=train_config.TransformerActivationCheckpointingConfig()
        if args.olmo_core.activation_checkpointing
        else None,
    )
    native = convert.convert_state_from_hf(hf_config, hf_state, model_type=hf_config.model_type)
    distributed_state.set_model_state_dict(
        module.model, native, options=distributed_state.StateDictOptions(full_state_dict=True, strict=True)
    )
    return module


def iter_export_state(module, hf, *, stream_moe=True, fused_experts=False):
    if fused_experts:
        raise ValueError("Fused expert publication applies to routed MoE models only")
    for name, value in module.model.state_dict().items():
        native = value.full_tensor() if isinstance(value, DTensor) else value
        yield from convert.convert_state_to_hf(hf, {name: native}).items()


def save_native(module, path):
    core_checkpoint.save_state_dict(str(path), module.state_dict())


def load_native(module, path, *, optim=True):
    state = module.state_dict(optim=optim)
    core_checkpoint.load_state_dict(str(path), state)
    module.load_state_dict(state)


def save_hf_config(hf, path):
    """Keep released Olmo 3 descriptors compatible with SGLang's config reader.

    Transformers 5 moves legacy rope_scaling/theta into rope_parameters. The
    pinned SGLang Olmo 3 config still consumes the released layout and initializes
    shape attributes after its PretrainedConfig superclass. Passing a new-format
    YaRN dict into that superclass triggers validation before those fields exist.
    """
    if hf.model_type != "olmo3":
        hf.save_pretrained(path)
        return
    document = hf.to_dict()
    rope = dict(document.pop("rope_parameters", None) or document.get("rope_scaling") or {})
    _rope_scaling("olmo3", rope)
    document["rope_theta"] = rope.pop("rope_theta", document.get("rope_theta", 10000))
    document["rope_scaling"] = rope if rope.get("rope_type", "default") != "default" else None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
