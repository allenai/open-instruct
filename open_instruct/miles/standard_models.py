"""Standard Core dense transformer construction, interchange, and checkpoints."""

from typing import Any

from olmo_core import config as core_config
from olmo_core.distributed import checkpoint as core_checkpoint
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn import attention, feed_forward, layer_norm, lm_head, transformer
from olmo_core.nn.hf import convert
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module import transformer as train_transformer
from olmo_core.train.train_module.transformer import config as train_config
from torch import distributed as dist
from torch.distributed.checkpoint import state_dict as distributed_state
from torch.distributed.tensor import DTensor

from open_instruct.miles.config import CoreConfig


def model_config_from_hf(hf: Any, options: Any) -> transformer.TransformerConfig:
    dtype = core_config.DType.bfloat16
    backend = attention.AttentionBackendName(options.attention_backend)
    if hf.model_type not in ("llama", "qwen2", "qwen3", "olmo2", "olmo3"):
        raise ValueError(f"No verified Core HF factory for {hf.model_type}; supply core.model_config")
    rope = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Scaled RoPE requires an explicit matching core.model_config")
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
    if layers and "sliding_attention" in layers:
        if set(layers) - {"sliding_attention", "full_attention"}:
            raise ValueError("Hybrid layers require an explicit Core model config")
        config.block.sequence_mixer.sliding_window = attention.SlidingWindowAttentionConfig(
            pattern=[hf.sliding_window if kind == "sliding_attention" else -1 for kind in layers],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        )
    config.lm_head.loss_implementation = lm_head.LMLossImplementation.default
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
