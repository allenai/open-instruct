"""Model construction and weight interchange owned by OLMo-core."""

import json
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from olmo_core import config as core_config
from olmo_core.distributed import checkpoint as core_checkpoint
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn import attention, feed_forward, layer_norm, lm_head, transformer
from olmo_core.nn.ddp import OLMoDDPModel
from olmo_core.nn.hf import config as hf_config_utils
from olmo_core.nn.hf import convert
from olmo_core.nn.moe.v2 import olmo3
from olmo_core.nn.transformer import config as transformer_config
from olmo_core.optim import AdamWConfig, OLMoDDPOptimizerConfig
from olmo_core.train.train_module import transformer as train_transformer
from olmo_core.train.train_module.transformer import config as train_config
from torch import distributed as dist
from torch.distributed.checkpoint import state_dict as distributed_state
from torch.distributed.tensor import DTensor

from open_instruct import logger_utils
from open_instruct.miles import fla_compat

logger = logger_utils.setup_logger(__name__)


def model_config_from_hf(
    hf: Any, options: Any
) -> transformer.TransformerConfig | transformer_config.OLMoDDPModelConfig:
    dtype = core_config.DType.bfloat16
    backend = attention.AttentionBackendName(options.attention_backend)
    if options.model_config:
        document = json.loads(Path(options.model_config).read_text())
        config = core_config.Config.from_dict(document.get("model", document))
        if not isinstance(config, (transformer.TransformerConfig, transformer_config.OLMoDDPModelConfig)):
            raise ValueError("core.model_config must describe a Core transformer")
        return config
    if hf.model_type == "olmo3moe":
        return olmo3.build_olmo3_moe_config_from_hf_config(
            hf,
            dtype=dtype,
            attention_backend=backend,
            router_aux_loss_weight=options.router_aux_loss_weight,
            router_z_loss_weight=options.router_z_loss_weight,
        )
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
            pattern=[hf.sliding_window - 1 if kind == "sliding_attention" else -1 for kind in layers],
            force_full_attention_on_first_layer=False,
            force_full_attention_on_last_layer=False,
        )
    config.lm_head.loss_implementation = lm_head.LMLossImplementation.default
    return config


class MetricSink:
    """The small trainer-facing surface used by Core's optimizer/metric methods."""

    global_step = 0
    global_train_tokens_seen = 0

    def __init__(self):
        self.metrics = {}

    def record_metric(self, name, value, *args, namespace=None, **kwargs):
        key = f"{namespace}/{name}" if namespace else name
        self.metrics[key] = float(value.detach().item() if isinstance(value, torch.Tensor) else value)


class HFInitializedMoETrainModule(train_transformer.OLMoDDPTrainModule):
    """Import weights after Core sharding but before optimizer master initialization."""

    def __init__(self, *, hf_config, hf_state, **kwargs):
        self._initial_hf_config = hf_config
        self._initial_hf_state = hf_state
        super().__init__(**kwargs)
        self._initial_hf_state = None

    def init_model_weights(self, model_parts, max_sequence_length, rank_microbatch_size):
        super().init_model_weights(model_parts, max_sequence_length, rank_microbatch_size)
        if len(model_parts) != 1:
            raise ValueError("HF import requires a non-pipeline model")
        assert self._initial_hf_state is not None
        olmo3.load_olmo3_moe_hf_state(model_parts[0], self._initial_hf_config, self._initial_hf_state)
        model_parts[0].refresh_rowwise_fp8_cache()


def build_train_module(args, source=None):
    hf_config_utils._register_olmo3moe_auto_classes()
    hf = transformers.AutoConfig.from_pretrained(source or args.hf_checkpoint, trust_remote_code=True)
    if hf.model_type == "olmo3moe":
        # Core DDP represents dense MLP blocks using a single shared expert.
        hf.dense_mlp_uses_shared_experts = True
    if "linear_attention" in (getattr(hf, "layer_types", None) or []):
        fla_compat.install_kda_triton_compat()
    config = model_config_from_hf(hf, args.olmo_core)
    config.init_seed = args.seed
    if isinstance(config, transformer_config.OLMoDDPModelConfig):
        config.recompute_each_block = args.olmo_core.activation_checkpointing
    model = config.build(init_device="meta")
    common: dict[str, Any] = dict(
        model=model,
        rank_microbatch_size=args.olmo_core.max_sequence_length,
        max_sequence_length=args.olmo_core.max_sequence_length,
        compile_model=False,
        device=torch.device("cuda", torch.cuda.current_device()),
        max_grad_norm=args.clip_grad,
    )
    optim: dict[str, Any] = dict(
        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps, weight_decay=args.weight_decay
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        source or args.hf_checkpoint, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    if isinstance(model, OLMoDDPModel):
        module = HFInitializedMoETrainModule(
            hf_config=hf,
            hf_state=hf_model.state_dict(),
            **common,
            optim=OLMoDDPOptimizerConfig(**optim, max_grad_norm=args.clip_grad),
            dp_config=train_config.TransformerDataParallelConfig(name=DataParallelType.ddp),
            ep_config=train_config.TransformerExpertParallelConfig(degree=args.olmo_core.expert_parallel_size)
            if args.olmo_core.expert_parallel_size > 1
            else None,
        )
    else:
        if args.olmo_core.expert_parallel_size != 1:
            raise ValueError("Expert parallelism requires an OLMoDDP MoE model")
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
    cast(Any, module)._trainer = MetricSink()
    if not isinstance(model, OLMoDDPModel):
        native = convert.convert_state_from_hf(hf, hf_model.state_dict(), model_type=hf.model_type)
        distributed_state.set_model_state_dict(
            module.model, native, options=distributed_state.StateDictOptions(full_state_dict=True, strict=True)
        )
    del hf_model
    return module, hf, config


def iter_export_state(module, hf):
    """Yield HF weights with at most one dense parameter gathered at a time.

    MoE conversion stages its full unsharded state on CPU. Its expert all-gather
    still needs room for one complete expert parameter on each EP rank.
    """
    if isinstance(module.model, OLMoDDPModel):
        yield from olmo3.gather_olmo3_moe_hf_state(module.model, hf, cpu=True).items()
        return
    for name, value in module.model.state_dict().items():
        native = value.full_tensor() if isinstance(value, DTensor) else value
        yield from convert.convert_state_to_hf(hf, {name: native}).items()


def export_state(module, hf):
    return {name: value.detach().cpu() for name, value in iter_export_state(module, hf)}


def save_native(module, path):
    if isinstance(module, train_transformer.OLMoDDPTrainModule):
        module.save_state_dict_direct(str(path))
    else:
        core_checkpoint.save_state_dict(str(path), module.state_dict())


def load_native(module, path, *, optim=True):
    if isinstance(module, train_transformer.OLMoDDPTrainModule):
        module.load_state_dict_direct(str(path), load_optim_state=optim)
    else:
        state = module.state_dict(optim=optim)
        core_checkpoint.load_state_dict(str(path), state)
        module.load_state_dict(state)
