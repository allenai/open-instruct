from collections.abc import Iterable
from functools import partial
from itertools import islice

import torch
from torch import nn
from transformers import Olmo2Config
from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Olmo3Config


def compute_kv_sharing_targets(layer_types: list[str], sharing_group_size: int) -> dict[int, int]:
    """Compute which global layers share KV cache with which other layers.

    For MixAttention-style sharing, global attention layers are grouped, and
    all layers in a group share the KV cache of the first layer in that group.

    Args:
        layer_types: List of layer type strings ("sliding_attention" or "full_attention")
        sharing_group_size: Number of global attention layers per sharing group (N)

    Returns:
        Dict mapping layer_idx -> target_layer_idx for layers that should share.
        Layers not in the dict should use their own KV cache.

    Example with sharing_group_size=2 and layer_types with global layers at [3, 7, 11, 15]:
        - Group 0: [3, 7] -> returns {7: 3} (layer 7 shares with layer 3)
        - Group 1: [11, 15] -> returns {15: 11} (layer 15 shares with layer 11)
    """
    if sharing_group_size <= 1:
        return {}

    global_layer_indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]

    sharing_map = {}
    for group_start in range(0, len(global_layer_indices), sharing_group_size):
        group = global_layer_indices[group_start : group_start + sharing_group_size]
        if len(group) > 1:
            target = group[0]
            for layer_idx in group[1:]:
                sharing_map[layer_idx] = target

    return sharing_map


class Olmo2SharedKVAttention(nn.Module):
    """Olmo2 attention with optional KV cache sharing."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", kv_sharing_target_layer_name: str | None = None):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        assert isinstance(self.config, (Olmo2Config, Olmo3Config))

        hidden_size = self.config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.num_attention_heads

        assert hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = self.config.num_key_value_heads or self.total_num_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings
        self.rope_theta = self.config.rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.tp_rank = get_tensor_model_parallel_rank()
        self.k_norm = RMSNorm(self.total_num_kv_heads * self.head_dim, eps=self.config.rms_norm_eps)
        self.q_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.scaling = self.head_dim**-0.5

        layer_idx = extract_layer_index(prefix)
        sliding_window = None
        if (layer_types := getattr(self.config, "layer_types", None)) is not None and layer_types[
            layer_idx
        ] == "sliding_attention":
            sliding_window = self.config.sliding_window

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
        )

        self.rope_scaling = self.config.rope_scaling if sliding_window is None else None
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Olmo2MLP(nn.Module):
    """MLP block for Olmo2."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert isinstance(config, (Olmo2Config, Olmo3Config))
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )

        self.act_fn = SiluAndMul()

        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Olmo2SharedKVDecoderLayer(nn.Module):
    """Decoder layer with shared KV attention support."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", kv_sharing_target_layer_name: str | None = None):
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert isinstance(config, (Olmo2Config, Olmo3Config))

        self.self_attn = Olmo2SharedKVAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
        )

        self.mlp = Olmo2MLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile
class Olmo2SharedKVModel(nn.Module):
    """Olmo2 model with KV cache sharing for global attention layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        assert isinstance(self.config, (Olmo2Config, Olmo3Config))

        kv_cache_sharing_group_size = getattr(self.config, "kv_cache_sharing_group_size", 1)

        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "sliding_attention" if (i + 1) % 4 != 0 else "full_attention"
                for i in range(self.config.num_hidden_layers)
            ]

        sharing_map = compute_kv_sharing_targets(layer_types, kv_cache_sharing_group_size)

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size, self.config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )

        self.start_layer, self.end_layer, self.layers = self._make_layers_with_sharing(
            vllm_config, prefix, sharing_map
        )

        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

    def _make_layers_with_sharing(
        self, vllm_config: VllmConfig, prefix: str, sharing_map: dict[int, int]
    ) -> tuple[int, int, nn.ModuleList]:
        """Create layers with KV cache sharing targets applied."""
        from vllm.distributed import get_pp_group

        pp_rank = get_pp_group().rank_in_group
        pp_size = get_pp_group().world_size
        num_layers = self.config.num_hidden_layers

        layers_per_pp = (num_layers + pp_size - 1) // pp_size
        start_layer = pp_rank * layers_per_pp
        end_layer = min(start_layer + layers_per_pp, num_layers)

        layers = nn.ModuleList()
        for i in range(start_layer, end_layer):
            layer_prefix = f"{prefix}.layers.{i}"

            kv_sharing_target = None
            if i in sharing_map:
                target_layer_idx = sharing_map[i]
                kv_sharing_target = f"{prefix}.layers.{target_layer_idx}.self_attn.attn"

            layer = Olmo2SharedKVDecoderLayer(
                vllm_config=vllm_config, prefix=layer_prefix, kv_sharing_target_layer_name=kv_sharing_target
            )
            layers.append(layer)

        return start_layer, end_layer, layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            assert isinstance(hidden_states, torch.Tensor)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Olmo2SharedKVForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    """Olmo2 causal LM with KV cache sharing for global attention layers."""

    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        assert isinstance(config, (Olmo2Config, Olmo3Config))
        self.config = config
        self.model = Olmo2SharedKVModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self, skip_prefixes=(["lm_head.weight"] if self.config.tie_word_embeddings else None)
        )
        return loader.load_weights(weights)


def register_shared_kv_model():
    """Register the custom shared KV model with vLLM's ModelRegistry."""
    from vllm import ModelRegistry

    ModelRegistry.register_model("Olmo2SharedKVForCausalLM", "open_instruct.models:Olmo2SharedKVForCausalLM")
