"""
GRPO-specific callbacks for OLMo-core Trainer.

These callbacks handle:
- vLLM weight synchronization after each training step
- Reference policy Polyak updates
"""

import contextlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import ray
import ray.exceptions
import torch
import torch.distributed as dist
import torch.nn as nn
from olmo_core.train.callbacks import Callback
from olmo_core.train.train_module import TransformerTrainModule
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from open_instruct import data_loader as data_loader_lib
from open_instruct import logger_utils, vllm_utils

logger = logger_utils.setup_logger(__name__)

_BLOCK_PATTERN = re.compile(r"blocks\.(\d+)\.(.*)")
_OLMO_CORE_TO_HF_LAYER_MAPPINGS = {
    "attention.w_q.weight": "self_attn.q_proj.weight",
    "attention.w_k.weight": "self_attn.k_proj.weight",
    "attention.w_v.weight": "self_attn.v_proj.weight",
    "attention.w_out.weight": "self_attn.o_proj.weight",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward_norm.weight": "post_attention_layernorm.weight",
}


def olmo_core_to_hf_name(name: str) -> str:
    """Convert OLMo-core parameter name to HuggingFace format for Qwen3/LLaMA models."""
    if name == "embeddings.weight":
        return "model.embed_tokens.weight"
    if name == "lm_head.norm.weight":
        return "model.norm.weight"
    if name == "lm_head.w_out.weight":
        return "lm_head.weight"

    layer_match = _BLOCK_PATTERN.match(name)
    if layer_match:
        layer_idx = layer_match.group(1)
        rest = layer_match.group(2)
        if rest in _OLMO_CORE_TO_HF_LAYER_MAPPINGS:
            return f"model.layers.{layer_idx}.{_OLMO_CORE_TO_HF_LAYER_MAPPINGS[rest]}"

    return name


@dataclass
class VLLMWeightSyncCallback(Callback):
    """Callback to synchronize weights from training model to vLLM inference engines.

    After each training step, this callback:
    1. Pauses vLLM actors via actor_manager
    2. Gathers FSDP-sharded parameters using summon_full_params
    3. Broadcasts weights from rank 0 to vLLM engines
    4. Resumes vLLM actors
    """

    vllm_engines: list[ray.actor.ActorHandle]
    model_update_group: dist.ProcessGroup | None = None
    actor_manager: ray.actor.ActorHandle | None = None
    sync_interval: int = 1
    name_mapper: Callable[[str], str] | None = None

    @property
    def train_module(self) -> TransformerTrainModule:
        return cast(TransformerTrainModule, self.trainer.train_module)

    def post_step(self) -> None:
        if self.trainer.global_step % self.sync_interval != 0:
            return

        torch.cuda.empty_cache()

        if self.actor_manager is not None:
            ray.get(self.actor_manager.set_should_stop.remote(True))

        model = self.train_module.model

        try:
            self._broadcast_weights(model)
        finally:
            if self.actor_manager is not None:
                ray.get(self.actor_manager.set_should_stop.remote(False))

    def _broadcast_weights(self, model: nn.Module) -> None:
        """Broadcast weights from training model to vLLM engines."""
        refs = vllm_utils.broadcast_weights_to_vllm(
            model=model,
            vllm_engines=self.vllm_engines,
            model_update_group=self.model_update_group,
            model_step=self.trainer.global_step,
            name_mapper=self.name_mapper,
        )
        if refs:
            ray.get(refs)


@dataclass
class RefPolicyUpdateCallback(Callback):
    """Callback to update reference policy using Polyak averaging.

    Updates reference policy parameters as:
        ref_param = (1 - alpha) * ref_param + alpha * policy_param

    This is used for KL divergence computation in GRPO.
    """

    ref_policy: nn.Module
    alpha: float = 0.6
    update_interval: int = 1
    _fsdp2_submodules: list[FSDPModule] | None = field(default=None, init=False, repr=False)

    @property
    def train_module(self) -> TransformerTrainModule:
        return cast(TransformerTrainModule, self.trainer.train_module)

    def _get_fsdp2_submodules(self, model: nn.Module) -> list[FSDPModule]:
        if self._fsdp2_submodules is None:
            self._fsdp2_submodules = [m for _, m in vllm_utils._get_fsdp2_submodules(model)]
        return self._fsdp2_submodules

    def post_step(self) -> None:
        if self.trainer.global_step % self.update_interval != 0:
            return

        model = self.train_module.model

        if isinstance(model, FSDP):
            ctx = FSDP.summon_full_params(model, writeback=False, rank0_only=False)
        else:
            ctx = contextlib.nullcontext()

        fsdp2_submodules: list[FSDPModule] = []
        if isinstance(model, FSDPModule):
            fsdp2_submodules = self._get_fsdp2_submodules(model)
            for m in fsdp2_submodules:
                m.unshard()

        try:
            with ctx:
                for ref_param, param in zip(self.ref_policy.parameters(), model.parameters(), strict=True):
                    ref_param.data.mul_(1.0 - self.alpha).add_(param.data, alpha=self.alpha)
        finally:
            for m in fsdp2_submodules:
                m.reshard()


@dataclass
class DataPreparationActorCheckpointCallback(Callback):
    """Callback to save and restore DataPreparationActor state during checkpointing."""

    def state_dict(self) -> dict[str, Any]:
        """Save DataPreparationActor state before checkpointing."""
        try:
            data_prep_actor = ray.get_actor(data_loader_lib.DATA_PREP_ACTOR_NAME)
            return {"data_prep_state": ray.get(data_prep_actor.get_state.remote())}
        except (ray.exceptions.RayError, ValueError) as e:
            logger.warning(f"Failed to get DataPreparationActor state: {e}")
            return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore DataPreparationActor state after loading checkpoint."""
        if "data_prep_state" not in state_dict:
            return

        try:
            data_prep_actor = ray.get_actor(data_loader_lib.DATA_PREP_ACTOR_NAME)
            ray.get(data_prep_actor.restore_state.remote(state_dict["data_prep_state"]))
            logger.info("Restored DataPreparationActor state from checkpoint")
        except (ray.exceptions.RayError, ValueError) as e:
            logger.warning(f"Failed to restore DataPreparationActor state: {e}")
