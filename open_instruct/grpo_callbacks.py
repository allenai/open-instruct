"""
GRPO-specific callbacks for OLMo-core Trainer.

These callbacks handle:
- vLLM weight synchronization after each training step
- Reference policy Polyak updates
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from olmo_core.train.callbacks import Callback
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

logger = logging.getLogger(__name__)


@dataclass
class VLLMWeightSyncCallback(Callback):
    """Callback to synchronize weights from training model to vLLM inference engines.

    After each training step, this callback:
    1. Pauses vLLM actors via actor_manager
    2. Gathers FSDP-sharded parameters using summon_full_params
    3. Broadcasts weights from rank 0 to vLLM engines
    4. Resumes vLLM actors
    """

    vllm_engines: list[Any] = field(default_factory=list)
    model_update_group: dist.ProcessGroup | None = None
    actor_manager: Any = None
    gather_whole_model: bool = True
    sync_interval: int = 1

    def post_step(self) -> None:
        if self.trainer.global_step % self.sync_interval != 0:
            return

        if not self.vllm_engines:
            return

        import ray

        torch.cuda.empty_cache()

        if self.actor_manager is not None:
            ray.get(self.actor_manager.pause.remote())

        model = self.trainer.train_module.model

        try:
            self._broadcast_weights(model)
        finally:
            if self.actor_manager is not None:
                ray.get(self.actor_manager.resume.remote())

    def _broadcast_weights(self, model: nn.Module) -> None:
        """Broadcast weights from training model to vLLM engines."""
        import ray

        refss = []
        count = 0
        num_params = len(list(model.named_parameters()))

        is_distributed = dist.is_initialized()
        is_rank0 = (not is_distributed) or (dist.get_rank() == 0)
        is_fsdp = isinstance(model, FSDP)

        if self.gather_whole_model and is_fsdp:
            with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
                for name, param in model.named_parameters():
                    count += 1
                    if is_rank0:
                        refs = [
                            engine.update_weight.remote(
                                name, dtype=str(param.dtype), shape=param.shape, empty_cache=(count == num_params)
                            )
                            for engine in self.vllm_engines
                        ]
                        refss.extend(refs)

                    if self.model_update_group is not None:
                        dist.broadcast(param.data, 0, group=self.model_update_group)
        elif is_fsdp:
            for name, param in model.named_parameters():
                count += 1
                with FSDP.summon_full_params(model, writeback=False, rank0_only=True):
                    if is_rank0:
                        refs = [
                            engine.update_weight.remote(
                                name, dtype=str(param.dtype), shape=param.shape, empty_cache=(count == num_params)
                            )
                            for engine in self.vllm_engines
                        ]
                        refss.extend(refs)

                    if self.model_update_group is not None:
                        dist.broadcast(param.data, 0, group=self.model_update_group)
        else:
            for name, param in model.named_parameters():
                count += 1
                if is_rank0:
                    refs = [
                        engine.update_weight.remote(
                            name, dtype=str(param.dtype), shape=param.shape, empty_cache=(count == num_params)
                        )
                        for engine in self.vllm_engines
                    ]
                    refss.extend(refs)

                if self.model_update_group is not None:
                    dist.broadcast(param.data, 0, group=self.model_update_group)

        if is_rank0 and refss:
            ray.get(refss)


@dataclass
class RefPolicyUpdateCallback(Callback):
    """Callback to update reference policy using Polyak averaging.

    Updates reference policy parameters as:
        ref_param = (1 - alpha) * ref_param + alpha * policy_param

    This is used for KL divergence computation in GRPO.
    """

    ref_policy: nn.Module | None = None
    alpha: float = 0.6
    update_interval: int = 1

    def post_step(self) -> None:
        if self.ref_policy is None:
            return

        if self.trainer.global_step % self.update_interval != 0:
            return

        model = self.trainer.train_module.model
        is_fsdp = isinstance(model, FSDP)

        if is_fsdp:
            with FSDP.summon_full_params(model, writeback=False, rank0_only=False):
                for ref_param, param in zip(self.ref_policy.parameters(), model.parameters(), strict=True):
                    ref_param.data.mul_(1.0 - self.alpha).add_(param.data, alpha=self.alpha)
        else:
            for ref_param, param in zip(self.ref_policy.parameters(), model.parameters(), strict=True):
                ref_param.data.mul_(1.0 - self.alpha).add_(param.data, alpha=self.alpha)


@dataclass
class DataPreparationActorCheckpointCallback(Callback):
    """Callback to save and restore DataPreparationActor state during checkpointing."""

    data_prep_actor_name: str = "data_prep_singleton"

    def pre_checkpoint(self) -> dict[str, Any]:
        """Save DataPreparationActor state before checkpointing."""
        import ray

        try:
            data_prep_actor = ray.get_actor(self.data_prep_actor_name)
            return {"data_prep_state": ray.get(data_prep_actor.get_state.remote())}
        except Exception as e:
            logger.warning(f"Failed to get DataPreparationActor state: {e}")
            return {}

    def post_checkpoint_load(self, state: dict[str, Any]) -> None:
        """Restore DataPreparationActor state after loading checkpoint."""
        import ray

        if "data_prep_state" not in state:
            return

        try:
            data_prep_actor = ray.get_actor(self.data_prep_actor_name)
            ray.get(data_prep_actor.restore_state.remote(state["data_prep_state"]))
            logger.info("Restored DataPreparationActor state from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to restore DataPreparationActor state: {e}")
