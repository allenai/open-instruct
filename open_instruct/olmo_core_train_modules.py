"""OLMo-core TrainModule classes for various training objectives."""

import contextlib
import math
from collections.abc import Generator
from typing import Any

import torch
import torch.nn as nn
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import EvalBatchSpec, TrainModule
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP

from open_instruct import dpo_utils, model_utils


def split_batch_dpo(batch: dict[str, Any], num_microbatch_instances: int) -> list[dict[str, Any]]:
    """Split a DPO batch into micro-batches using chosen_input_ids as the reference."""
    if num_microbatch_instances <= 0:
        raise RuntimeError("microbatch size is too small!")

    batch_size = batch["chosen_input_ids"].shape[0]
    if batch_size <= num_microbatch_instances:
        return [batch]

    micro_batches: dict[str, list] = {}
    for key, value in batch.items():
        if key == "input_ids":
            continue
        if isinstance(value, torch.Tensor):
            micro_batches[key] = value.split(num_microbatch_instances, dim=0)
        elif isinstance(value, list):
            micro_batches[key] = [
                value[num_microbatch_instances * i : num_microbatch_instances * (i + 1)]
                for i in range(math.ceil(batch_size / num_microbatch_instances))
            ]
        else:
            raise RuntimeError(f"unexpected item in batch: '{key}={value}'")

    return [
        {key: value[i] for key, value in micro_batches.items()} for i in range(len(micro_batches["chosen_input_ids"]))
    ]


class DPOTrainModule(TrainModule):
    """Training module for DPO with OLMo-core's Trainer.

    Uses OLMo-core's scheduler.set_lr() pattern for learning rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        args: dpo_utils.ExperimentConfig,
        reference_cache: model_utils.TensorCache,
        scheduler: Scheduler,
        rank_microbatch_size: int,
        device: torch.device | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
        self.args = args
        self.reference_cache = reference_cache
        self.scheduler = scheduler
        self.rank_microbatch_size = rank_microbatch_size
        self.device = device
        self.max_grad_norm = max_grad_norm

        if args.concatenated_forward:
            self._forward_fn = dpo_utils.concatenated_forward_olmo
        else:
            self._forward_fn = dpo_utils.separate_forward_olmo

    def state_dict(self, *, optim: bool | None = None) -> dict[str, Any]:
        state_dict: dict[str, Any] = {"model": self.model.state_dict()}
        if optim is not False:
            state_dict["optim"] = self.optim.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict["model"])
        if "optim" in state_dict:
            self.optim.load_state_dict(state_dict["optim"])

    def zero_grads(self) -> None:
        self.optim.zero_grad(set_to_none=True)

    def optim_step(self) -> None:
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.trainer.record_metric("total grad norm", grad_norm, reduce_type=None, namespace="optim")
        for group_idx, group in enumerate(self.optim.param_groups):
            new_lr = self.scheduler.set_lr(group, self.trainer)
            self.trainer.record_metric(f"LR (group {group_idx})", new_lr, namespace="optim")
        self.optim.step()

    def num_flops_per_token(self, seq_len: int) -> int:
        return self.model.num_flops_per_token(seq_len)

    def global_num_flops_in_batch(self, batch: dict[str, Any]) -> int:
        seq_len = batch["input_ids"].shape[1]
        flops_per_token = self.num_flops_per_token(seq_len)
        global_num_tokens = self.trainer.data_loader.global_num_tokens_in_batch(batch)
        return flops_per_token * global_num_tokens

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(rank_batch_size=1)

    def eval_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(**batch)

    @contextlib.contextmanager
    def _train_microbatch_context(self, micro_batch_idx: int, num_micro_batches: int) -> Generator[None, None, None]:
        is_last_mb = micro_batch_idx == num_micro_batches - 1
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, FSDPModule):
                self.model.set_is_last_backward(is_last_mb)
            elif isinstance(self.model, DDP) and not is_last_mb:
                stack.enter_context(self.model.no_sync())
            yield

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        self.model.train()

        micro_batches = split_batch_dpo(batch, self.rank_microbatch_size)
        num_micro_batches = len(micro_batches)

        total_loss = torch.tensor(0.0, device=self.device)
        total_chosen_logps = torch.tensor(0.0, device=self.device)
        total_rejected_logps = torch.tensor(0.0, device=self.device)
        total_chosen_rewards = torch.tensor(0.0, device=self.device)
        total_rejected_rewards = torch.tensor(0.0, device=self.device)
        total_aux_loss = torch.tensor(0.0, device=self.device) if self.args.load_balancing_loss else None

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                policy_chosen_logps, policy_rejected_logps, aux_loss = self._forward_fn(
                    self.model,
                    micro_batch,
                    average_log_prob=self.args.loss_type.is_average_loss,
                    output_router_logits=self.args.load_balancing_loss,
                )

                losses, chosen_rewards, rejected_rewards = dpo_utils.compute_loss(
                    self.args,
                    micro_batch,
                    policy_chosen_logps,
                    policy_rejected_logps,
                    self.reference_cache if self.args.loss_type.needs_reference_model else None,
                )

                loss = losses.mean()
                if self.args.load_balancing_loss and aux_loss is not None:
                    loss = loss + self.args.load_balancing_weight * aux_loss

                loss = loss / num_micro_batches

                total_loss += loss.detach()
                total_chosen_logps += policy_chosen_logps.mean().detach() / num_micro_batches
                total_rejected_logps += policy_rejected_logps.mean().detach() / num_micro_batches
                if self.args.loss_type.computes_reward_metrics:
                    total_chosen_rewards += chosen_rewards.mean().detach() / num_micro_batches
                    total_rejected_rewards += rejected_rewards.mean().detach() / num_micro_batches
                if total_aux_loss is not None and aux_loss is not None:
                    total_aux_loss += aux_loss.detach() / num_micro_batches

                loss.backward()

        if not dry_run:
            self.record_metric("train/loss", total_loss, ReduceType.mean)
            self.record_metric("train/logps_chosen", total_chosen_logps, ReduceType.mean)
            self.record_metric("train/logps_rejected", total_rejected_logps, ReduceType.mean)

            if self.args.loss_type.computes_reward_metrics:
                accuracy = (total_chosen_rewards > total_rejected_rewards).float()
                margin = total_chosen_rewards - total_rejected_rewards
                self.record_metric("train/rewards_chosen", total_chosen_rewards, ReduceType.mean)
                self.record_metric("train/rewards_rejected", total_rejected_rewards, ReduceType.mean)
                self.record_metric("train/rewards_accuracy", accuracy, ReduceType.mean)
                self.record_metric("train/rewards_margin", margin, ReduceType.mean)

            if total_aux_loss is not None:
                self.record_metric("train/aux_loss", total_aux_loss, ReduceType.mean)
