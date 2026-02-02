"""OLMo-core TrainModule classes for various training objectives.

This module provides training modules for DPO and GRPO using
OLMo-core's native training infrastructure.
"""

import math
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import EvalBatchSpec, TrainModule, TransformerTrainModule
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig
from transformers import PreTrainedTokenizer

from open_instruct import data_types, dpo_utils, grpo_utils, logger_utils, model_utils, utils
from open_instruct.rl_utils import masked_mean

logger = logger_utils.setup_logger(__name__)


class DPOTrainModule(TrainModule):
    """Training module for DPO with OLMo-core's Trainer.

    Uses OLMo-core's scheduler.set_lr() pattern for learning rate scheduling.
    """

    def __init__(
        self,
        model: Transformer,
        optim: torch.optim.Optimizer,
        args: dpo_utils.ExperimentConfig,
        reference_cache: model_utils.TensorCache,
        scheduler: Scheduler,
        device: torch.device | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
        self.args = args
        self.reference_cache = reference_cache
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm

        if args.packing:
            self._forward_fn = partial(dpo_utils.concatenated_forward_olmo, packing=True)
        elif args.concatenated_forward:
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
        self.optim.zero_grad()

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
        assert global_num_tokens is not None
        return flops_per_token * global_num_tokens

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(rank_batch_size=1)

    def eval_batch(self, batch: dict[str, Any], labels: Any | None = None) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(**batch)

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        self.model.train()

        policy_chosen_logps, policy_rejected_logps, aux_loss = self._forward_fn(
            self.model,
            batch,
            average_log_prob=self.args.loss_type.is_average_loss,
            output_router_logits=self.args.load_balancing_loss,
        )

        logger.info(f"[PACKING_DEBUG] packing={self.args.packing}")
        logger.info(f"[PACKING_DEBUG] batch indices: {batch.get('index', 'N/A')}")
        logger.info(f"[PACKING_DEBUG] chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
        logger.info(f"[PACKING_DEBUG] chosen_input_ids[:50]: {batch['chosen_input_ids'].flatten()[:50].tolist()}")
        logger.info(f"[PACKING_DEBUG] policy_chosen_logps: {policy_chosen_logps.tolist()}")
        logger.info(f"[PACKING_DEBUG] policy_rejected_logps: {policy_rejected_logps.tolist()}")

        losses, chosen_rewards, rejected_rewards = dpo_utils.compute_loss(
            self.args,
            batch,
            policy_chosen_logps,
            policy_rejected_logps,
            self.reference_cache if self.args.loss_type.needs_reference_model else None,
        )

        loss = losses.mean()

        if self.args.load_balancing_loss and aux_loss is not None:
            loss = loss + self.args.load_balancing_weight * aux_loss

        if not dry_run:
            self.record_metric("train/loss", loss.detach(), ReduceType.mean)
            self.record_metric("train/logps_chosen", policy_chosen_logps.mean().detach(), ReduceType.mean)
            self.record_metric("train/logps_rejected", policy_rejected_logps.mean().detach(), ReduceType.mean)

            if self.args.loss_type.computes_reward_metrics:
                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                margin = (chosen_rewards - rejected_rewards).mean()
                self.record_metric("train/rewards_chosen", chosen_rewards.mean().detach(), ReduceType.mean)
                self.record_metric("train/rewards_rejected", rejected_rewards.mean().detach(), ReduceType.mean)
                self.record_metric("train/rewards_accuracy", accuracy.detach(), ReduceType.mean)
                self.record_metric("train/rewards_margin", margin.detach(), ReduceType.mean)

            if self.args.load_balancing_loss and aux_loss is not None:
                self.record_metric("train/aux_loss", aux_loss.detach(), ReduceType.mean)

        loss.backward()


class GRPOTrainModule(TransformerTrainModule):
    """
    GRPO training module using OLMo-core Transformer models.

    Subclasses TransformerTrainModule to inherit:
    - optim_step with proper gradient clipping
    - zero_grads
    - eval_batch and eval_batch_spec
    - num_flops_per_token
    - state_dict/load_state_dict via dist_cp_sd


    Only train_batch differs (GRPO loss instead of CE loss).
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        rank_microbatch_size: int,
        max_sequence_length: int,
        grpo_config: grpo_utils.ExperimentConfig,
        tokenizer: PreTrainedTokenizer,
        ref_policy: Transformer | None = None,
        dp_config: TransformerDataParallelConfig | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
    ):
        super().__init__(
            model=model,
            optim=optim,
            rank_microbatch_size=rank_microbatch_size,
            max_sequence_length=max_sequence_length,
            dp_config=dp_config,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
        )

        self.grpo_config = grpo_config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        self.ref_policy = ref_policy
        if ref_policy is not None:
            self.ref_policy = ref_policy.to(device=self.device).eval().requires_grad_(False)

    def state_dict(self, *, optim: bool | None = None) -> dict[str, Any]:
        state = super().state_dict(optim=optim)
        if self.ref_policy is not None:
            state["ref_policy"] = self.ref_policy.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if "ref_policy" in state_dict and self.ref_policy is not None:
            self.ref_policy.load_state_dict(state_dict["ref_policy"])

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        """Execute one training step with GRPO loss.

        This method implements the GRPO training algorithm with:
        - Multi-epoch PPO-style training
        - DAPO/CISPO loss variants
        - KL penalty computation
        - Importance sampling with clipping
        """
        self.model.train()
        data_BT: data_types.CollatedBatchData = batch["batch"]

        with torch.no_grad():
            if self.grpo_config.load_ref_policy and self.ref_policy is not None:
                ref_logprobs_BT = grpo_utils.compute_logprobs(
                    self.ref_policy,
                    data_BT,
                    self.pad_token_id,
                    self.grpo_config.temperature,
                    use_grad=False,
                    batch_size=3 * self.rank_microbatch_size,
                )
            else:
                ref_logprobs_BT = None

        with torch.no_grad():
            old_logprobs_BT = grpo_utils.compute_logprobs(
                self.model,
                data_BT,
                self.pad_token_id,
                self.grpo_config.temperature,
                use_grad=False,
                batch_size=3 * self.rank_microbatch_size,
            )

        num_samples = len(data_BT.query_responses)
        accumulation_steps = max(math.ceil(num_samples / self.grpo_config.num_mini_batches), 1)

        if self.grpo_config.loss_denominator == "token" or self.grpo_config.loss_denominator is None:
            accumulation_token_counts = grpo_utils.calculate_token_counts(
                accumulation_steps, data_BT, self.device, self.trainer.dp_process_group
            )
        else:
            accumulation_token_counts = {
                int(group_idx * accumulation_steps): float(self.grpo_config.loss_denominator)
                for group_idx in range((num_samples // accumulation_steps) + 1)
            }

        dp_world_size = dist.get_world_size(self.trainer.dp_process_group) if self.trainer.dp_process_group else 1

        loss_stats_B = {
            "pg_loss": torch.zeros(num_samples, device=self.device),
            "kl": torch.zeros(num_samples, device=self.device),
            "clip_frac": torch.zeros(num_samples, device=self.device),
            "entropy": torch.zeros(num_samples, device=self.device),
            "token_count": torch.tensor(
                [data_BT.response_masks[i][:, 1:].sum().float() for i in range(num_samples)], device=self.device
            ),
        }

        num_steps = 0
        local_step = 0

        for _epoch_idx in range(self.grpo_config.num_epochs):
            for sample_idx in range(num_samples):
                new_logprobs, entropy = grpo_utils.forward_for_logprobs(
                    self.model,
                    data_BT.query_responses[sample_idx],
                    data_BT.attention_masks[sample_idx],
                    data_BT.position_ids[sample_idx],
                    self.pad_token_id,
                    self.grpo_config.temperature,
                    return_entropy=self.grpo_config.record_entropy,
                )

                response_mask = data_BT.response_masks[sample_idx][:, 1:].bool().to(new_logprobs.device)
                new_logprobs = torch.masked_fill(new_logprobs, ~response_mask, utils.INVALID_LOGPROB)

                old_logprobs = old_logprobs_BT[sample_idx]
                advantages = data_BT.advantages[sample_idx].to(new_logprobs.device)

                log_ratio = new_logprobs - old_logprobs
                ratio = torch.exp(log_ratio)

                pg_losses, pg_losses2, pg_loss, kl = grpo_utils.compute_grpo_loss(
                    new_logprobs=new_logprobs,
                    ratio=ratio,
                    advantages=advantages[:, 1:],
                    ref_logprobs=ref_logprobs_BT[sample_idx] if ref_logprobs_BT is not None else None,
                    config=self.grpo_config,
                )

                batch_start = (sample_idx // accumulation_steps) * accumulation_steps
                loss_denominator = accumulation_token_counts[batch_start]
                loss = masked_mean(pg_loss + self.grpo_config.beta * kl, response_mask, None, loss_denominator)

                loss = loss * dp_world_size
                loss.backward()

                with torch.no_grad():
                    loss_stats_B["pg_loss"][sample_idx] = masked_mean(pg_loss, response_mask)
                    loss_stats_B["kl"][sample_idx] = masked_mean(kl, response_mask)
                    loss_stats_B["clip_frac"][sample_idx] = (pg_losses2 > pg_losses).float().mean()
                    if entropy is not None:
                        loss_stats_B["entropy"][sample_idx] = entropy[response_mask].mean()

                num_steps += 1
                local_step += 1

                if local_step % accumulation_steps == 0:
                    if not dry_run:
                        self.optim_step()
                    self.zero_grads()

        if local_step % accumulation_steps != 0:
            if not dry_run:
                self.optim_step()
            self.zero_grads()

        if not dry_run and num_steps > 0:
            local_token_counts = loss_stats_B["token_count"]

            local_pg_loss_sum = (loss_stats_B["pg_loss"] * local_token_counts).sum()
            local_kl_sum = (loss_stats_B["kl"] * local_token_counts).sum()
            local_clip_frac_sum = (loss_stats_B["clip_frac"] * local_token_counts).sum()
            local_total_tokens = local_token_counts.sum()

            local_sums_list = [local_total_tokens, local_pg_loss_sum, local_kl_sum, local_clip_frac_sum]
            if self.grpo_config.record_entropy:
                local_entropy_sum = (loss_stats_B["entropy"] * local_token_counts).sum()
                local_sums_list.append(local_entropy_sum)

            local_sums = torch.stack(local_sums_list)
            dist.all_reduce(local_sums, op=dist.ReduceOp.SUM, group=self.trainer.dp_process_group)

            global_total_tokens, global_pg_loss_sum, global_kl_sum, global_clip_frac_sum = local_sums[:4]

            self.record_metric("train/pg_loss", (global_pg_loss_sum / global_total_tokens).item(), reduce_type=None)
            self.record_metric("train/kl", (global_kl_sum / global_total_tokens).item(), reduce_type=None)
            self.record_metric(
                "train/clip_frac", (global_clip_frac_sum / global_total_tokens).item(), reduce_type=None
            )
            if self.grpo_config.record_entropy:
                global_entropy_sum = local_sums[4]
                self.record_metric(
                    "train/entropy", (global_entropy_sum / global_total_tokens).item(), reduce_type=None
                )

    def global_num_flops_in_batch(self, batch: dict[str, Any]) -> int:
        data_BT: data_types.CollatedBatchData = batch["batch"]
        seq_len = data_BT.query_responses[0].shape[1]
        flops_per_token = self.num_flops_per_token(seq_len)
        global_num_tokens = self.trainer.data_loader.global_num_tokens_in_batch(batch)
        assert global_num_tokens is not None
        return flops_per_token * global_num_tokens
