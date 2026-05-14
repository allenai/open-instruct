"""OLMo-core TrainModule classes for various training objectives.

This module provides training modules for DPO and GRPO using
OLMo-core's native training infrastructure.
"""

import math
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import wandb
from olmo_core.distributed import utils as dist_utils
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMHead, LMOutputWithLoss
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.train_module import TransformerTrainModule
from olmo_core.train.train_module.transformer import config as transformer_config
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers import PreTrainedTokenizer

from open_instruct import data_loader as data_loader_lib
from open_instruct import dpo_utils, grpo_utils, logger_utils, model_utils, padding_free_collator
from open_instruct.rl_utils import masked_mean

logger = logger_utils.setup_logger(__name__)


_DOC_LENS_ATTN_BACKENDS = frozenset(
    {AttentionBackendName.flash_2, AttentionBackendName.flash_3, AttentionBackendName.flash_4}
)


class DPOLMHead(LMHead):
    """LM head that returns per-token log-probabilities for DPO training.

    All DTensor handling happens inside this module (which is torch.compiled),
    avoiding DTensor/compile backward incompatibilities in the DPO loss code.
    Returns per-token logps with the same semantics as calculate_per_token_logps:
    output[i] = log p(labels[i+1] | logits[i]), with label shifting done internally.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        ignore_index: int = -100,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: float | None = None,
        loss_div_factor: torch.Tensor | float | None = None,
        return_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
    ) -> torch.Tensor | LMOutputWithLoss:
        if labels is None:
            return super().forward(
                x,
                labels=labels,
                loss_reduction=loss_reduction,
                z_loss_multiplier=z_loss_multiplier,
                loss_div_factor=loss_div_factor,
                return_logits=return_logits,
                logits_to_keep=logits_to_keep,
            )

        h = self.norm(x) if self.norm is not None else x
        logits = self.w_out(h)

        local_logits = dist_utils.get_local_tensor(logits)
        local_labels = dist_utils.get_local_tensor(labels)
        per_token_logps = padding_free_collator.calculate_per_token_logps(local_logits, local_labels)
        if self.tp_enabled:
            per_token_logps = (
                DTensor.from_local(per_token_logps, self._tp_mesh, (Shard(1),))
                .redistribute(placements=(Replicate(),))
                .to_local()
            )
        return LMOutputWithLoss(
            logits=None, loss=per_token_logps, ce_loss=per_token_logps.detach().clone(), z_loss=None
        )


def split_batch_dpo(batch: dict[str, Any], num_microbatch_instances: int) -> list[dict[str, Any]]:
    """Split a DPO batch into micro-batches using chosen_input_ids as the reference."""
    if num_microbatch_instances <= 0:
        raise RuntimeError("microbatch size is too small!")

    batch_size = batch["chosen_input_ids"].shape[0]
    if batch_size <= num_microbatch_instances:
        return [batch]

    micro_batches = {k: v.split(num_microbatch_instances, dim=0) for k, v in batch.items() if k != "input_ids"}

    return [
        {key: value[i] for key, value in micro_batches.items()} for i in range(len(micro_batches["chosen_input_ids"]))
    ]


class DPOTrainModule(TransformerTrainModule):
    """Training module for DPO with OLMo-core's Trainer.

    Subclasses TransformerTrainModule to inherit:
    - optim_step with proper gradient clipping and scheduler support
    - zero_grads
    - eval_batch and eval_batch_spec
    - num_flops_per_token
    - state_dict/load_state_dict via dist_cp_sd
    - _train_microbatch_context for FSDP/DDP sync control
    """

    def __init__(
        self,
        model: Transformer,
        optim: OptimConfig,
        sample_microbatch_size: int,
        max_sequence_length: int,
        dpo_config: dpo_utils.DPOExperimentConfig,
        attn_implementation: AttentionBackendName,
        dp_config: transformer_config.TransformerDataParallelConfig | None = None,
        tp_config: transformer_config.TransformerTensorParallelConfig | None = None,
        cp_config: transformer_config.TransformerContextParallelConfig | None = None,
        ac_config: transformer_config.TransformerActivationCheckpointingConfig | None = None,
        compile_model: bool = True,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
    ) -> None:
        if dpo_config.packing:
            assert attn_implementation in _DOC_LENS_ATTN_BACKENDS, (
                f"DPOTrainModule with packing requires a flash attention backend for intra-document "
                f"masking via doc_lens/max_doc_lens; got {attn_implementation}."
            )
        # TODO(finbarrtimbers): Remove this hack once Transformer supports configuring the LM head.
        model.lm_head.__class__ = DPOLMHead
        rank_microbatch_size_tokens = sample_microbatch_size * max_sequence_length * 2
        super().__init__(
            model=model,
            optim=optim,
            rank_microbatch_size=rank_microbatch_size_tokens,
            max_sequence_length=max_sequence_length,
            dp_config=dp_config,
            tp_config=tp_config,
            cp_config=cp_config,
            ac_config=ac_config,
            compile_model=compile_model,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
        )

        self.sample_microbatch_size = sample_microbatch_size
        self.dpo_config = dpo_config
        self.reference_cache: model_utils.TensorCache | None = None

        self._metrics: dict[str, torch.Tensor] = {
            k: torch.tensor(0.0, device=device)
            for k in ["loss", "chosen_logps", "rejected_logps", "chosen_rewards", "rejected_rewards", "accuracy"]
        }
        if dpo_config.load_balancing_loss:
            self._metrics["aux_loss"] = torch.tensor(0.0, device=device)

        if dpo_config.concatenated_forward or dpo_config.packing:
            self._forward_fn = dpo_utils.concatenated_forward_olmo
        else:
            self._forward_fn = dpo_utils.separate_forward_olmo
        self._forward_kwargs: dict[str, Any] = {}
        if dpo_config.packing:
            self._forward_kwargs["packing"] = True

    def pre_train(self):
        pass

    def global_num_flops_in_batch(self, batch: dict[str, Any]) -> int | None:
        global_num_tokens = self.trainer.data_loader.global_num_tokens_in_batch(batch)
        if global_num_tokens is None:
            return None
        seq_len = batch["chosen_input_ids"].shape[1]
        flops_per_token = self.num_flops_per_token(seq_len=seq_len)
        return flops_per_token * global_num_tokens if flops_per_token is not None else None

    def _compute_microbatch_loss(self, micro_batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        policy_chosen_logps, policy_rejected_logps, aux_loss = self._forward_fn(
            self.model,
            micro_batch,
            average_log_prob=self.dpo_config.loss_type.is_average_loss,
            output_router_logits=self.dpo_config.load_balancing_loss,
            **self._forward_kwargs,
        )

        losses, chosen_rewards, rejected_rewards = dpo_utils.compute_loss(
            self.dpo_config,
            micro_batch,
            policy_chosen_logps,
            policy_rejected_logps,
            self.reference_cache if self.dpo_config.loss_type.needs_reference_model else None,
        )

        loss = losses.mean()
        if self.dpo_config.load_balancing_loss and aux_loss is not None:
            loss = loss + self.dpo_config.load_balancing_weight * aux_loss

        step_metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "chosen_logps": policy_chosen_logps.mean(),
            "rejected_logps": policy_rejected_logps.mean(),
            "chosen_rewards": chosen_rewards.mean(),
            "rejected_rewards": rejected_rewards.mean(),
            "accuracy": (chosen_rewards > rejected_rewards).float().mean(),
        }
        if aux_loss is not None and "aux_loss" in self._metrics:
            step_metrics["aux_loss"] = aux_loss

        return loss, step_metrics

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        self.model.train()

        micro_batches = split_batch_dpo(batch, self.sample_microbatch_size)
        num_micro_batches = len(micro_batches)
        device = batch["chosen_input_ids"].device
        total_tokens = padding_free_collator.get_num_tokens(batch)

        for v in self._metrics.values():
            v.zero_()

        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                loss, step_metrics = self._compute_microbatch_loss(micro_batch)
                micro_tokens = padding_free_collator.get_num_tokens(micro_batch)
                weight = micro_tokens / total_tokens
                for k, v in step_metrics.items():
                    self._metrics[k] += v.detach() * micro_tokens
                (loss * weight).backward()

        self.model.post_batch(dry_run=dry_run)

        if not dry_run:
            metric_keys = sorted(self._metrics.keys())
            local_sums_list = [torch.tensor(total_tokens, dtype=torch.float32, device=device)] + [
                self._metrics[k] for k in metric_keys
            ]
            local_sums = torch.stack(local_sums_list)
            dist.all_reduce(local_sums, op=dist.ReduceOp.SUM, group=self.trainer.dp_process_group)

            global_total_tokens = local_sums[0]
            global_metrics = {k: local_sums[i + 1] / global_total_tokens for i, k in enumerate(metric_keys)}

            self.record_metric("train/loss", global_metrics["loss"].item(), reduce_type=None)
            self.record_metric("train/logps_chosen", global_metrics["chosen_logps"].item(), reduce_type=None)
            self.record_metric("train/logps_rejected", global_metrics["rejected_logps"].item(), reduce_type=None)
            token_count = self.trainer.data_loader.global_num_tokens_in_batch(batch)
            assert token_count is not None
            self.record_metric("train/token_count", token_count, reduce_type=None)

            if self.dpo_config.loss_type.computes_reward_metrics:
                margin = global_metrics["chosen_rewards"] - global_metrics["rejected_rewards"]
                self.record_metric("train/rewards_chosen", global_metrics["chosen_rewards"].item(), reduce_type=None)
                self.record_metric(
                    "train/rewards_rejected", global_metrics["rejected_rewards"].item(), reduce_type=None
                )
                self.record_metric(
                    "train/rewards_average",
                    ((global_metrics["chosen_rewards"] + global_metrics["rejected_rewards"]) / 2).item(),
                    reduce_type=None,
                )
                self.record_metric("train/rewards_accuracy", global_metrics["accuracy"].item(), reduce_type=None)
                self.record_metric("train/rewards_margin", margin.item(), reduce_type=None)

            if "aux_loss" in global_metrics:
                self.record_metric("train/aux_loss", global_metrics["aux_loss"].item(), reduce_type=None)


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
        sample_microbatch_size: int,
        max_sequence_length: int,
        grpo_config: grpo_utils.GRPOExperimentConfig,
        temperature: float,
        tokenizer: PreTrainedTokenizer,
        streaming_config: data_loader_lib.StreamingDataLoaderConfig,
        attn_implementation: AttentionBackendName,
        ref_policy: Transformer | None = None,
        dp_config: transformer_config.TransformerDataParallelConfig | None = None,
        ac_config: transformer_config.TransformerActivationCheckpointingConfig | None = None,
        compile_model: bool = True,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
    ):
        assert attn_implementation in _DOC_LENS_ATTN_BACKENDS, (
            f"GRPOTrainModule requires a flash attention backend for intra-document masking via "
            f"doc_lens/max_doc_lens; got {attn_implementation}."
        )
        rank_microbatch_size_tokens = sample_microbatch_size * max_sequence_length
        super().__init__(
            model=model,
            optim=optim,
            rank_microbatch_size=rank_microbatch_size_tokens,
            max_sequence_length=max_sequence_length,
            dp_config=dp_config,
            ac_config=ac_config,
            compile_model=compile_model,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
        )

        self.sample_microbatch_size = sample_microbatch_size
        self.grpo_config = grpo_config
        self.temperature = temperature
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.attn_implementation = attn_implementation

        self.ref_policy = ref_policy
        if ref_policy is not None:
            self.ref_policy = ref_policy.to(device=self.device).eval().requires_grad_(False)

        self.streaming_config = streaming_config
        self._num_total_tokens = 0
        self._grad_norms: list[float] = []
        self._last_num_step_tokens: int = 0

    def pre_train(self):
        # GRPO batches are prompt-grouped and do their own accumulation/token normalization
        # inside train_batch(), so the base TransformerTrainModule global-batch validation
        # does not apply here.
        pass

    def optim_step(self) -> None:
        # No-op: train_batch invokes _do_optim_step internally per accumulation boundary.
        pass

    def zero_grads(self) -> None:
        # No-op: train_batch zeroes grads internally after each optim step.
        pass

    def _do_optim_step(self) -> None:
        grad_norm = None
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
        if self.scheduler is not None:
            for group in self.optim.param_groups:
                self.scheduler.set_lr(group, self.trainer)
        self.optim.step()
        self.model.post_optim_step()
        if grad_norm is not None:
            grad_norm_val = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
            self._grad_norms.append(float(grad_norm_val))

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
        data_BT = batch["batch"].to(self.device)

        with torch.no_grad():
            if self.grpo_config.load_ref_policy and self.ref_policy is not None:
                ref_logprobs_BT = grpo_utils.compute_logprobs(
                    self.ref_policy,
                    data_BT,
                    self.pad_token_id,
                    self.temperature,
                    use_grad=False,
                    batch_size=3 * self.rank_microbatch_size,
                    pass_olmo_core_doc_lens=True,
                )
            else:
                ref_logprobs_BT = None

        num_samples = len(data_BT.query_responses)
        num_mini_batches = self.grpo_config.num_mini_batches
        accumulation_steps = max(math.ceil(num_samples / num_mini_batches), 1)

        old_logprobs_BT: list[torch.Tensor | None] = [None for _ in range(num_samples)]
        if num_mini_batches > 1:
            with torch.no_grad():
                local_old_logprobs_BT = None
                if not self.grpo_config.use_vllm_logprobs:
                    local_old_logprobs_BT = grpo_utils.compute_logprobs(
                        self.model,
                        data_BT,
                        self.pad_token_id,
                        self.temperature,
                        use_grad=False,
                        batch_size=3 * self.rank_microbatch_size,
                        pass_olmo_core_doc_lens=True,
                    )

                for i in range(num_samples):
                    if self.grpo_config.use_vllm_logprobs:
                        old_logprobs_BT[i] = grpo_utils.mask_logprobs(
                            data_BT.vllm_logprobs[i][:, 1:], data_BT.response_masks[i][:, 1:]
                        )
                    else:
                        assert local_old_logprobs_BT is not None
                        old_logprobs_BT[i] = local_old_logprobs_BT[i]

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

        token_counts = torch.tensor(
            [data_BT.response_masks[i][:, 1:].sum().float() for i in range(num_samples)], device=self.device
        )
        loss_stats_B = grpo_utils.create_loss_stats(
            num_samples, self.device, record_entropy=self.grpo_config.record_entropy
        )

        debug_metrics_sum: dict[str, float] = {}
        debug_metrics_count = 0

        num_steps = 0
        local_step = 0
        rho_histograms: dict[str, list[torch.Tensor]] = {}

        for epoch_idx in range(self.grpo_config.num_epochs):
            for sample_idx in range(num_samples):
                new_logprobs, entropy = grpo_utils.forward_for_logprobs(
                    self.model,
                    data_BT.query_responses[sample_idx],
                    data_BT.attention_masks[sample_idx],
                    data_BT.position_ids[sample_idx],
                    self.pad_token_id,
                    self.temperature,
                    return_entropy=self.grpo_config.record_entropy,
                    pass_olmo_core_doc_lens=True,
                )

                response_mask = data_BT.response_masks[sample_idx][:, 1:]
                new_logprobs = grpo_utils.mask_logprobs(new_logprobs, response_mask)

                vllm_logprobs = grpo_utils.mask_logprobs(data_BT.vllm_logprobs[sample_idx][:, 1:], response_mask)

                step_debug_metrics = grpo_utils.compute_vllm_local_debug_metrics(
                    local_logprobs=new_logprobs, vllm_logprobs=vllm_logprobs, response_mask=response_mask
                )
                for k, v in step_debug_metrics.items():
                    debug_metrics_sum[k] = debug_metrics_sum.get(k, 0.0) + v
                debug_metrics_count += 1

                old_logprob = grpo_utils.resolve_old_logprob(
                    old_logprobs_BT,
                    sample_idx,
                    epoch_idx,
                    num_mini_batches,
                    self.grpo_config.use_vllm_logprobs,
                    vllm_logprobs,
                    new_logprobs,
                )

                advantages = data_BT.advantages[sample_idx]

                log_ratio = new_logprobs - old_logprob
                ratio = torch.exp(log_ratio)

                rho = grpo_utils.compute_rho_correction(old_logprob, vllm_logprobs, response_mask, self.grpo_config)
                grpo_utils.accumulate_rho_histograms(rho_histograms, rho)

                pg_losses, pg_losses2, pg_loss, kl = grpo_utils.compute_grpo_loss(
                    new_logprobs=new_logprobs,
                    ratio=ratio,
                    advantages=advantages[:, 1:],
                    ref_logprobs=ref_logprobs_BT[sample_idx] if ref_logprobs_BT is not None else None,
                    config=self.grpo_config,
                    rho_weights=rho.weights,
                )

                batch_start = (sample_idx // accumulation_steps) * accumulation_steps
                loss_denominator = accumulation_token_counts[batch_start]
                loss = masked_mean(pg_loss + self.grpo_config.beta * kl, response_mask, None, loss_denominator)

                loss = loss * dp_world_size
                loss.backward()

                grpo_utils.populate_sample_loss_stats(
                    loss_stats_B,
                    sample_idx,
                    pg_losses,
                    pg_losses2,
                    pg_loss,
                    ratio,
                    loss,
                    response_mask,
                    new_logprobs,
                    ref_logprobs_BT[sample_idx] if ref_logprobs_BT is not None else None,
                    entropy,
                    self.grpo_config,
                    rho_metrics=rho.metrics,
                )

                num_steps += 1
                local_step += 1

                if local_step % accumulation_steps == 0:
                    if not dry_run:
                        self._do_optim_step()
                    self.optim.zero_grad(set_to_none=True)

        if local_step % accumulation_steps != 0:
            if not dry_run:
                self._do_optim_step()
            self.optim.zero_grad(set_to_none=True)

        if not dry_run and num_steps > 0:
            local_metrics = grpo_utils.compute_metrics_from_loss_stats(loss_stats_B, token_counts)
            local_tokens = token_counts.sum().item()

            keys = sorted(local_metrics.keys())
            values = [local_tokens] + [local_metrics[k] * local_tokens for k in keys]
            tensor = torch.tensor(values, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.trainer.dp_process_group)

            global_tokens = tensor[0].item()
            for i, k in enumerate(keys):
                self.record_metric(k, (tensor[i + 1] / global_tokens).item(), reduce_type=None)
            if self.scheduler is not None and self.trainer.max_steps is not None:
                lr = self.scheduler.get_lr(
                    self.optim.param_groups[0].get("initial_lr", self.optim.param_groups[0]["lr"]),
                    self.trainer.global_step,
                    self.trainer.max_steps,
                )
                self.record_metric("lr", float(lr), reduce_type=None)
            self.record_metric("_token_count", global_tokens, reduce_type=None)

            self._record_step_counter_metrics(int(global_tokens))
            data_prep_metrics = dict(batch.get("metrics") or {})
            data_prep_metrics.update(grpo_utils.finalize_rho_histograms(rho_histograms))
            self._record_data_prep_metrics(data_prep_metrics)
            for k, v in debug_metrics_sum.items():
                self.record_metric(k, v / debug_metrics_count, reduce_type=None)

            if self._grad_norms:
                self.record_metric("optim/grad_norm", sum(self._grad_norms) / len(self._grad_norms), reduce_type=None)
            self._grad_norms = []
            self._last_num_step_tokens = int(global_tokens)

    def _record_step_counter_metrics(self, global_tokens: int) -> None:
        self._num_total_tokens += global_tokens
        self.record_metric("training_step", float(self.trainer.global_step), reduce_type=None)
        self.record_metric("global_step", float(self.trainer.global_step), reduce_type=None)
        self.record_metric("val/num_step_tokens", float(global_tokens), reduce_type=None)
        self.record_metric("val/num_total_tokens", float(self._num_total_tokens), reduce_type=None)

        samples_per_step = (
            self.streaming_config.num_unique_prompts_rollout * self.streaming_config.num_samples_per_prompt_rollout
        )
        episode = self.trainer.global_step * samples_per_step
        self.record_metric("episode", float(episode), reduce_type=None)

    def _record_data_prep_metrics(self, data_prep_metrics: dict[str, Any]) -> None:
        histogram_metrics: dict[str, Any] = {}
        for metric_key, metric_value in data_prep_metrics.items():
            if isinstance(metric_value, (bool, int, float, np.integer, np.floating)):
                self.record_metric(metric_key, float(metric_value), reduce_type=None)
            elif isinstance(metric_value, np.ndarray):
                histogram_metrics[metric_key] = metric_value
            elif isinstance(metric_value, list) and metric_value and isinstance(metric_value[0], (int, float)):
                histogram_metrics[metric_key] = np.asarray(metric_value)

        if histogram_metrics and dist_utils.get_rank() == 0 and wandb.run is not None:
            wandb.log(
                {k: wandb.Histogram(v) for k, v in histogram_metrics.items()},  # ty: ignore[invalid-argument-type]
                step=self.trainer.global_step,
            )
