"""
GRPO training module using OLMo-core's Trainer infrastructure.

This module provides GRPO (Group Relative Policy Optimization) training using
OLMo-core's native training infrastructure, subclassing TransformerTrainModule.
"""

from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
import torch.nn.functional as F
from olmo_core.nn.transformer import Transformer
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import TransformerTrainModule
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig
from transformers import PreTrainedTokenizer

from open_instruct import data_types, grpo_utils, logger_utils

logger = logger_utils.setup_logger(__name__)


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
            self.ref_policy = ref_policy.to(device=self.device).eval()

    def state_dict(self, *, optim: bool | None = None) -> dict[str, Any]:
        state = super().state_dict(optim=optim)
        if self.ref_policy is not None:
            state["ref_policy"] = self.ref_policy.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if "ref_policy" in state_dict and self.ref_policy is not None:
            self.ref_policy.load_state_dict(state_dict["ref_policy"])

    def forward(
        self,
        model: nn.Module,
        query_responses: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        pad_token_id: int,
        temperature: float,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass to compute log probabilities.

        Args:
            model: The model to use for forward pass.
            query_responses: Input token IDs [batch, seq_len].
            attention_mask: Attention mask [batch, seq_len].
            position_ids: Position IDs [batch, seq_len].
            pad_token_id: Padding token ID.
            temperature: Temperature for logprobs.
            return_entropy: Whether to return entropy.

        Returns:
            Tuple of (logprobs [batch, seq_len], entropy or None).
        """
        device = next(model.parameters()).device
        query_responses = query_responses.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if position_ids is not None:
            position_ids = position_ids.to(device)

        output = model(input_ids=query_responses, attention_mask=attention_mask, position_ids=position_ids)
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits / temperature
        logits = logits[:, :-1]
        labels = query_responses[:, 1:].clone()
        labels[labels == pad_token_id] = 0
        log_probs = F.log_softmax(logits, dim=-1)
        logprob_BT = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        entropy = None
        if return_entropy:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)

        return logprob_BT, entropy

    def compute_logprobs(
        self, model: nn.Module, data_BT: data_types.CollatedBatchData, use_grad: bool = False
    ) -> list[torch.Tensor]:
        """Compute log probabilities for all samples in batch."""
        logprobs_BT: list[torch.Tensor] = []
        ctx = torch.enable_grad() if use_grad else torch.no_grad()

        with ctx:
            for i in range(len(data_BT.query_responses)):
                logprob_BT, _ = self.forward(
                    model,
                    data_BT.query_responses[i],
                    data_BT.attention_masks[i],
                    data_BT.position_ids[i],
                    self.pad_token_id,
                    self.grpo_config.temperature,
                    return_entropy=False,
                )
                response_mask_BT = data_BT.response_masks[i].to(logprob_BT.device)
                logprob_BT = torch.where(
                    response_mask_BT[:, 1:].bool(),
                    logprob_BT,
                    torch.tensor(0.0, device=logprob_BT.device, dtype=logprob_BT.dtype),
                )
                logprobs_BT.append(logprob_BT)

        return logprobs_BT

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
                ref_logprobs_BT = self.compute_logprobs(self.ref_policy, data_BT, use_grad=False)
            else:
                ref_logprobs_BT = None

        with torch.no_grad():
            old_logprobs_BT = self.compute_logprobs(self.model, data_BT, use_grad=False)

        num_samples = len(data_BT.query_responses)
        accumulation_steps = num_samples * self.grpo_config.num_epochs * self.grpo_config.num_mini_batches

        total_pg_loss = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        total_clip_frac = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        num_steps = 0

        for _epoch_idx in range(self.grpo_config.num_epochs):
            for sample_idx in range(num_samples):
                new_logprobs, entropy = self.forward(
                    self.model,
                    data_BT.query_responses[sample_idx],
                    data_BT.attention_masks[sample_idx],
                    data_BT.position_ids[sample_idx],
                    self.pad_token_id,
                    self.grpo_config.temperature,
                    return_entropy=self.grpo_config.record_entropy,
                )

                response_mask = data_BT.response_masks[sample_idx][:, 1:].bool().to(new_logprobs.device)
                new_logprobs = torch.where(
                    response_mask,
                    new_logprobs,
                    torch.tensor(0.0, device=new_logprobs.device, dtype=new_logprobs.dtype),
                )

                old_logprobs = old_logprobs_BT[sample_idx]
                advantages = data_BT.advantages[sample_idx].to(new_logprobs.device)

                log_ratio = new_logprobs - old_logprobs
                ratio = torch.exp(log_ratio)

                if self.grpo_config.loss_fn == "dapo":
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - self.grpo_config.clip_lower, 1.0 + self.grpo_config.clip_higher
                    )
                    pg_loss = -advantages.unsqueeze(-1) * torch.min(ratio, clipped_ratio)
                else:
                    clipped_ratio = torch.clamp(ratio, max=1.0 + self.grpo_config.clip_higher)
                    pg_loss = -advantages.unsqueeze(-1) * clipped_ratio * new_logprobs

                if ref_logprobs_BT is not None:
                    ref_logprobs = ref_logprobs_BT[sample_idx]
                    kl = self._estimate_kl(new_logprobs - ref_logprobs, ratio, self.grpo_config.kl_estimator)
                    loss = pg_loss + self.grpo_config.beta * kl
                else:
                    kl = torch.zeros_like(pg_loss)
                    loss = pg_loss

                num_tokens = response_mask.sum()
                if self.grpo_config.loss_denominator == "token" or self.grpo_config.loss_denominator is None:
                    loss = loss.sum() / max(num_tokens, 1)
                else:
                    loss = loss.sum() / float(self.grpo_config.loss_denominator)

                loss = loss / accumulation_steps
                loss.backward()

                total_pg_loss = total_pg_loss + pg_loss.sum().detach()
                total_kl = total_kl + kl.sum().detach()
                clip_frac = (
                    ((ratio < 1.0 - self.grpo_config.clip_lower) | (ratio > 1.0 + self.grpo_config.clip_higher))
                    .float()
                    .mean()
                )
                total_clip_frac = total_clip_frac + clip_frac.detach()
                if entropy is not None:
                    total_entropy = total_entropy + entropy[response_mask].mean().detach()
                num_steps += 1

        if not dry_run:
            self.optim_step()

        self.zero_grads()

        if not dry_run and num_steps > 0:
            self.record_metric("train/pg_loss", (total_pg_loss / num_steps).item(), ReduceType.mean)
            self.record_metric("train/kl", (total_kl / num_steps).item(), ReduceType.mean)
            self.record_metric("train/clip_frac", (total_clip_frac / num_steps).item(), ReduceType.mean)
            if self.grpo_config.record_entropy:
                self.record_metric("train/entropy", (total_entropy / num_steps).item(), ReduceType.mean)

    def _estimate_kl(self, log_ratio: torch.Tensor, ratio: torch.Tensor, estimator: int) -> torch.Tensor:
        """Estimate KL divergence using different estimators.

        Args:
            log_ratio: log(pi/pi_ref) = log(pi) - log(pi_ref)
            ratio: pi/pi_old (importance sampling ratio)
            estimator: Which estimator to use (0-3)

        Returns:
            KL divergence estimate
        """
        if estimator == 0:
            return log_ratio
        elif estimator == 1:
            return log_ratio.pow(2) / 2
        elif estimator == 2:
            return torch.expm1(-log_ratio) + log_ratio
        elif estimator == 3:
            return ratio * log_ratio
        else:
            raise ValueError(f"Unknown KL estimator: {estimator}")

    def global_num_flops_in_batch(self, batch: dict[str, Any]) -> int:
        data_BT: data_types.CollatedBatchData = batch["batch"]
        seq_len = data_BT.query_responses[0].shape[1]
        flops_per_token = self.num_flops_per_token(seq_len)
        global_num_tokens = self.trainer.data_loader.global_num_tokens_in_batch(batch)
        assert global_num_tokens is not None
        return flops_per_token * global_num_tokens
