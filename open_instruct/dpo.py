"""
DPO training with OLMo-core's Trainer.

This module provides DPO (Direct Preference Optimization) training using
OLMo-core's native training infrastructure.
"""

import enum
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast

import peft
import torch
import torch.nn as nn
from olmo_core import config, train
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import EvalBatchSpec, TrainModule
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from open_instruct import data_types
from open_instruct.data_loader import HFDataLoader
from open_instruct.dataset_transformation import (
    TOKENIZED_PREFERENCE_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
)
from open_instruct.dpo_utils import (
    DataCollatorForSeq2SeqDPO,
    concatenated_forward,
    dpo_loss,
    separate_forward,
    simpo_loss,
    wpo_loss,
)
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO
from open_instruct.utils import ArgumentParserPlus


class DPOLossType(enum.Enum):
    dpo = "dpo"
    dpo_norm = "dpo_norm"
    simpo = "simpo"
    wpo = "wpo"


@dataclass
class ReferenceLogprobsCache:
    """Cache for reference model log probabilities.

    Stores pre-computed reference model log probabilities for each epoch
    and batch to avoid keeping the reference model in memory during training.
    """

    chosen_logps: list[list[torch.Tensor]]
    rejected_logps: list[list[torch.Tensor]]

    @classmethod
    def build_cache(
        cls,
        model: nn.Module,
        dataloader: HFDataLoader,
        num_epochs: int,
        average_log_prob: bool,
        forward_fn: Callable,
        use_lora: bool = False,
        device: torch.device | None = None,
    ) -> "ReferenceLogprobsCache":
        """Build the reference logprobs cache by iterating through all epochs.

        Args:
            model: The model to use for computing reference logprobs.
            dataloader: The data loader to iterate over.
            num_epochs: Number of training epochs.
            average_log_prob: Whether to average log probabilities.
            forward_fn: The forward function to use.
            use_lora: Whether the model uses LoRA adapters.
            device: The device to use for computation.

        Returns:
            A ReferenceLogprobsCache instance with cached logprobs.
        """
        model.eval()
        epoch_chosen_logps: list[list[torch.Tensor]] = []
        epoch_rejected_logps: list[list[torch.Tensor]] = []

        for epoch in range(num_epochs):
            dataloader.reshuffle(epoch=epoch)
            batch_chosen_logps: list[torch.Tensor] = []
            batch_rejected_logps: list[torch.Tensor] = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"Caching reference logprobs (epoch {epoch})"):
                    if device is not None:
                        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                    if use_lora:
                        assert isinstance(model, peft.PeftModel)
                        with model.disable_adapter():
                            chosen_logps, rejected_logps, _ = forward_fn(
                                model, batch, average_log_prob=average_log_prob
                            )
                    else:
                        chosen_logps, rejected_logps, _ = forward_fn(model, batch, average_log_prob=average_log_prob)

                    batch_chosen_logps.append(chosen_logps.cpu())
                    batch_rejected_logps.append(rejected_logps.cpu())

            epoch_chosen_logps.append(batch_chosen_logps)
            epoch_rejected_logps.append(batch_rejected_logps)

        model.train()
        return cls(chosen_logps=epoch_chosen_logps, rejected_logps=epoch_rejected_logps)

    def get(self, global_step: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached reference logprobs for a given global step.

        Args:
            global_step: The global step number (0-based).
            device: The device to move tensors to.

        Returns:
            Tuple of (chosen_logps, rejected_logps) on the specified device.
        """
        num_epochs = len(self.chosen_logps)
        batches_per_epoch = len(self.chosen_logps[0])
        epoch_idx = (global_step // batches_per_epoch) % num_epochs
        batch_idx = global_step % batches_per_epoch
        return (
            self.chosen_logps[epoch_idx][batch_idx].to(device),
            self.rejected_logps[epoch_idx][batch_idx].to(device),
        )


@dataclass
class DPOConfig(config.Config):
    """Configuration for DPO-specific settings."""

    dpo_beta: float = 0.1
    dpo_loss_type: DPOLossType = DPOLossType.dpo
    dpo_gamma_beta_ratio: float = 0.3
    dpo_label_smoothing: float = 0.0
    load_balancing_loss: bool = False
    load_balancing_weight: float = 1e-3
    concatenated_forward: bool = True
    packing: bool = False


class DPOTrainModule(TrainModule):
    """Training module for DPO with OLMo-core's Trainer."""

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        dpo_config: DPOConfig,
        reference_cache: ReferenceLogprobsCache,
        device: torch.device | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
        self.dpo_config = dpo_config
        self.reference_cache = reference_cache
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.average_log_prob = dpo_config.dpo_loss_type in (DPOLossType.simpo, DPOLossType.dpo_norm)

        if dpo_config.packing:
            self._forward_fn = partial(concatenated_forward, packing=True)
        elif dpo_config.concatenated_forward:
            self._forward_fn = concatenated_forward
        else:
            self._forward_fn = separate_forward

        self._global_step = 0

    def on_attach(self) -> None:
        self._global_step = 0

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optim.step()

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(rank_batch_size=1)

    def eval_batch(self, batch: dict[str, Any], labels: Any | None = None) -> Any:
        self.model.eval()
        with torch.no_grad():
            if self.device is not None:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            return self.model(**batch)

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        self.model.train()

        if self.device is not None:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        policy_chosen_logps, policy_rejected_logps, aux_loss = self._forward_fn(
            self.model,
            batch,
            average_log_prob=self.average_log_prob,
            output_router_logits=self.dpo_config.load_balancing_loss,
        )

        if self.dpo_config.dpo_loss_type in (DPOLossType.dpo, DPOLossType.dpo_norm):
            reference_chosen_logps, reference_rejected_logps = self.reference_cache.get(
                self._global_step, policy_chosen_logps.device
            )
            losses, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.dpo_config.dpo_beta,
                label_smoothing=self.dpo_config.dpo_label_smoothing,
            )
        elif self.dpo_config.dpo_loss_type == DPOLossType.simpo:
            losses, chosen_rewards, rejected_rewards = simpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                beta=self.dpo_config.dpo_beta,
                gamma_beta_ratio=self.dpo_config.dpo_gamma_beta_ratio,
                label_smoothing=self.dpo_config.dpo_label_smoothing,
            )
        elif self.dpo_config.dpo_loss_type == DPOLossType.wpo:
            reference_chosen_logps, reference_rejected_logps = self.reference_cache.get(
                self._global_step, policy_chosen_logps.device
            )
            losses, chosen_rewards, rejected_rewards = wpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.dpo_config.dpo_beta,
                label_smoothing=self.dpo_config.dpo_label_smoothing,
                chosen_loss_mask=batch["chosen_labels"] != -100,
                rejected_loss_mask=batch["rejected_labels"] != -100,
            )
        else:
            raise ValueError(f"Unknown DPO loss type: {self.dpo_config.dpo_loss_type}")

        loss = losses.mean()

        if self.dpo_config.load_balancing_loss and aux_loss is not None:
            loss = loss + self.dpo_config.load_balancing_weight * aux_loss

        if not dry_run:
            self.record_metric("train/loss", loss.detach(), ReduceType.mean)
            self.record_metric("train/logps_chosen", policy_chosen_logps.mean().detach(), ReduceType.mean)
            self.record_metric("train/logps_rejected", policy_rejected_logps.mean().detach(), ReduceType.mean)

            if self.dpo_config.dpo_loss_type in (DPOLossType.dpo, DPOLossType.dpo_norm, DPOLossType.wpo):
                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                margin = (chosen_rewards - rejected_rewards).mean()
                self.record_metric("train/rewards_chosen", chosen_rewards.mean().detach(), ReduceType.mean)
                self.record_metric("train/rewards_rejected", rejected_rewards.mean().detach(), ReduceType.mean)
                self.record_metric("train/rewards_accuracy", accuracy.detach(), ReduceType.mean)
                self.record_metric("train/rewards_margin", margin.detach(), ReduceType.mean)

            if self.dpo_config.load_balancing_loss and aux_loss is not None:
                self.record_metric("train/aux_loss", aux_loss.detach(), ReduceType.mean)

        loss.backward()
        self._global_step += 1 if not dry_run else 0


@dataclass
class DPOExperimentConfig(config.Config):
    """Configuration for a DPO training experiment."""

    exp_name: str = "dpo_experiment"
    run_name: str | None = None
    seed: int = 42

    model_name_or_path: str | None = None
    use_flash_attn: bool = True
    model_revision: str | None = None

    dpo_config: DPOConfig = field(default_factory=DPOConfig)

    num_epochs: int = 2
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048

    dataset_mixer_list: list[str] = field(
        default_factory=lambda: ["allenai/tulu-3-wildchat-reused-on-policy-8b", "1.0"]
    )
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]
    )

    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: float = 16
    lora_dropout: float = 0.1

    output_dir: str = "output/"
    save_folder: str | None = None
    checkpoint_every: int = 500
    keep_last_n_checkpoints: int = 3

    log_every: int = 10
    wandb_project: str = "open_instruct"
    wandb_entity: str | None = None

    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_PREFERENCE_DATASET_KEYS)
    dataset_cache_mode: data_types.DatasetCacheMode = data_types.DatasetCacheMode.local
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_skip_cache: bool = False
    hf_entity: str | None = None


def main(args: DPOExperimentConfig, tc: TokenizerConfig) -> None:
    """Main entry point for DPO training with OLMo-core."""
    train.prepare_training_environment(seed=args.seed)

    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
    dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        target_columns=args.dataset_target_columns,
        dataset_cache_mode=args.dataset_cache_mode.value,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
    )
    dataset = dataset.shuffle(seed=args.seed)
    dataset.set_format(type="pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name_or_path is None:
        raise ValueError("model_name_or_path must be specified")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
    ).to(device)

    if args.dpo_config.packing:
        collator = TensorDataCollatorWithFlatteningDPO(return_position_ids=True, return_flash_attn_kwargs=True)
    else:
        collator = DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=None, padding="longest")

    data_loader_instance = HFDataLoader(
        dataset=dataset,
        batch_size=args.per_device_train_batch_size,
        seed=args.seed,
        rank=0,
        world_size=1,
        work_dir=args.output_dir,
        collator=collator,
    )

    forward_fn = concatenated_forward if args.dpo_config.concatenated_forward else separate_forward
    if args.dpo_config.packing:
        forward_fn = partial(concatenated_forward, packing=True)
    average_log_prob = args.dpo_config.dpo_loss_type in (DPOLossType.simpo, DPOLossType.dpo_norm)

    print("Caching reference logprobs...")
    reference_cache = ReferenceLogprobsCache.build_cache(
        model=model,
        dataloader=data_loader_instance,
        num_epochs=args.num_epochs,
        average_log_prob=average_log_prob,
        forward_fn=forward_fn,
        use_lora=args.use_lora,
        device=device,
    )
    print("Reference logprobs cached.")
    data_loader_instance.reshuffle(epoch=0)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_module = DPOTrainModule(
        model=model,
        optim=optim,
        dpo_config=args.dpo_config,
        reference_cache=reference_cache,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )

    trainer_config = train.TrainerConfig(
        save_folder=args.output_dir,
        max_duration=train.Duration.epochs(args.num_epochs),
        metrics_collect_interval=args.log_every,
    )
    trainer = trainer_config.build(train_module, data_loader_instance)

    print("Starting training...")
    trainer.fit()
    print("Training complete.")

    train.teardown_training_environment()


if __name__ == "__main__":
    parser = ArgumentParserPlus([DPOExperimentConfig, TokenizerConfig])
    args, tc = cast(tuple[DPOExperimentConfig, TokenizerConfig], parser.parse_args_into_dataclasses())
    main(args, tc)
