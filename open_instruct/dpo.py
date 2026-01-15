"""
DPO training with OLMo-core's Trainer.

This module provides DPO (Direct Preference Optimization) training using
OLMo-core's native training infrastructure.
"""

import contextlib
import os
import pathlib
import shutil
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import peft
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from huggingface_hub import HfApi
from olmo_core import config, train
from olmo_core.config import DType
from olmo_core.distributed import utils as distributed_utils
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh, get_dp_model_mesh
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import ConstantWithWarmup, CosWithWarmup, LinearWithWarmup
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train import callbacks
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import EvalBatchSpec, TrainModule
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)

from open_instruct import data_loader, dataset_transformation, dpo_utils, logger_utils, model_utils, utils
from open_instruct import dpo_config as dpo_config_lib
from open_instruct.beaker_callback import BeakerCallbackV2
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO

logger = logger_utils.setup_logger(__name__)


OLMO_MODEL_CONFIG_MAP: dict[str, str] = {
    "allenai/OLMo-2-0425-1B": "olmo2_1B",
    "allenai/OLMo-2-1124-7B": "olmo2_7B",
    "allenai/OLMo-2-1124-13B": "olmo2_13B",
    "allenai/OLMo-2-0325-32B": "olmo2_32B",
    "allenai/Olmo-3-1025-7B": "olmo3_7B",
    "allenai/OLMoE-1B-7B-0924": "olmoe_1B_7B",
    "Qwen/Qwen3-0.6B": "qwen3_0_6B",
    "Qwen/Qwen3-1.7B": "qwen3_1_7B",
    "Qwen/Qwen3-4B": "qwen3_4B",
    "Qwen/Qwen3-8B": "qwen3_8B",
    "Qwen/Qwen3-14B": "qwen3_14B",
    "Qwen/Qwen3-32B": "qwen3_32B",
}


def get_transformer_config(
    model_name_or_path: str, vocab_size: int, config_name_override: str | None = None
) -> TransformerConfig:
    """Get the appropriate TransformerConfig for a given model name.

    Args:
        model_name_or_path: HuggingFace model name or path.
        vocab_size: Vocabulary size for the model.
        config_name_override: Optional override for the config name. If provided, this
            takes precedence over the automatic lookup in OLMO_MODEL_CONFIG_MAP.
            Must be a valid TransformerConfig method name (e.g., 'olmo2_7B').

    Returns:
        TransformerConfig for the specified model.

    Raises:
        ValueError: If model not in OLMO_MODEL_CONFIG_MAP and no override provided.
        AttributeError: If config_name_override is not a valid TransformerConfig method.
    """
    config_name = config_name_override or OLMO_MODEL_CONFIG_MAP.get(model_name_or_path)
    if config_name is None:
        available_models = ", ".join(OLMO_MODEL_CONFIG_MAP.keys())
        available_configs = [
            name for name in dir(TransformerConfig) if name.startswith("olmo") and not name.startswith("_")
        ]
        raise ValueError(
            f"Model '{model_name_or_path}' not found in OLMO_MODEL_CONFIG_MAP. "
            f"Available models: {available_models}. "
            f"You can also use --olmo_config_name to specify a config directly. "
            f"Available config names: {', '.join(available_configs)}"
        )
    config_fn = getattr(TransformerConfig, config_name)
    return config_fn(vocab_size=vocab_size)


@dataclass
class DPOConfig(config.Config):
    """Configuration for DPO-specific settings."""

    dpo_beta: float = 0.1
    dpo_loss_type: dpo_utils.DPOLossType = dpo_utils.DPOLossType.dpo
    dpo_gamma_beta_ratio: float = 0.3
    dpo_label_smoothing: float = 0.0
    load_balancing_loss: bool = False
    load_balancing_weight: float = 1e-3
    concatenated_forward: bool = True
    packing: bool = False


class DPOTrainModule(TrainModule):
    """Training module for DPO with OLMo-core's Trainer.

    Uses OLMo-core's scheduler.set_lr() pattern for learning rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        dpo_config: DPOConfig,
        reference_cache: model_utils.TensorCache,
        scheduler: Scheduler,
        device: torch.device | None = None,
        max_grad_norm: float | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim = optim
        self.dpo_config = dpo_config
        self.reference_cache = reference_cache
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.average_log_prob = dpo_config.dpo_loss_type in (
            dpo_utils.DPOLossType.simpo,
            dpo_utils.DPOLossType.dpo_norm,
        )

        if dpo_config.packing:
            self._forward_fn = partial(dpo_utils.concatenated_forward_olmo, packing=True)
        elif dpo_config.concatenated_forward:
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
        return flops_per_token * global_num_tokens

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(rank_batch_size=1)

    def eval_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(**batch)

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        self.model.train()

        policy_chosen_logps, policy_rejected_logps, aux_loss = self._forward_fn(
            self.model,
            batch,
            average_log_prob=self.average_log_prob,
            output_router_logits=self.dpo_config.load_balancing_loss,
        )

        if self.dpo_config.dpo_loss_type in (dpo_utils.DPOLossType.dpo, dpo_utils.DPOLossType.dpo_norm):
            ref_logps = self.reference_cache[batch["index"]]
            losses, chosen_rewards, rejected_rewards = dpo_utils.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_logps["chosen_logps"],
                ref_logps["rejected_logps"],
                beta=self.dpo_config.dpo_beta,
                label_smoothing=self.dpo_config.dpo_label_smoothing,
            )
        elif self.dpo_config.dpo_loss_type == dpo_utils.DPOLossType.simpo:
            losses, chosen_rewards, rejected_rewards = dpo_utils.simpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                beta=self.dpo_config.dpo_beta,
                gamma_beta_ratio=self.dpo_config.dpo_gamma_beta_ratio,
                label_smoothing=self.dpo_config.dpo_label_smoothing,
            )
        elif self.dpo_config.dpo_loss_type == dpo_utils.DPOLossType.wpo:
            ref_logps = self.reference_cache[batch["index"]]
            losses, chosen_rewards, rejected_rewards = dpo_utils.wpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_logps["chosen_logps"],
                ref_logps["rejected_logps"],
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

            if self.dpo_config.dpo_loss_type in (
                dpo_utils.DPOLossType.dpo,
                dpo_utils.DPOLossType.dpo_norm,
                dpo_utils.DPOLossType.wpo,
            ):
                accuracy = (chosen_rewards > rejected_rewards).float().mean()
                margin = (chosen_rewards - rejected_rewards).mean()
                self.record_metric("train/rewards_chosen", chosen_rewards.mean().detach(), ReduceType.mean)
                self.record_metric("train/rewards_rejected", rejected_rewards.mean().detach(), ReduceType.mean)
                self.record_metric("train/rewards_accuracy", accuracy.detach(), ReduceType.mean)
                self.record_metric("train/rewards_margin", margin.detach(), ReduceType.mean)

            if self.dpo_config.load_balancing_loss and aux_loss is not None:
                self.record_metric("train/aux_loss", aux_loss.detach(), ReduceType.mean)

        loss.backward()


@dataclass
class DPOExperimentConfig(
    dpo_config_lib.ExperimentConfig,
    dpo_config_lib.ModelConfig,
    dpo_config_lib.DPOHyperparamsConfig,
    dpo_config_lib.TrainingConfig,
    dpo_config_lib.DatasetConfig,
    dpo_config_lib.LoRAConfig,
    dpo_config_lib.LoggingConfig,
    dpo_config_lib.HubConfig,
    dpo_config_lib.CheckpointConfig,
    dpo_config_lib.EvalConfig,
    dpo_config_lib.Ai2EvalConfig,
    config.Config,
):
    """Configuration for a DPO training experiment."""

    dpo_loss_type: dpo_utils.DPOLossType = dpo_utils.DPOLossType.dpo
    olmo_config_name: str | None = None
    """Override for OLMo-core TransformerConfig name (e.g., 'olmo2_7B'). If not set, auto-detected from model_name_or_path."""

    @property
    def dpo_config(self) -> DPOConfig:
        return DPOConfig(
            dpo_beta=self.dpo_beta,
            dpo_loss_type=dpo_utils.DPOLossType(self.dpo_loss_type),
            dpo_gamma_beta_ratio=self.dpo_gamma_beta_ratio,
            dpo_label_smoothing=self.dpo_label_smoothing,
            load_balancing_loss=self.load_balancing_loss,
            load_balancing_weight=self.load_balancing_weight,
            concatenated_forward=self.concatenated_forward,
            packing=self.packing,
        )

    checkpointing_steps: int = 250
    async_checkpointing: bool = False

    save_folder: str | None = None
    checkpoint_every: int = 500

    log_every: int = 10

    reference_logprobs_cache_path: str = "/weka/oe-adapt-default/allennlp/deletable_reference_logprobs_cache"


def main(args: DPOExperimentConfig, tc: dataset_transformation.TokenizerConfig) -> None:
    """Main entry point for DPO training with OLMo-core."""
    if args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required. Specify a HuggingFace model name or path.")

    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer

    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if utils.is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"

    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]

    dcs = dataset_transformation.load_dataset_configs(
        args.dataset_mixer_list,
        args.dataset_mixer_list_splits,
        args.dataset_transform_fn,
        transform_fn_args,
        args.dataset_target_columns,
    )
    dataset_config_hash = args.dataset_config_hash or dataset_transformation.compute_config_hash(dcs, tc)

    ref_cache_hash = dpo_utils.compute_reference_logprobs_cache_hash(
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        dpo_loss_type=args.dpo_loss_type,
        concatenated_forward=args.concatenated_forward,
        packing=args.packing,
        use_lora=args.use_lora,
        use_qlora=False,
        max_train_samples=None,
        dataset_config_hash=dataset_config_hash,
    )
    reference_cache_path = pathlib.Path(args.reference_logprobs_cache_path) / f"{ref_cache_hash}.pt"
    logger.info(f"Reference logprobs cache path: {reference_cache_path}")

    if args.cache_dataset_only:
        dataset = dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )
        logger.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    train.prepare_training_environment(seed=args.seed)

    dp_rank = distributed_utils.get_rank() if distributed_utils.is_distributed() else 0
    is_main_process = dp_rank == 0

    if is_main_process:
        dataset = dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )

    if distributed_utils.is_distributed():
        dist.barrier()  # type: ignore[attr-defined]

    if not is_main_process:
        dataset = dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )

    dataset = dataset.shuffle(seed=args.seed)
    dataset.set_format(type="pt")

    dp_world_size = distributed_utils.get_world_size() if distributed_utils.is_distributed() else 1

    logger_utils.setup_logger(rank=dp_rank)

    if args.add_seed_and_date_to_exp_name:
        args.exp_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.exp_name)

    if distributed_utils.is_distributed():
        path_list = [args.output_dir]
        dist.broadcast_object_list(path_list, src=0)  # type: ignore[attr-defined]
        args.output_dir = path_list[0]

    beaker_config = None
    if utils.is_beaker_job() and is_main_process:
        beaker_config = utils.maybe_get_beaker_config()

    if args.push_to_hub and is_main_process:
        if args.hf_repo_id is None:
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:
            args.hf_entity = utils.maybe_use_ai2_hf_entity()
        if args.hf_entity is None:
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.exp_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.wandb_entity is None:
        args.wandb_entity = utils.maybe_use_ai2_wandb_entity()

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed_utils.is_distributed():
        dist.barrier()  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
    vocab_size = hf_config.vocab_size
    logger.info(f"Building OLMo-core model with vocab_size={vocab_size}")
    model_config = get_transformer_config(args.model_name_or_path, vocab_size, args.olmo_config_name)
    model = model_config.build(init_device="cpu")

    logger.info(f"Loading HuggingFace weights from {args.model_name_or_path}")
    load_hf_model(args.model_name_or_path, model.state_dict(), work_dir=args.output_dir)
    model = model.to(device=device, dtype=torch.bfloat16)

    if args.gradient_checkpointing:
        from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode

        logger.info("Enabling activation checkpointing...")
        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)

    if args.dpo_config.packing:
        collator = TensorDataCollatorWithFlatteningDPO(return_position_ids=True, return_flash_attn_kwargs=True)
    else:
        collator = dpo_utils.DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=None, padding="longest")

    global_batch_size = args.per_device_train_batch_size * dp_world_size
    data_loader_instance = data_loader.HFDataLoader(
        dataset=dataset,
        batch_size=global_batch_size,
        seed=args.seed,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        work_dir=args.output_dir,
        collator=collator,
        device=device,
    )

    forward_fn = (
        dpo_utils.concatenated_forward_olmo
        if args.dpo_config.concatenated_forward
        else dpo_utils.separate_forward_olmo
    )
    if args.dpo_config.packing:
        forward_fn = partial(dpo_utils.concatenated_forward_olmo, packing=True)
    average_log_prob = args.dpo_config.dpo_loss_type in (dpo_utils.DPOLossType.simpo, dpo_utils.DPOLossType.dpo_norm)

    logger.info("Caching reference logprobs (before HSDP)...")

    def make_disable_adapter_context() -> contextlib.AbstractContextManager:
        if args.use_lora:
            assert isinstance(model, peft.PeftModel)
            return model.disable_adapter()
        return contextlib.nullcontext()

    def all_reduce_max(tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)

    reference_cache = dpo_utils.build_reference_logprobs_cache(
        model=model,
        dataloader=data_loader_instance,
        average_log_prob=average_log_prob,
        forward_fn=forward_fn,
        full_dataset_size=len(dataset),
        use_lora=args.use_lora,
        device=device,
        cache_path=reference_cache_path,
        disable_adapter_context=make_disable_adapter_context,
        is_distributed=distributed_utils.is_distributed,
        all_reduce_fn=all_reduce_max,
        is_main_process=distributed_utils.get_rank() == 0,
    )
    logger.info("Reference logprobs cached.")
    data_loader_instance.reshuffle(epoch=0)

    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp,
        num_replicas=None,
        shard_degree=None,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
    )
    world_mesh = build_world_mesh(dp=dp_config, device_type=device.type)
    dp_mesh = get_dp_model_mesh(world_mesh)
    logger.info(f"Applying HSDP with dp_mesh: {dp_mesh}")
    model.apply_fsdp(
        dp_mesh=dp_mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        wrapping_strategy=dp_config.wrapping_strategy,
    )

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.dpo_use_paged_optimizer:
        from bitsandbytes.optim import AdamW  # type: ignore[import-unresolved]

        optim = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)

    num_training_steps = len(data_loader_instance) * args.num_epochs
    warmup_steps = int(num_training_steps * args.warmup_ratio)
    if args.lr_scheduler_type == "cosine":
        scheduler = CosWithWarmup(warmup_steps=warmup_steps)
    elif args.lr_scheduler_type == "linear":
        scheduler = LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)
    else:
        scheduler = ConstantWithWarmup(warmup_steps=warmup_steps)

    train_module = DPOTrainModule(
        model=model,
        optim=optim,
        dpo_config=args.dpo_config,
        reference_cache=reference_cache,
        max_grad_norm=args.max_grad_norm,
        device=device,
        scheduler=scheduler,
    )

    json_config = dpo_utils.config_to_json_serializable(args.as_dict())
    trainer_callbacks: dict[str, callbacks.Callback] = {"beaker": BeakerCallbackV2(config=json_config)}
    device_name = utils.get_device_name(torch.cuda.get_device_name(0))
    device_peak_flops = int(utils.GPU_SPECS[device_name]["flops"])
    trainer_callbacks["speed_monitor"] = callbacks.SpeedMonitorCallback(
        num_flops_per_token=model.num_flops_per_token(args.max_seq_length),
        device_peak_flops_per_second=device_peak_flops,
    )
    trainer_callbacks["gpu_memory"] = callbacks.GPUMemoryMonitorCallback()
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        trainer_callbacks["slack"] = callbacks.SlackNotifierCallback(
            name=args.run_name or args.exp_name, webhook_url=slack_webhook_url
        )
    if args.with_tracking:
        trainer_callbacks["wandb"] = callbacks.WandBCallback(
            name=args.run_name or args.exp_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=json_config,
        )
    trainer_callbacks["checkpointer"] = CheckpointerCallback(
        save_interval=args.checkpointing_steps, save_async=args.async_checkpointing
    )

    metrics_collect_interval = args.logging_steps if args.logging_steps is not None else args.log_every
    trainer = train.TrainerConfig(
        save_folder=args.save_folder if args.save_folder else args.output_dir,
        max_duration=train.Duration.epochs(args.num_epochs),
        metrics_collect_interval=metrics_collect_interval,
        callbacks=trainer_callbacks,
        save_overwrite=True,
    ).build(train_module, data_loader_instance)

    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training complete.")

    distributed_utils.barrier()
    output_path = pathlib.Path(args.output_dir).resolve()
    beaker_output_path = pathlib.Path("/output").resolve()
    if (
        args.try_auto_save_to_beaker
        and is_main_process
        and utils.is_beaker_job()
        and beaker_config is not None
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and output_path != beaker_output_path
    ):
        if distributed_utils.is_distributed():
            dist.barrier()
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if utils.is_beaker_job() and is_main_process and args.try_launch_beaker_eval_jobs:
        wandb_url = None
        if args.with_tracking:
            wandb_tracker = trainer_callbacks.get("wandb")
            if wandb_tracker is not None and hasattr(wandb_tracker, "run") and wandb_tracker.run is not None:
                wandb_url = wandb_tracker.run.get_url()  # type: ignore[union-attr]
        if args.hf_repo_revision is not None:
            eval_path = args.output_dir
            if beaker_config is not None and beaker_config.beaker_dataset_ids:
                eval_path = beaker_config.beaker_dataset_ids[-1]
            utils.launch_ai2_evals_on_weka(
                path=eval_path,
                leaderboard_name=args.hf_repo_revision,
                oe_eval_max_length=args.oe_eval_max_length,
                wandb_url=wandb_url,
                oe_eval_tasks=args.oe_eval_tasks,
                gs_bucket_path=args.gs_bucket_path,
                eval_workspace=args.eval_workspace,
                eval_priority=args.eval_priority,
                oe_eval_gpu_multiplier=args.oe_eval_gpu_multiplier,
            )

    if args.push_to_hub and is_main_process:
        model_utils.push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    train.teardown_training_environment()


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus([DPOExperimentConfig, dataset_transformation.TokenizerConfig])
    args, tc = cast(
        tuple[DPOExperimentConfig, dataset_transformation.TokenizerConfig], parser.parse_args_into_dataclasses()
    )
    main(args, tc)
