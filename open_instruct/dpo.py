"""
DPO training with OLMo-core's Trainer.

This module provides DPO (Direct Preference Optimization) training using
OLMo-core's native training infrastructure.
"""

import enum
import hashlib
import json
import logging
import os
import pathlib
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, cast

import peft
import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub import HfApi
from olmo_core import config, train
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh, get_dp_model_mesh
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
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
from tqdm.auto import tqdm

from open_instruct import logger_utils
from open_instruct.beaker_callback import BeakerCallbackV2
from open_instruct.data_loader import HFDataLoader
from open_instruct.dataset_transformation import (
    TOKENIZED_PREFERENCE_DATASET_KEYS,
    TokenizerConfig,
    compute_config_hash,
    get_cached_dataset_tulu,
    load_dataset_configs,
)
from open_instruct.dpo_utils import (
    DataCollatorForSeq2SeqDPO,
    concatenated_forward_olmo,
    dpo_loss,
    separate_forward_olmo,
    simpo_loss,
    wpo_loss,
)
from open_instruct.model_utils import TensorCache, push_folder_to_hub
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO
from open_instruct.utils import (
    GPU_SPECS,
    ArgumentParserPlus,
    get_device_name,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)


class DPOLossType(enum.StrEnum):
    dpo = "dpo"
    dpo_norm = "dpo_norm"
    simpo = "simpo"
    wpo = "wpo"


logger = logging.getLogger(__name__)


def config_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: config_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [config_to_json_serializable(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


OLMO_MODEL_CONFIG_MAP: dict[str, str] = {
    "allenai/OLMo-2-0425-1B": "olmo2_1B",
    "allenai/OLMo-2-1124-7B": "olmo2_7B",
    "allenai/OLMo-2-1124-13B": "olmo2_13B",
    "allenai/OLMo-2-0325-32B": "olmo2_32B",
    "allenai/Olmo-3-1025-7B": "olmo3_7B",
    "allenai/OLMoE-1B-7B-0924": "olmoe_1B_7B",
}


def get_transformer_config(model_name_or_path: str, vocab_size: int) -> TransformerConfig:
    """Get the appropriate TransformerConfig for a given model name.

    Args:
        model_name_or_path: HuggingFace model name or path.
        vocab_size: Vocabulary size for the model.

    Returns:
        TransformerConfig configured for the specified model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    config_name = OLMO_MODEL_CONFIG_MAP.get(model_name_or_path)
    if config_name is None:
        available = ", ".join(OLMO_MODEL_CONFIG_MAP.keys())
        raise ValueError(
            f"Model '{model_name_or_path}' not found in OLMO_MODEL_CONFIG_MAP. "
            f"Available models: {available}. "
            f"Add your model to OLMO_MODEL_CONFIG_MAP in dpo.py."
        )
    config_fn = getattr(TransformerConfig, config_name)
    return config_fn(vocab_size=vocab_size)


def compute_reference_logprobs_cache_hash(
    model_name_or_path: str,
    model_revision: str | None,
    dpo_loss_type: DPOLossType,
    concatenated_forward: bool,
    packing: bool,
    use_lora: bool,
    dataset_config_hash: str,
) -> str:
    """Compute deterministic hash for reference logprobs cache."""
    cache_key = {
        "model_name_or_path": model_name_or_path,
        "model_revision": model_revision,
        "dpo_loss_type": dpo_loss_type.value,
        "concatenated_forward": concatenated_forward,
        "packing": packing,
        "use_lora": use_lora,
        "dataset_config_hash": dataset_config_hash,
    }
    config_str = json.dumps(cache_key, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def build_reference_logprobs_cache(
    model: nn.Module,
    dataloader: HFDataLoader,
    average_log_prob: bool,
    forward_fn: Callable,
    full_dataset_size: int,
    use_lora: bool = False,
    device: torch.device | None = None,
    cache_path: str | pathlib.Path | None = None,
) -> TensorCache:
    """Build a TensorCache with reference logprobs by computing logprobs once for all samples."""
    if cache_path is not None:
        cache_path = pathlib.Path(cache_path)
        if cache_path.exists():
            logger.info(f"Loading reference logprobs cache from {cache_path}")
            return TensorCache.from_disk(cache_path)

    model.eval()
    chosen_tensor = torch.zeros(full_dataset_size, dtype=torch.float32, device=device)
    rejected_tensor = torch.zeros(full_dataset_size, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Caching reference logprobs"):
            if use_lora:
                assert isinstance(model, peft.PeftModel)
                with model.disable_adapter():
                    chosen_logps, rejected_logps, _ = forward_fn(model, batch, average_log_prob=average_log_prob)
            else:
                chosen_logps, rejected_logps, _ = forward_fn(model, batch, average_log_prob=average_log_prob)

            dataset_indices = batch["dataset_index"]
            chosen_tensor[dataset_indices] = chosen_logps
            rejected_tensor[dataset_indices] = rejected_logps

    dist.all_reduce(chosen_tensor, op=dist.ReduceOp.SUM)  # type: ignore[attr-defined]
    dist.all_reduce(rejected_tensor, op=dist.ReduceOp.SUM)  # type: ignore[attr-defined]

    model.train()
    cache = TensorCache(tensors={"chosen_logps": chosen_tensor, "rejected_logps": rejected_tensor})

    if cache_path is not None:
        logger.info(f"Saving reference logprobs cache to {cache_path}")
        cache.to_disk(cache_path)

    return cache


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
    """Training module for DPO with OLMo-core's Trainer.

    Uses OLMo-core's scheduler.set_lr() pattern for learning rate scheduling.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        dpo_config: DPOConfig,
        reference_cache: TensorCache,
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
        self.average_log_prob = dpo_config.dpo_loss_type in (DPOLossType.simpo, DPOLossType.dpo_norm)

        if dpo_config.packing:
            self._forward_fn = partial(concatenated_forward_olmo, packing=True)
        elif dpo_config.concatenated_forward:
            self._forward_fn = concatenated_forward_olmo
        else:
            self._forward_fn = separate_forward_olmo

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

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        return EvalBatchSpec(rank_batch_size=1)

    def eval_batch(self, batch: dict[str, Any], labels: Any | None = None) -> Any:
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

        if self.dpo_config.dpo_loss_type in (DPOLossType.dpo, DPOLossType.dpo_norm):
            ref_logps = self.reference_cache[batch["dataset_index"]]
            losses, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_logps["chosen_logps"],
                ref_logps["rejected_logps"],
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
            ref_logps = self.reference_cache[batch["dataset_index"]]
            losses, chosen_rewards, rejected_rewards = wpo_loss(
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


@dataclass
class DPOExperimentConfig(config.Config):
    """Configuration for a DPO training experiment."""

    exp_name: str = "dpo_experiment"
    run_name: str | None = None
    seed: int = 42
    add_seed_and_date_to_exp_name: bool = True

    model_name_or_path: str | None = None
    use_flash_attn: bool = True
    model_revision: str | None = None

    dpo_beta: float = 0.1
    dpo_loss_type: DPOLossType = DPOLossType.dpo
    dpo_gamma_beta_ratio: float = 0.3
    dpo_label_smoothing: float = 0.0
    load_balancing_loss: bool = False
    load_balancing_weight: float = 1e-3
    concatenated_forward: bool = True
    packing: bool = False

    @property
    def dpo_config(self) -> DPOConfig:
        return DPOConfig(
            dpo_beta=self.dpo_beta,
            dpo_loss_type=self.dpo_loss_type,
            dpo_gamma_beta_ratio=self.dpo_gamma_beta_ratio,
            dpo_label_smoothing=self.dpo_label_smoothing,
            load_balancing_loss=self.load_balancing_loss,
            load_balancing_weight=self.load_balancing_weight,
            concatenated_forward=self.concatenated_forward,
            packing=self.packing,
        )

    num_epochs: int = 2
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_seq_length: int = 2048

    lr_scheduler_type: Literal["linear", "cosine", "constant"] = "linear"
    max_train_steps: int | None = None
    checkpointing_steps: int = 250
    async_checkpointing: bool = False
    clip_grad_norm: float = -1

    use_8bit_optimizer: bool = False
    dpo_use_paged_optimizer: bool = False
    gradient_checkpointing: bool = False
    fused_optimizer: bool = True
    low_cpu_mem_usage: bool = False

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
    resume_from_checkpoint: str | None = None

    log_every: int = 10
    logging_steps: int | None = None
    with_tracking: bool = False
    wandb_project: str = "open_instruct_internal"
    wandb_entity: str | None = None
    report_to: str | list[str] = "all"

    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_PREFERENCE_DATASET_KEYS)
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_skip_cache: bool = False
    cache_dataset_only: bool = False
    dataset_config_hash: str | None = None
    reference_logprobs_cache_path: str = "/weka/oe-adapt-default/allennlp/deletable_reference_logprobs_cache"

    push_to_hub: bool = True
    hf_entity: str | None = None
    hf_repo_id: str | None = None
    hf_repo_revision: str | None = None
    hf_repo_url: str | None = None

    try_launch_beaker_eval_jobs: bool = True
    try_auto_save_to_beaker: bool = True
    oe_eval_tasks: list[str] | None = None
    oe_eval_max_length: int = 4096
    oe_eval_gpu_multiplier: int | None = None
    eval_workspace: str | None = "ai2/tulu-3-results"
    eval_priority: str | None = "high"
    gs_bucket_path: str | None = None


def main(args: DPOExperimentConfig, tc: TokenizerConfig) -> None:
    """Main entry point for DPO training with OLMo-core."""
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer

    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"

    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]

    dcs = load_dataset_configs(
        args.dataset_mixer_list,
        args.dataset_mixer_list_splits,
        args.dataset_transform_fn,
        transform_fn_args,
        args.dataset_target_columns,
    )
    dataset_config_hash = args.dataset_config_hash or compute_config_hash(dcs, tc)

    ref_cache_hash = compute_reference_logprobs_cache_hash(
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        dpo_loss_type=args.dpo_loss_type,
        concatenated_forward=args.concatenated_forward,
        packing=args.packing,
        use_lora=args.use_lora,
        dataset_config_hash=dataset_config_hash,
    )
    reference_cache_path = pathlib.Path(args.reference_logprobs_cache_path) / f"{ref_cache_hash}.pt"
    logger.info(f"Reference logprobs cache path: {reference_cache_path}")

    if args.cache_dataset_only:
        dataset = get_cached_dataset_tulu(
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

    rank = get_rank() if is_distributed() else 0
    is_main_process = rank == 0

    if is_main_process:
        dataset = get_cached_dataset_tulu(
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

    if is_distributed():
        dist.barrier()  # type: ignore[attr-defined]

    if not is_main_process:
        dataset = get_cached_dataset_tulu(
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

    world_size = get_world_size() if is_distributed() else 1

    logger_utils.setup_logger(rank=rank)

    if args.add_seed_and_date_to_exp_name:
        args.exp_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.exp_name)

    if is_distributed():
        path_list = [args.output_dir]
        dist.broadcast_object_list(path_list, src=0)  # type: ignore[attr-defined]
        args.output_dir = path_list[0]

    beaker_config = None
    if is_beaker_job() and is_main_process:
        beaker_config = maybe_get_beaker_config()

    if args.push_to_hub and is_main_process:
        if args.hf_repo_id is None:
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.exp_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name_or_path is None:
        raise ValueError("model_name_or_path must be specified")

    logger.info(f"Building OLMo-core model with vocab_size={tokenizer.vocab_size}")
    model_config = get_transformer_config(args.model_name_or_path, tokenizer.vocab_size)
    model = model_config.build(init_device="cpu")

    logger.info(f"Loading HuggingFace weights from {args.model_name_or_path}")
    load_hf_model(args.model_name_or_path, model.state_dict(), work_dir=args.output_dir)
    model = model.to(device=device, dtype=torch.bfloat16)

    if args.dpo_config.packing:
        collator = TensorDataCollatorWithFlatteningDPO(return_position_ids=True, return_flash_attn_kwargs=True)
    else:
        collator = DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=None, padding="longest")

    global_batch_size = args.per_device_train_batch_size * world_size
    data_loader_instance = HFDataLoader(
        dataset=dataset,
        batch_size=global_batch_size,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        work_dir=args.output_dir,
        collator=collator,
        device=device,
    )

    forward_fn = concatenated_forward_olmo if args.dpo_config.concatenated_forward else separate_forward_olmo
    if args.dpo_config.packing:
        forward_fn = partial(concatenated_forward_olmo, packing=True)
    average_log_prob = args.dpo_config.dpo_loss_type in (DPOLossType.simpo, DPOLossType.dpo_norm)

    logger.info("Caching reference logprobs (before HSDP)...")
    reference_cache = build_reference_logprobs_cache(
        model=model,
        dataloader=data_loader_instance,
        average_log_prob=average_log_prob,
        forward_fn=forward_fn,
        full_dataset_size=len(dataset),
        use_lora=args.use_lora,
        device=device,
        cache_path=reference_cache_path,
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

    json_config = config_to_json_serializable(args.as_dict())
    trainer_callbacks: dict[str, callbacks.Callback] = {"beaker": BeakerCallbackV2(config=json_config)}
    device_name = get_device_name(torch.cuda.get_device_name(0))
    device_peak_flops = int(GPU_SPECS[device_name]["flops"])
    trainer_callbacks["speed_monitor"] = callbacks.SpeedMonitorCallback(
        num_flops_per_token=model.num_flops_per_token(args.max_seq_length), device_peak_flops=device_peak_flops
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
    ).build(train_module, data_loader_instance)

    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training complete.")

    if (
        args.try_auto_save_to_beaker
        and is_main_process
        and is_beaker_job()
        and beaker_config is not None
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if is_beaker_job() and is_main_process and args.try_launch_beaker_eval_jobs:
        wandb_url = None
        if args.with_tracking:
            wandb_tracker = trainer_callbacks.get("wandb")
            if wandb_tracker is not None and hasattr(wandb_tracker, "run") and wandb_tracker.run is not None:
                wandb_url = wandb_tracker.run.get_url()  # type: ignore[union-attr]
        if args.hf_repo_revision is not None:
            eval_path = args.output_dir
            if beaker_config is not None and beaker_config.beaker_dataset_ids:
                eval_path = beaker_config.beaker_dataset_ids[-1]
            launch_ai2_evals_on_weka(
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
        push_folder_to_hub(None, args.output_dir, args.hf_repo_id, args.hf_repo_revision)

    train.teardown_training_environment()


if __name__ == "__main__":
    parser = ArgumentParserPlus([DPOExperimentConfig, TokenizerConfig])
    args, tc = cast(tuple[DPOExperimentConfig, TokenizerConfig], parser.parse_args_into_dataclasses())
    main(args, tc)
