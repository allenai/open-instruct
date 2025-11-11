import contextlib
import dataclasses
import enum

from olmo_core import train
from olmo_core import data
from olmo_core import config
from olmo_core import nn
from olmo_core.internal import experiment


@contextlib.contextmanager
def prepare_dpo_training_environment():
    train.prepare_training_environment()
    try:
        yield
    finally:
        train.teardown_training_environment()


class DPOLossType(enum.Enum):
    dpo = 1
    dpo_norm = 2
    simpo = 3
    wpo = 4

class DatasetCacheMode(enum.Enum):
    LOCAL = 1
    HF = 2


@dataclasses.dataclass
class DPOExperiment:
    trainer: train.Trainer

    def run():
        with prepare_dpo_training_environment():
            self.trainer.fit()


@dataclasses.dataclass
class DPOTrainerConfig:

@dataclasses.dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "open_instruct"
    entity: str = "ai2-llm"
    group: str = "dpo_experiments"


@dataclasses.dataclass
class DPOExperimentConfig(config.Config):
    # Experiment names
    exp_name: str = "dpo_experiment" # Name of this experiment (e.g. DPO).
    run_name: str | None = None # Unique name of this run.

    #
    checkpoint_every: int | str = 500 # Save a checkpoint every N steps or "epoch" to save at the end of each epoch.
    keep_last_n_checkpoints: int = 3 # Number of last checkpoints to keep. Older checkpoints will be deleted.

    # Model config
    model_name_or_path: str | None
    use_flash_attn: bool = True
    model_revision: str | None = None # Model revision to use (branch, tag, or commit id).
    additional_model_kwargs: dict[str, config.ConfigField] | None = None # Additional kwargs for model initialization.

    # DPO loss config
    dpo_beta: float = 0.1  # DPO beta parameter.
    dpo_loss_type: DPOLossType # Type of DPO loss to use.
    dpo_gamma_beta_ratio: float = 0.3 # Ratio for SIMPO loss. Not used for other loss types.
    dpo_label_smoothing: float = 0.0 # Label smoothing for DPO loss. Default is 0.0 (no smoothing).

    # Optimization config
    clip_grad_norm: float = 1.0 # Gradient clipping norm.
    gradient_accumulation_steps: int = 1 # Number of gradient accumulation steps.
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine" # Learning rate scheduler type.
    log_every: int = 10 # Logging frequency (in steps).
    num_epochs: int = 2
    per_device_train_batch_size: int = 8
    warmup_ratio: float = 0.03 # Warmup ratio for learning rate scheduler.
    weight_decay: float = 0.01 # Weight decay for AdamW.
    gradient_checkpointing: bool = False # Whether to use gradient checkpointing.
    use_liger_kernel: bool = False
    max_train_steps: int | None = None # Maximum number of training steps. If set, overrides num_epochs.
    seed: int = 42
    fused_optimizer: bool = True # Whether to use fused AdamW.
    load_balancing_loss: bool = False # Whether to include a load balancing loss (for OLMoE).
    load_balancing_weight: float = 1e-3 # Weight for the load balancing loss.





    # LoRA config
    use_lora: bool = False
    use_qlora: bool = False
    lora_rank: int = 64
    lora_alpha: float = 16
    lora_dropout: float = 0.1

    # Export config
    output_dir: str = "output/"
    save_to_hub: str | None = None # Repository name to save the model to the Hugging Face Hub. E.g. allenai/your-model.
    push_to_hub: bool = False # Whether to push the model to the Hugging Face Hub.
    hf_entity: str | None = None # Hugging Face Hub entity (user or organization).


    # Checkpoint config
    checkpoint_config: CheckpointConfig

    # Reporting config
    report_to: list[str] = dataclasses.field(
        default_factory=lambda: ["all"]
    ) # Options are tensorboard, wandb, comet_ml, clearml, all.
    wandb_config: WandbConfig


    # Dataset config
    dataset_name: str | None = None
    dataset_mixer: dict[str, float] | None = None
    dataset_mixer_list: list[str] = field(
        default_factory=lambda: ["allenai/tulu-3-wildchat-reused-on-policy-8b", "1.0"]
    )
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]
    )
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_PREFERENCE_DATASET_KEYS)
    dataset_cache_mode: DatasetCacheMode = DatasetCacheMode.LOCAL
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: str | None = None
    dataset_skip_cache: bool = False
    dataset_mix_dir: str | None = None
    dataset_config_name: str | None = None
    max_train_samples: int | None = None
    max_seq_length: int | None = None
    overwrite_cache: bool = False
    timeout_s: int = 1800 # Timeout for the training process. Useful if tokenization is slow.




    model: nn.transformer.TransformerConfig
    dataset: data.NumpyDatasetConfig
    data_loader: data.NumpyDataLoaderConfig
    trainer_config: train.TrainerConfig
    train_module: train.TransformerTrainModuleConfig
    init_seed: int = 123
    load_path: str | None = None
    load_trainer_state: bool = False


@dataclasses.dataclass
class DPOConfig(train.TransformerTrainModuleConfig):

    def build(self, model: Transformer,
              device: torch.device | None = None) -> 'DPOTrainModule':
        return DPOTrainModule(self)


class DPOTrainModule(olmo_core.train.TransformerTrainModule):
    def __init__(self, model, device, dpo_config: DPOConfig):
        super().__init__(model, device)
        self.config = dpo_config


    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        """This is largely copy/pasted from the base class, with modifications for DPO loss."""
        # Set model to train mode if it isn't already.
        self._set_model_mode("train")

        # Generate labels.
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)

        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric(
                "train/masked instances (%)", (~instance_mask).float().mean(), ReduceType.mean
            )

        # Calculate and record how many tokens are going to be used in the loss.
        batch_num_tokens = batch["labels"].numel()
        batch_num_tokens_for_loss = move_to_device(
            (batch["labels"] != self.label_ignore_index).sum(), self.device
        )
        self.record_metric(
            "train/masked labels (%)",
            (batch_num_tokens - batch_num_tokens_for_loss) / batch_num_tokens,
            ReduceType.mean,
        )

        # Batch losses to record.
        policy_chosen_ = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss: Optional[torch.Tensor] = None
        if self.z_loss_multiplier is not None:
            z_batch_loss = move_to_device(torch.tensor(0.0), self.device)

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)
        batch_loss = move_to_device(torch.tensor(0.0), self.device)
        aux_loss = move_to_device(torch.tensor(0.0), self.device)
        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                input_ids, labels, model_kwargs = self._prepare_batch(micro_batch)

                # Run forward pass, get losses.
                # `aux_loss` is only used when `args.load_balancing_loss = True`
                policy_chosen_logps, policy_rejected_logps, aux_loss = forward_fn(
                    model, batch, average_log_prob=average_log_prob,
                    output_router_logits=args.load_balancing_loss)
                losses, _, _ = dpo_utils.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                    loss_kwargs=self.config.loss_kwargs
                )

                # Update total batch CE and Z loss.
                batch_loss += get_local_tensor(losses)
                del losses
                if aux_batch_loss is not None:
                    assert aux_loss is not None
                    aux_loss += get_local_tensor(aux_batch_loss
                    del aux_batch_loss

                # Run backward pass.
                loss.backward()

        del batch  # In case this helps with memory utilization.

        self.model.post_batch(dry_run=dry_run)

        if dry_run:
            self.model.reset_auxiliary_metrics()
            return

        # Record loss metrics.
        if isinstance(self.optim, SkipStepOptimizer):
            # Need to reduce the loss right away for the SkipStepOptimizer.
            if is_distributed():
                ce_batch_loss.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch_loss)
                ce_batch_loss.div_(self.world_size)
                ce_batch_loss.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch_loss)
            self.optim.latest_loss = ce_batch_loss
        else:
            self.record_ce_loss(ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            assert self.z_loss_multiplier is not None
            self.record_metric(
                "Aux batch loss",
                aux_batch_loss,
                ReduceType.mean,
                namespace="train",
            )

        # And additional metrics.
        for metric_name, (metric_val, reduction) in self.model.compute_auxiliary_metrics(
            reset=True
        ).items():
            self.record_metric(
                metric_name,
                metric_val,
                reduction,
                namespace="train",
            )


def build_config(config) -> DPOExperimentConfig:
    config.wandb_config.config = config.as_dict()
    run_name = f"{common.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    trainer_config = train.TrainerConfig(
        save_folder=f"/weka/oe-adapt-default/checkpoints/{common.run_name}",
        save_overwrite=True,
        max_duration=train.Duration.steps(config.steps)
    )
    .with_callback(
        "checkpointer",
        callbacks.CheckpointerCallback(**config.checkpoint_config.as_dict()
                                       ))
    .with_callback(
        "wandb",
        callbacks.WandbCallback(**config.wandb_config),
    )
    model_config = nn.transformer.TransformerConfig.olmo3_7B(
        vocab_size=common.tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_2,
    )
    dataset_config = data.NumpyDatasetConfig(
        dataset_name=config.dataset_name,
        dataset_mixer=config.dataset_mixer,
        target_columns=config.dataset_target_columns,
        transform_fn=config.dataset_transform_fn,
        max_samples=config.max_train_samples,
        max_seq_length=config.max_seq_length,
        cache_mode=config.dataset_cache_mode,
        local_cache_dir=config.dataset_local_cache_dir,
        config_hash=config.dataset_config_hash,
        skip_cache=config.dataset_skip_cache,
        mix_dir=config.dataset_mix_dir,
        config_name=config.dataset_config_name,
        overwrite_cache=config.overwrite_cache,
        timeout_s=config.timeout_s,
    )
    return DPOExperimentConfig(
        exp_name=common.run_name,
        run_name=run_name,
        trainer_config=trainer_config,
        model=model_config,
        dataset=dataset_config,




if __name__ == "__main__":
    config_builder = functools.partial(config_builder = partial(
        build_config,
        global_batch_size=GLOBAL_BATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        trainer_config_builder=build_trainer_config,
        include_default_evals=False,
        include_instance_filter=False,  # We use SkipStepOptimizer for this problem.
    )
    experiment.main(config_builder=config_builder)
