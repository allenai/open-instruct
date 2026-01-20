"""
DPO training with OLMo-core's Trainer.

This module provides DPO (Direct Preference Optimization) training using
OLMo-core's native training infrastructure.
"""

import contextlib
import os
import pathlib
import shutil
from functools import partial

import peft
import torch
import torch.distributed as dist
import transformers
from olmo_core import train
from olmo_core.config import DType
from olmo_core.distributed import utils as distributed_utils
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh, get_dp_model_mesh
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import ConstantWithWarmup, CosWithWarmup, LinearWithWarmup
from olmo_core.train import callbacks
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)

from open_instruct import data_loader as data_loader_lib
from open_instruct import dataset_transformation, dpo_utils, logger_utils, model_utils, utils
from open_instruct.beaker_callback import BeakerCallbackV2
from open_instruct.olmo_core_train_modules import DPOTrainModule
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


def get_transformer_config(model_name_or_path: str, vocab_size: int) -> TransformerConfig:
    """Get the appropriate TransformerConfig for a given model name.

    Args:
        model_name_or_path: HuggingFace model name or path.
        vocab_size: Vocabulary size for the model.

    Returns:
        TransformerConfig for the specified model.

    Raises:
        ValueError: If model not in OLMO_MODEL_CONFIG_MAP.
    """
    config_name = OLMO_MODEL_CONFIG_MAP.get(model_name_or_path)
    if config_name is None:
        available_models = ", ".join(OLMO_MODEL_CONFIG_MAP.keys())
        available_configs = [
            name for name in dir(TransformerConfig) if name.startswith("olmo") and not name.startswith("_")
        ]
        raise ValueError(
            f"Model '{model_name_or_path}' not found in OLMO_MODEL_CONFIG_MAP. "
            f"Available models: {available_models}. "
            f"Available config names: {', '.join(available_configs)}"
        )
    return getattr(TransformerConfig, config_name)(vocab_size=vocab_size)


def _load_dataset_distributed(
    args: dpo_utils.ExperimentConfig,
    tc: dataset_transformation.TokenizerConfig,
    transform_fn_args: list[dict],
    is_main_process: bool,
):
    """Load dataset with distributed coordination."""

    def _load():
        return dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list=args.mixer_list,
            dataset_mixer_list_splits=args.mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.target_columns,
            dataset_cache_mode=args.cache_mode,
            dataset_config_hash=args.config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.local_cache_dir,
            dataset_skip_cache=args.skip_cache,
        )

    if is_main_process:
        dataset = _load()
    if distributed_utils.is_distributed():
        dist.barrier()
    if not is_main_process:
        dataset = _load()
    return dataset


def _setup_model(args: dpo_utils.ExperimentConfig, device: torch.device):
    """Load and configure OLMo-core model."""
    hf_config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
    vocab_size = hf_config.vocab_size
    logger.info(f"Building OLMo-core model with vocab_size={vocab_size}")
    model_config = get_transformer_config(args.model_name_or_path, vocab_size)
    model = model_config.build(init_device="cpu")

    logger.info(f"Loading HuggingFace weights from {args.model_name_or_path}")
    load_hf_model(args.model_name_or_path, model.state_dict(), work_dir=args.output_dir)
    model = model.to(device=device, dtype=torch.bfloat16)

    if args.gradient_checkpointing:
        from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode

        logger.info("Enabling activation checkpointing...")
        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)

    return model


def _apply_hsdp(model, device: torch.device):
    """Apply HSDP to model."""
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp,
        # When None, OLMo-core automatically determines optimal values based on
        # world size. For HSDP, it computes num_replicas and shard_degree to
        # maximize efficiency across the available device mesh.
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
    return model


def _setup_optimizer_and_scheduler(args: dpo_utils.ExperimentConfig, model, num_training_steps: int):
    """Return (optimizer, scheduler)."""
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.dpo_use_paged_optimizer:
        from bitsandbytes.optim import AdamW

        optim = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)

    warmup_steps = int(num_training_steps * args.warmup_ratio)
    if args.lr_scheduler_type == "cosine":
        scheduler = CosWithWarmup(warmup_steps=warmup_steps)
    elif args.lr_scheduler_type == "linear":
        scheduler = LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)
    else:
        scheduler = ConstantWithWarmup(warmup_steps=warmup_steps)

    return optim, scheduler


def _setup_callbacks(args: dpo_utils.ExperimentConfig, model):
    """Return callbacks dict."""
    json_config = dpo_utils.config_to_json_serializable(vars(args))
    trainer_callbacks: dict[str, callbacks.Callback] = {"beaker": BeakerCallbackV2(config=json_config)}
    device_name = utils.get_device_name(torch.cuda.get_device_name(0))
    device_peak_flops = int(utils.GPU_SPECS[device_name]["flops"])
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
    checkpointing_steps = int(args.checkpointing_steps) if args.checkpointing_steps else 500
    trainer_callbacks["checkpointer"] = CheckpointerCallback(save_interval=checkpointing_steps, save_async=False)
    return trainer_callbacks


def _handle_post_training(args: dpo_utils.ExperimentConfig, trainer_callbacks, beaker_config, is_main_process: bool):
    """Save to beaker, launch evals, push to hub."""
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
                wandb_url = wandb_tracker.run.get_url()
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


def main(args: dpo_utils.ExperimentConfig, tc: dataset_transformation.TokenizerConfig) -> None:
    """Main entry point for DPO training with OLMo-core."""
    if args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required. Specify a HuggingFace model name or path.")

    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer

    args.local_cache_dir = os.path.abspath(args.local_cache_dir)
    if utils.is_beaker_job():
        args.local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"

    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
    ref_cache_hash = dpo_utils.compute_reference_cache_hash(args, tc)
    reference_cache_path = pathlib.Path(dpo_utils.REFERENCE_LOGPROBS_CACHE_PATH) / f"{ref_cache_hash}.pt"
    logger.info(f"Reference logprobs cache path: {reference_cache_path}")

    if args.cache_dataset_only:
        dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list=args.mixer_list,
            dataset_mixer_list_splits=args.mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.target_columns,
            dataset_cache_mode=args.cache_mode,
            dataset_config_hash=args.config_hash,
            hf_entity=args.hf_entity,
            dataset_local_cache_dir=args.local_cache_dir,
            dataset_skip_cache=args.skip_cache,
        )
        logger.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    train.prepare_training_environment(seed=args.seed)

    dp_rank = distributed_utils.get_rank() if distributed_utils.is_distributed() else 0
    is_main_process = dp_rank == 0

    dataset = _load_dataset_distributed(args, tc, transform_fn_args, is_main_process)
    dataset = dataset.shuffle(seed=args.seed)
    dataset.set_format(type="pt")  # Must be after shuffle (shuffle resets format)

    dp_world_size = distributed_utils.get_world_size() if distributed_utils.is_distributed() else 1

    logger_utils.setup_logger(rank=dp_rank)

    beaker_config = utils.setup_experiment_paths(args, is_main_process)

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed_utils.is_distributed():
        dist.barrier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _setup_model(args, device)

    if args.packing:
        collator = TensorDataCollatorWithFlatteningDPO(return_position_ids=True, return_flash_attn_kwargs=True)
    else:
        collator = dpo_utils.DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=None, padding="longest")

    global_batch_size = args.per_device_train_batch_size * dp_world_size
    data_loader = data_loader_lib.HFDataLoader(
        dataset=dataset,
        batch_size=global_batch_size,
        seed=args.seed,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        work_dir=args.output_dir,
        collator=collator,
        device=device,
    )

    forward_fn = dpo_utils.concatenated_forward_olmo if args.concatenated_forward else dpo_utils.separate_forward_olmo
    if args.packing:
        forward_fn = partial(dpo_utils.concatenated_forward_olmo, packing=True)
    average_log_prob = args.loss_type.is_average_loss

    logger.info("Caching reference logprobs (before HSDP)...")

    def make_disable_adapter_context() -> contextlib.AbstractContextManager:
        if args.use_lora:
            assert isinstance(model, peft.PeftModel)
            return model.disable_adapter()
        return contextlib.nullcontext()

    reference_cache = dpo_utils.build_reference_logprobs_cache(
        model=model,
        dataloader=data_loader,
        average_log_prob=average_log_prob,
        forward_fn=forward_fn,
        full_dataset_size=len(dataset),
        device=device,
        cache_path=reference_cache_path,
        is_main_process=is_main_process,
        use_lora=args.use_lora,
        disable_adapter_context=make_disable_adapter_context if args.use_lora else None,
    )
    logger.info("Reference logprobs cached.")
    data_loader.reshuffle(epoch=0)

    model = _apply_hsdp(model, device)

    num_training_steps = len(data_loader) * args.num_epochs
    optim, scheduler = _setup_optimizer_and_scheduler(args, model, num_training_steps)

    max_grad_norm = args.max_grad_norm if args.max_grad_norm > 0 else None
    train_module = DPOTrainModule(
        model=model,
        optim=optim,
        args=args,
        reference_cache=reference_cache,
        scheduler=scheduler,
        device=device,
        max_grad_norm=max_grad_norm,
    )

    trainer_callbacks = _setup_callbacks(args, model)

    trainer = train.TrainerConfig(
        save_folder=args.output_dir,
        max_duration=train.Duration.epochs(args.num_epochs),
        metrics_collect_interval=args.logging_steps,
        callbacks=trainer_callbacks,
        save_overwrite=True,
    ).build(train_module, data_loader)

    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training complete.")

    _handle_post_training(args, trainer_callbacks, beaker_config, is_main_process)

    train.teardown_training_environment()


if __name__ == "__main__":
    from open_instruct.utils import ArgumentParserPlus

    parser = ArgumentParserPlus((dpo_utils.ExperimentConfig, dataset_transformation.TokenizerConfig))
    args, tc = parser.parse()
    main(args, tc)
