"""
DPO training using OLMo-core's Trainer.

This module provides DPO (Direct Preference Optimization) training using
OLMo-core's native training infrastructure.
"""

import os
import pathlib
import shutil
from functools import partial

import bitsandbytes.optim
import torch
import torch.distributed as dist
import transformers
from olmo_core import train
from olmo_core.config import DType
from olmo_core.distributed import utils as distributed_utils
from olmo_core.distributed.parallel import DataParallelType, build_world_mesh, get_dp_model_mesh
from olmo_core.nn.attention.backend import has_flash_attn_3
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.optim import ConstantWithWarmup, CosWithWarmup, LinearWithWarmup
from olmo_core.train import callbacks
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
)

from open_instruct import data_loader as data_loader_lib
from open_instruct import dataset_transformation, dpo_utils, logger_utils, model_utils, olmo_core_utils, utils
from open_instruct.beaker_callback import BeakerCallbackV2
from open_instruct.olmo_core_train_modules import DPOTrainModule
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO

logger = logger_utils.setup_logger(__name__)


def export_to_hf(
    model, model_config, tokenizer, save_dir: str, original_model_name_or_path: str, is_main_process: bool
):
    """Export an FSDP-wrapped model to HuggingFace format.

    All ranks must call this function as state_dict() and full_tensor() are collective operations.
    Only the main process saves to disk.
    """
    logger.info("Gathering FSDP state dict...")
    state_dict = model.state_dict()
    state_dict = {k: v.full_tensor().cpu() if hasattr(v, "full_tensor") else v.cpu() for k, v in state_dict.items()}

    if is_main_process:
        logger.info(f"Exporting model to HuggingFace format at {save_dir}")
        olmo_core_utils.save_state_dict_as_hf(
            model_config, state_dict, save_dir, original_model_name_or_path, tokenizer
        )


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
    config_name_for_lookup = args.config_name if args.config_name else args.model_name_or_path

    attn_backend = args.attn_backend
    if attn_backend == "auto":
        device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
        is_h100 = "h100" in device_name or "h800" in device_name
        attn_backend = "flash_3" if (is_h100 and has_flash_attn_3()) else "flash_2"
        logger.info(f"Auto-detected attn_backend={attn_backend} for device: {device_name}")

    model_config = olmo_core_utils.get_transformer_config(
        config_name_for_lookup, vocab_size, attn_backend=attn_backend
    )
    model = model_config.build(init_device="cpu")

    logger.info(f"Loading HuggingFace weights from {args.model_name_or_path}")
    load_hf_model(args.model_name_or_path, model.state_dict(), work_dir=args.output_dir)
    model = model.to(device=device, dtype=torch.bfloat16)

    if args.gradient_checkpointing:
        logger.info("Enabling activation checkpointing...")
        model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)

    return model, model_config


def _apply_parallelism(
    model,
    device: torch.device,
    tensor_parallel_degree: int = 1,
    context_parallel_degree: int = 1,
    pipeline_parallel_degree: int = 1,
):
    """Apply parallelism strategies to model (HSDP, TP, CP, PP).

    Args:
        model: The model to apply parallelism to.
        device: The device to use.
        tensor_parallel_degree: Tensor parallelism degree (default 1, disabled).
        context_parallel_degree: Context parallelism degree (default 1, disabled).
        pipeline_parallel_degree: Pipeline parallelism degree (default 1, disabled).

    Returns:
        The model with parallelism applied.
    """
    if tensor_parallel_degree > 1 and context_parallel_degree > 1:
        raise ValueError("Cannot use both tensor parallelism and context parallelism simultaneously.")

    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp,
        num_replicas=None,
        shard_degree=None,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        wrapping_strategy=TransformerDataParallelWrappingStrategy.blocks,
    )

    tp_config = tensor_parallel_degree if tensor_parallel_degree > 1 else None
    cp_config = context_parallel_degree if context_parallel_degree > 1 else None
    pp_config = pipeline_parallel_degree if pipeline_parallel_degree > 1 else None

    world_mesh = build_world_mesh(dp=dp_config, tp=tp_config, cp=cp_config, pp=pp_config, device_type=device.type)
    dp_mesh = get_dp_model_mesh(world_mesh)

    if tensor_parallel_degree > 1:
        logger.info(f"Applying tensor parallelism with degree={tensor_parallel_degree}")
        tp_mesh = world_mesh["tp"]
        model.apply_tp(tp_mesh)

    if context_parallel_degree > 1:
        logger.info(f"Applying context parallelism with degree={context_parallel_degree}")

    if pipeline_parallel_degree > 1:
        logger.info(f"Applying pipeline parallelism with degree={pipeline_parallel_degree}")
        pp_mesh = world_mesh["pp"]
        model.apply_pp(pp_mesh)

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
        optim = bitsandbytes.optim.AdamW(
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
    checkpointing_steps = int(args.checkpointing_steps)
    trainer_callbacks["checkpointer"] = CheckpointerCallback(save_interval=checkpointing_steps, save_async=False)
    return trainer_callbacks


def _handle_post_training(
    args: dpo_utils.ExperimentConfig,
    model,
    model_config,
    tokenizer,
    trainer_callbacks,
    beaker_config,
    is_main_process: bool,
):
    """Save HF model, copy to beaker, launch evals, push to hub."""
    hf_model_path = os.path.join(args.output_dir, "hf_model")
    export_to_hf(model, model_config, tokenizer, hf_model_path, args.model_name_or_path, is_main_process)

    if distributed_utils.is_distributed():
        dist.barrier()

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
        shutil.copytree(hf_model_path, "/output", dirs_exist_ok=True)

    if utils.is_beaker_job() and is_main_process and args.try_launch_beaker_eval_jobs:
        wandb_url = None
        if args.with_tracking:
            wandb_tracker = trainer_callbacks.get("wandb")
            if wandb_tracker is not None and hasattr(wandb_tracker, "run") and wandb_tracker.run is not None:
                wandb_url = wandb_tracker.run.get_url()
        if args.hf_repo_revision is not None:
            eval_path = hf_model_path
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
        model_utils.push_folder_to_hub(hf_model_path, args.hf_repo_id, args.hf_repo_revision)


def main(args: dpo_utils.ExperimentConfig, tc: dataset_transformation.TokenizerConfig) -> None:
    """Main entry point for DPO training with OLMo-core."""
    if args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required. Specify a HuggingFace model name or path.")

    if args.use_lora:
        raise ValueError("LoRA is not supported with OLMo-core DPO training. Use dpo_tune_cache.py instead.")

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

    world_size = distributed_utils.get_world_size() if distributed_utils.is_distributed() else 1
    parallelism_factor = args.tensor_parallel_degree * args.context_parallel_degree * args.pipeline_parallel_degree
    dp_world_size = world_size // parallelism_factor

    logger_utils.setup_logger(rank=dp_rank)

    beaker_config = utils.setup_experiment_paths(args, is_main_process)

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if distributed_utils.is_distributed():
        dist.barrier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = _setup_model(args, device)

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
    # 4x batch size: forward-only (no backward), so no activation storage needed.
    cache_batch_size = args.per_device_train_batch_size * 4 * dp_world_size
    cache_data_loader = data_loader_lib.HFDataLoader(
        dataset=dataset,
        batch_size=cache_batch_size,
        seed=args.seed,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        work_dir=args.output_dir,
        collator=collator,
        device=device,
        drop_last=False,
    )

    forward_fn = dpo_utils.concatenated_forward_olmo if args.concatenated_forward else dpo_utils.separate_forward_olmo
    if args.packing:
        forward_fn = partial(dpo_utils.concatenated_forward_olmo, packing=True)
    average_log_prob = args.loss_type.is_average_loss

    cache_kwargs = dict(
        dataloader=cache_data_loader,
        average_log_prob=average_log_prob,
        forward_fn=forward_fn,
        full_dataset_size=len(dataset),
        device=device,
        cache_path=reference_cache_path,
        is_main_process=is_main_process,
        model_dims=utils.ModelDims.from_hf_config(args.model_name_or_path),
        use_lora=False,
        disable_adapter_context=None,
    )

    model_is_sharded = False
    logger.info("Caching reference logprobs (trying unsharded first)...")
    try:
        reference_cache = dpo_utils.build_reference_logprobs_cache(model=model, **cache_kwargs)
        logger.info("Reference logprobs cached (unsharded).")
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM with unsharded model, falling back to FSDP-sharded.")
        torch.cuda.empty_cache()
        model_is_sharded = True
        model = _apply_parallelism(
            model, device, args.tensor_parallel_degree, args.context_parallel_degree, args.pipeline_parallel_degree
        )
        reference_cache = dpo_utils.build_reference_logprobs_cache(model=model, **cache_kwargs)
        logger.info("Reference logprobs cached (sharded).")

    if args.cache_logprobs_only:
        logger.info("--cache_logprobs_only set, exiting after cache build.")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    if not model_is_sharded:
        model = _apply_parallelism(
            model, device, args.tensor_parallel_degree, args.context_parallel_degree, args.pipeline_parallel_degree
        )
    data_loader.reshuffle(epoch=0)

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

    _handle_post_training(args, model, model_config, tokenizer, trainer_callbacks, beaker_config, is_main_process)

    train.teardown_training_environment()


if __name__ == "__main__":
    from open_instruct.utils import ArgumentParserPlus

    parser = ArgumentParserPlus((dpo_utils.ExperimentConfig, dataset_transformation.TokenizerConfig))
    args, tc = parser.parse()
    main(args, tc)
