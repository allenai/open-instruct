# !/usr/bin/env python
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SFT script for OLMo models using OLMo-core training infrastructure.

Usage:
    torchrun --nproc_per_node=8 open_instruct/olmo_core_finetune.py \
        --model_name_or_path allenai/OLMo-2-0325-32B-DPO \
        --mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
        --max_seq_length 4096 \
        --learning_rate 8e-5 \
        --num_epochs 3
"""

import dataclasses
import os
from typing import Any, cast

import torch
import torch.distributed as dist
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import ConfigSaverCallback, GarbageCollectorCallback, GPUMemoryMonitorCallback
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerActivationCheckpointingMode,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

from open_instruct import data_loader as data_loader_lib
from open_instruct import logger_utils, olmo_core_utils, utils
from open_instruct.dataset_transformation import TOKENIZED_SFT_DATASET_KEYS, TokenizerConfig, visualize_token
from open_instruct.olmo_core_callbacks import BeakerCallbackV2
from open_instruct.padding_free_collator import TensorDataCollatorWithFlattening

logger = logger_utils.setup_logger(__name__)


_DEFAULT_EPHEMERAL_SAVE_INTERVAL = 500


@dataclasses.dataclass
class SFTArguments:
    tracking: olmo_core_utils.ExperimentConfig
    model: olmo_core_utils.ModelConfig
    training: olmo_core_utils.TrainingConfig
    dataset: olmo_core_utils.DatasetConfig
    logging: olmo_core_utils.LoggingConfig
    checkpoint: olmo_core_utils.CheckpointConfig


def main(args: SFTArguments, tc: TokenizerConfig) -> None:
    assert args.model.model_name_or_path is not None, "model_name_or_path is required"
    tokenizer = olmo_core_utils.setup_tokenizer_and_cache(args.model, args.dataset, tc)

    transform_fn_args = [{"max_seq_length": args.training.max_seq_length}, {}]

    if args.dataset.cache_dataset_only:
        olmo_core_utils.load_dataset_distributed(args.dataset, tc, transform_fn_args, is_main_process=True)
        logger.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    prepare_training_environment(seed=args.tracking.seed)

    global_rank = get_rank() if is_distributed() else 0
    is_main_process = global_rank == 0
    world_size = get_world_size() if is_distributed() else 1
    dataset = olmo_core_utils.load_dataset_distributed(args.dataset, tc, transform_fn_args, is_main_process)
    dataset = dataset.shuffle(seed=args.tracking.seed)
    dataset.set_format(type="pt")

    logger_utils.setup_logger(rank=global_rank)

    if is_main_process:
        visualize_token(dataset[0]["input_ids"], tokenizer)
        os.makedirs(args.checkpoint.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = olmo_core_utils.setup_model(
        args.model.model_name_or_path,
        args.model.config_name,
        cast(AttentionBackendName, args.model.attn_implementation),
    )

    rank_batch_size_seqs = args.training.per_device_train_batch_size * args.training.gradient_accumulation_steps
    global_batch_size_seqs = rank_batch_size_seqs * world_size

    collator = TensorDataCollatorWithFlattening(
        return_position_ids=True,
        return_flash_attn_kwargs=True,
        max_seq_length=rank_batch_size_seqs * args.training.max_seq_length,
    )

    data_loader = data_loader_lib.HFDataLoader(
        dataset=dataset,
        batch_size=global_batch_size_seqs,
        seed=args.tracking.seed,
        dp_rank=global_rank,
        dp_world_size=world_size,
        work_dir=args.checkpoint.output_dir,
        collator=collator,
        device=device,
        drop_last=True,
        fs_local_rank=global_rank,
        max_seq_length=args.training.max_seq_length,
    )
    num_training_steps = len(data_loader) * args.training.num_epochs
    effective_steps = (
        args.training.max_train_steps if args.training.max_train_steps is not None else num_training_steps
    )
    logger.info(
        f"Total training steps: {effective_steps} (data_loader len={len(data_loader)}, epochs={args.training.num_epochs})"
    )

    scheduler = LinearWithWarmup(warmup_steps=int(effective_steps * args.training.warmup_ratio), alpha_f=0.0)

    rank_microbatch_size = rank_batch_size_seqs * args.training.max_seq_length
    dp_shard_degree = min(world_size, torch.cuda.device_count() if torch.cuda.is_available() else 1)

    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp if world_size > dp_shard_degree else DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        shard_degree=dp_shard_degree,
    )

    if args.training.activation_memory_budget < 1.0 and args.training.compile_model:
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.budget,
            activation_memory_budget=args.training.activation_memory_budget,
        )
    else:
        ac_config = None

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=args.training.max_seq_length,
        z_loss_multiplier=None,
        compile_model=args.training.compile_model,
        optim=SkipStepAdamWConfig(
            lr=args.training.learning_rate, weight_decay=args.training.weight_decay, betas=(0.9, 0.95), compile=False
        ),
        dp_config=dp_config,
        ac_config=ac_config,
        scheduler=scheduler,
        max_grad_norm=args.training.max_grad_norm,
    )

    train_module = train_module_config.build(model)

    logger.info("Reloading HuggingFace weights after parallelization...")
    sd = train_module.model.state_dict()
    load_hf_model(args.model.model_name_or_path, sd, work_dir=args.checkpoint.output_dir)
    train_module.model.load_state_dict(sd)

    if args.training.max_train_steps is not None:
        max_duration = Duration.steps(args.training.max_train_steps)
    else:
        max_duration = Duration.epochs(args.training.num_epochs)

    run_name = args.tracking.run_name or f"sft-{os.path.basename(args.model.model_name_or_path)}"
    config_dict = dataclasses.asdict(args)

    trainer_callbacks: dict[str, Any] = {
        "gpu_monitor": GPUMemoryMonitorCallback(),
        "config_saver": ConfigSaverCallback(_config=config_dict),
        "garbage_collector": GarbageCollectorCallback(),
        "checkpointer": olmo_core_utils.build_checkpointer_callback(
            args.checkpoint.checkpointing_steps, args.checkpoint.ephemeral_save_interval
        ),
        "beaker": BeakerCallbackV2(config=config_dict),
    }

    if args.logging.with_tracking and args.logging.wandb_project:
        trainer_callbacks["wandb"] = WandBCallback(
            name=run_name,
            entity=args.logging.wandb_entity or "ai2-llm",
            project=args.logging.wandb_project,
            config=config_dict,
            enabled=True,
            cancel_check_interval=10,
        )

    trainer = TrainerConfig(
        save_folder=args.checkpoint.output_dir,
        max_duration=max_duration,
        metrics_collect_interval=args.logging.logging_steps,
        callbacks=trainer_callbacks,
        save_overwrite=True,
        checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=32, throttle_uploads=True),
    ).build(train_module, data_loader)

    logger.info("Starting training...")
    trainer.fit()
    logger.info("Training complete.")

    teardown_training_environment()


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus(
        (  # ty: ignore[invalid-argument-type]
            olmo_core_utils.ExperimentConfig,
            olmo_core_utils.ModelConfig,
            olmo_core_utils.TrainingConfig,
            olmo_core_utils.DatasetConfig,
            olmo_core_utils.LoggingConfig,
            olmo_core_utils.CheckpointConfig,
            TokenizerConfig,
        )
    )
    parser.set_defaults(
        exp_name="sft",
        ephemeral_save_interval=_DEFAULT_EPHEMERAL_SAVE_INTERVAL,
        num_epochs=3,
        learning_rate=8e-5,
        warmup_ratio=0.03,
        mixer_list=["allenai/tulu-3-sft-olmo-2-mixture", "1.0"],
        transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
        target_columns=list(TOKENIZED_SFT_DATASET_KEYS),
    )
    tracking, model, training, dataset, logging_cfg, checkpoint, tc = parser.parse()  # ty: ignore[invalid-assignment, not-iterable]
    args = SFTArguments(
        tracking=tracking, model=model, training=training, dataset=dataset, logging=logging_cfg, checkpoint=checkpoint
    )
    main(args, tc)
