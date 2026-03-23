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
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
        --max_seq_length 4096 \
        --learning_rate 8e-5 \
        --num_train_epochs 3
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.distributed as dist
import transformers
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.attention.backend import has_flash_attn_3
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig, prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
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
from open_instruct.dataset_transformation import (
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.olmo_core_callbacks import BeakerCallbackV2
from open_instruct.padding_free_collator import TensorDataCollatorWithFlattening

log = logger_utils.setup_logger(__name__)

DEFAULT_SEQUENCE_LENGTH = 4096
GPUS_PER_NODE = 8


@dataclass
class SFTArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from HF."})
    config_name: str | None = field(default=None, metadata={"help": "OLMo-core config name override."})
    run_name: str | None = field(default=None, metadata={"help": "Name for this run (wandb/checkpointing)."})
    output_dir: str = field(default="output/", metadata={"help": "Output directory for checkpoints."})

    dataset_mixer_list: list[str] = field(
        default_factory=lambda: ["allenai/tulu-3-sft-olmo-2-mixture", "1.0"],
        metadata={"help": "Datasets and weights, e.g. dataset1 1.0 dataset2 0.5"},
    )
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: str | None = None
    dataset_skip_cache: bool = False
    hf_entity: str | None = None

    max_seq_length: int = field(default=DEFAULT_SEQUENCE_LENGTH, metadata={"help": "Maximum sequence length."})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Per-device batch size (in sequences)."})
    learning_rate: float = field(default=8e-5, metadata={"help": "Peak learning rate."})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Warmup fraction of total steps."})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    max_train_steps: int | None = field(default=None, metadata={"help": "Override num_epochs with step count."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps."})

    compile_model: bool = field(default=True, metadata={"help": "Whether to torch.compile the model."})
    attn_backend: str = field(default="auto", metadata={"help": "Attention backend: auto, flash_2, flash_3."})

    wandb_project: str | None = field(default=None, metadata={"help": "W&B project name."})
    wandb_entity: str | None = field(default=None, metadata={"help": "W&B entity."})
    with_tracking: bool = field(default=False, metadata={"help": "Enable wandb tracking."})

    save_interval: int = field(default=1000, metadata={"help": "Save checkpoint every N steps."})
    ephemeral_save_interval: int = field(default=500, metadata={"help": "Ephemeral checkpoint interval."})
    logging_steps: int = field(default=10, metadata={"help": "Metrics collection interval."})

    cache_dataset_only: bool = field(default=False, metadata={"help": "Cache dataset and exit."})


def _load_dataset_distributed(
    args: SFTArguments, tc: TokenizerConfig, transform_fn_args: list[dict], is_main_process: bool
):
    def _load():
        return get_cached_dataset_tulu(
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

    if is_main_process:
        dataset = _load()
    if is_distributed():
        dist.barrier()
    if not is_main_process:
        dataset = _load()
    return dataset


def _setup_model(args: SFTArguments, device: torch.device):
    hf_config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
    vocab_size = hf_config.vocab_size
    log.info(f"Building OLMo-core model with vocab_size={vocab_size}")
    config_name_for_lookup = args.config_name if args.config_name else args.model_name_or_path

    attn_backend = args.attn_backend
    if attn_backend == "auto":
        device_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
        is_h100 = "h100" in device_name or "h800" in device_name
        attn_backend = "flash_3" if (is_h100 and has_flash_attn_3()) else "flash_2"
        log.info(f"Auto-detected attn_backend={attn_backend} for device: {device_name}")

    model_config = olmo_core_utils.get_transformer_config(
        config_name_for_lookup, vocab_size, attn_backend=attn_backend
    )
    model = model_config.build(init_device="cpu")

    return model, model_config


def main(args: SFTArguments, tc: TokenizerConfig) -> None:
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer

    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if utils.is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.dataset_local_cache_dir = beaker_cache_dir

    transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]

    if args.cache_dataset_only:
        _load_dataset_distributed(args, tc, transform_fn_args, is_main_process=True)
        log.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    prepare_training_environment(seed=args.seed)

    global_rank = get_rank() if is_distributed() else 0
    is_main_process = global_rank == 0
    world_size = get_world_size() if is_distributed() else 1
    dp_world_size = world_size

    dataset = _load_dataset_distributed(args, tc, transform_fn_args, is_main_process)
    dataset = dataset.shuffle(seed=args.seed)
    dataset.set_format(type="pt")

    logger_utils.setup_logger(rank=global_rank)

    if is_main_process:
        visualize_token(dataset[0]["input_ids"], tokenizer)
        os.makedirs(args.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = _setup_model(args, device)

    collator = TensorDataCollatorWithFlattening(
        return_position_ids=True, return_flash_attn_kwargs=True, max_seq_length=args.max_seq_length
    )

    rank_batch_size_seqs = args.per_device_train_batch_size * args.gradient_accumulation_steps
    global_batch_size_seqs = rank_batch_size_seqs * dp_world_size

    data_loader = data_loader_lib.HFDataLoader(
        dataset=dataset,
        batch_size=global_batch_size_seqs,
        seed=args.seed,
        dp_rank=global_rank,
        dp_world_size=dp_world_size,
        work_dir=args.output_dir,
        collator=collator,
        device=device,
        drop_last=True,
        fs_local_rank=global_rank,
    )
    # The trainer validates global_batch_size against rank_microbatch_size (both in tokens),
    # but HFDataLoader internally uses it as example count. Override for trainer validation.
    data_loader._global_batch_size = global_batch_size_seqs * args.max_seq_length

    data_loader.reshuffle(epoch=0)
    num_training_steps = len(data_loader) * args.num_train_epochs
    effective_steps = args.max_train_steps if args.max_train_steps is not None else num_training_steps
    log.info(
        f"Total training steps: {effective_steps} (data_loader len={len(data_loader)}, epochs={args.num_train_epochs})"
    )

    warmup_steps = int(effective_steps * args.warmup_ratio)
    scheduler = LinearWithWarmup(warmup_steps=warmup_steps, alpha_f=0.0)

    rank_microbatch_size = args.per_device_train_batch_size * args.max_seq_length
    dp_shard_degree = GPUS_PER_NODE if world_size >= GPUS_PER_NODE else world_size

    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.hsdp if world_size > dp_shard_degree else DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        shard_degree=dp_shard_degree,
    )

    ac_config = TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.selected_modules, modules=["blocks.*.feed_forward"]
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=args.max_seq_length,
        z_loss_multiplier=None,
        compile_model=args.compile_model,
        optim=SkipStepAdamWConfig(lr=args.learning_rate, weight_decay=0.0, betas=(0.9, 0.95), compile=False),
        dp_config=dp_config,
        ac_config=ac_config,
        scheduler=scheduler,
        max_grad_norm=1.0,
    )

    train_module = train_module_config.build(model)

    log.info("Reloading HuggingFace weights after parallelization...")
    sd = train_module.model.state_dict()
    load_hf_model(args.model_name_or_path, sd, work_dir=args.output_dir)
    train_module.model.load_state_dict(sd)

    if args.max_train_steps is not None:
        max_duration = Duration.steps(args.max_train_steps)
    else:
        max_duration = Duration.epochs(args.num_train_epochs)

    run_name = args.run_name or f"sft-{os.path.basename(args.model_name_or_path)}"
    json_config: dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "dataset_mixer_list": args.dataset_mixer_list,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "global_batch_size_sequences": global_batch_size_seqs,
        "world_size": world_size,
        "seed": args.seed,
    }

    trainer_callbacks: dict[str, Any] = {
        "gpu_monitor": GPUMemoryMonitorCallback(),
        "config_saver": ConfigSaverCallback(_config=json_config),
        "garbage_collector": GarbageCollectorCallback(),
        "checkpointer": CheckpointerCallback(
            save_interval=args.save_interval, ephemeral_save_interval=args.ephemeral_save_interval, save_async=True
        ),
        "beaker": BeakerCallbackV2(config=json_config),
    }

    if args.with_tracking and args.wandb_project:
        trainer_callbacks["wandb"] = WandBCallback(
            name=run_name,
            entity=args.wandb_entity or "ai2-llm",
            project=args.wandb_project,
            config=json_config,
            enabled=True,
            cancel_check_interval=10,
        )

    trainer = TrainerConfig(
        save_folder=args.output_dir,
        max_duration=max_duration,
        metrics_collect_interval=args.logging_steps,
        callbacks=trainer_callbacks,
        save_overwrite=True,
        checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=32, throttle_uploads=True),
    ).build(train_module, data_loader)

    log.info("Starting training...")
    trainer.fit()
    log.info("Training complete.")

    teardown_training_environment()


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus((SFTArguments, TokenizerConfig))
    args, tc = parser.parse()
    main(args, tc)
