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
import datetime
import hashlib
import json
import os
from typing import Any

import torch
import torch.distributed as dist
from olmo_core import data as oc_data
from olmo_core import optim
from olmo_core.config import DType
from olmo_core.distributed import parallel
from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    callbacks,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train import train_module as train_module_lib
from olmo_core.train.callbacks.wandb import WandBCallback
from olmo_core.train.checkpoint import CheckpointerConfig

from open_instruct import (
    dataset_transformation,
    logger_utils,
    numpy_dataset_conversion,
    olmo_core_callbacks,
    olmo_core_utils,
    utils,
)

logger = logger_utils.setup_logger(__name__)


_DEFAULT_EPHEMERAL_SAVE_INTERVAL = 500

_TOKENIZE_BARRIER_TIMEOUT_HOURS = 24


def _numpy_cache_dir_hash(
    mixer_list: list[str],
    mixer_list_splits: list[str],
    max_seq_length: int,
    tokenizer_name: str | None,
    chat_template_name: str | None,
    transform_fn: list[str],
) -> str:
    payload = json.dumps(
        {
            "mixer_list": mixer_list,
            "mixer_list_splits": mixer_list_splits,
            "max_seq_length": max_seq_length,
            "tokenizer": tokenizer_name,
            "chat_template": chat_template_name,
            "transform_fn": transform_fn,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def _resolve_numpy_dir(args: "SFTArguments", tc: dataset_transformation.TokenizerConfig) -> str:
    if args.dataset.dataset_path is not None:
        return args.dataset.dataset_path
    cache_hash = _numpy_cache_dir_hash(
        mixer_list=args.dataset.mixer_list,
        mixer_list_splits=args.dataset.mixer_list_splits,
        max_seq_length=args.training.max_seq_length,
        tokenizer_name=tc.tokenizer_name_or_path,
        chat_template_name=tc.chat_template_name,
        transform_fn=args.dataset.transform_fn,
    )
    return os.path.join(args.dataset.local_cache_dir, "numpy_sft", cache_hash)


def _numpy_dir_is_populated(numpy_dir: str) -> bool:
    return os.path.exists(os.path.join(numpy_dir, "token_ids_part_0000.npy"))


def _tokenize_to_numpy_dir(
    numpy_dir: str,
    args: "SFTArguments",
    tc: dataset_transformation.TokenizerConfig,
    transform_fn_args: list[dict],
    visualize: bool,
) -> None:
    logger.info(f"Tokenizing dataset into numpy format at {numpy_dir}")
    numpy_dataset_conversion.convert_hf_to_numpy_sft(
        output_dir=numpy_dir,
        dataset_mixer_list=args.dataset.mixer_list,
        dataset_mixer_list_splits=args.dataset.mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset.transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_target_columns=list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE),
        max_seq_length=args.training.max_seq_length,
        dataset_cache_mode=args.dataset.cache_mode,
        dataset_local_cache_dir=args.dataset.local_cache_dir,
        dataset_skip_cache=args.dataset.skip_cache,
        dataset_config_hash=args.dataset.config_hash,
        shuffle_seed=args.tracking.seed,
        resume=True,
        visualize=visualize,
    )


@dataclasses.dataclass
class SFTArguments:
    tracking: olmo_core_utils.ExperimentConfig
    model: olmo_core_utils.ModelConfig
    training: olmo_core_utils.TrainingConfig
    dataset: olmo_core_utils.DatasetConfig
    logging: olmo_core_utils.LoggingConfig
    checkpoint: olmo_core_utils.CheckpointConfig


def main(args: SFTArguments, tc: dataset_transformation.TokenizerConfig) -> None:
    assert args.model.model_name_or_path is not None, "model_name_or_path is required"

    use_hf_ckpt = olmo_core_utils.is_hf_checkpoint(args.model.model_name_or_path)

    olmo_core_utils.setup_tokenizer_and_cache(args.model, args.dataset, tc)
    transform_fn_args = [{"max_seq_length": args.training.max_seq_length}, {}]

    numpy_dir = _resolve_numpy_dir(args, tc)

    if args.dataset.cache_dataset_only:
        pre_init_rank = int(os.environ.get("RANK", 0))
        if pre_init_rank == 0:
            if _numpy_dir_is_populated(numpy_dir):
                logger.info(f"Numpy SFT files already present at {numpy_dir}; nothing to do.")
            else:
                _tokenize_to_numpy_dir(numpy_dir, args, tc, transform_fn_args, visualize=True)
            logger.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    prepare_training_environment(
        seed=args.tracking.seed, timeout=datetime.timedelta(hours=_TOKENIZE_BARRIER_TIMEOUT_HOURS)
    )

    global_rank = get_rank() if is_distributed() else 0
    is_main_process = global_rank == 0
    world_size = get_world_size() if is_distributed() else 1

    logger_utils.setup_logger(rank=global_rank)

    if is_main_process:
        os.makedirs(args.checkpoint.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_device = "meta" if not use_hf_ckpt else "cpu"
    model, model_config = olmo_core_utils.setup_model(args.model, init_device=init_device)

    cp_config = olmo_core_utils.build_cp_config(args.training)
    cp_degree = args.training.cp_degree or 1
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    dp_shard_degree = gpus_per_node // cp_degree

    dp_config = train_module_lib.TransformerDataParallelConfig(
        name=parallel.DataParallelType.hsdp if world_size > dp_shard_degree else parallel.DataParallelType.fsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        shard_degree=dp_shard_degree,
    )

    ac_config = olmo_core_utils.build_ac_config(args.training)

    rank_batch_size_seqs = args.training.per_device_train_batch_size * args.training.gradient_accumulation_steps
    rank_microbatch_size = args.training.per_device_train_batch_size * args.training.max_seq_length
    dp_world_size = world_size // cp_degree

    if is_main_process and not _numpy_dir_is_populated(numpy_dir):
        _tokenize_to_numpy_dir(numpy_dir, args, tc, transform_fn_args, visualize=True)
    if is_distributed():
        dist.barrier()
    if not _numpy_dir_is_populated(numpy_dir):
        raise RuntimeError(f"Expected tokenized numpy files at {numpy_dir} after tokenization barrier.")

    oc_tokenizer_config = olmo_core_utils.to_oc_tokenizer_config(tc)
    np_dataset_config = oc_data.NumpyPackedFSLDatasetConfig(
        tokenizer=oc_tokenizer_config,
        work_dir=args.checkpoint.output_dir,
        paths=[os.path.join(numpy_dir, "token_ids_part_*.npy")],
        expand_glob=True,
        label_mask_paths=[os.path.join(numpy_dir, "labels_mask_*.npy")],
        generate_doc_lengths=True,
        long_doc_strategy=oc_data.LongDocStrategy.truncate,
        sequence_length=args.training.max_seq_length,
    )
    np_dataset = np_dataset_config.build()
    np_dataset.prepare()

    global_batch_size_seqs = rank_batch_size_seqs * dp_world_size

    num_training_steps = len(np_dataset) // global_batch_size_seqs * args.training.num_epochs
    effective_steps = (
        args.training.max_train_steps if args.training.max_train_steps is not None else num_training_steps
    )
    logger.info(f"Total training steps: {effective_steps} (epochs={args.training.num_epochs})")
    scheduler = optim.LinearWithWarmup(warmup_steps=int(effective_steps * args.training.warmup_ratio), alpha_f=0.0)

    train_module_config = train_module_lib.TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_size,
        max_sequence_length=args.training.max_seq_length,
        z_loss_multiplier=None,
        compile_model=args.training.compile_model,
        optim=optim.SkipStepAdamWConfig(
            lr=args.training.learning_rate, weight_decay=args.training.weight_decay, betas=(0.9, 0.95), compile=False
        ),
        dp_config=dp_config,
        cp_config=cp_config,
        ac_config=ac_config,
        scheduler=scheduler,
        max_grad_norm=args.training.max_grad_norm if args.training.max_grad_norm > 0 else None,
    )

    train_module = train_module_config.build(model)

    data_loader_seed = (
        args.tracking.data_loader_seed if args.tracking.data_loader_seed is not None else args.tracking.seed
    )
    global_batch_size_tokens = global_batch_size_seqs * args.training.max_seq_length
    data_loader_config = oc_data.NumpyDataLoaderConfig(
        global_batch_size=global_batch_size_tokens,
        seed=data_loader_seed,
        work_dir=args.checkpoint.output_dir,
        num_workers=4,
        target_device_type=device.type,
    )
    data_loader = data_loader_config.build(
        np_dataset,
        collator=oc_data.DataCollator(
            pad_token_id=oc_tokenizer_config.pad_token_id, vocab_size=oc_tokenizer_config.padded_vocab_size()
        ),
        dp_process_group=train_module.dp_process_group,
    )
    data_loader.reshuffle(epoch=1)

    first_batch = next(iter(data_loader))
    if is_main_process:
        input_ids = first_batch.get("input_ids")
        if input_ids is not None:
            logger.info(f"DEBUG first batch input_ids shape: {input_ids.shape}")
            logger.info(f"DEBUG first batch input_ids[:100]: {input_ids.flatten()[:100].tolist()}")
            logger.info(f"DEBUG first batch input_ids[-100:]: {input_ids.flatten()[-100:].tolist()}")
        label_mask = first_batch.get("label_mask")
        if label_mask is not None:
            logger.info(f"DEBUG first batch label_mask trainable count: {int(label_mask.sum().item())}")
            logger.info(f"DEBUG first batch label_mask[:100]: {label_mask.flatten()[:100].tolist()}")
    data_loader.reshuffle(epoch=1)

    if use_hf_ckpt:
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
        "gpu_monitor": callbacks.GPUMemoryMonitorCallback(),
        "config_saver": callbacks.ConfigSaverCallback(_config=config_dict),
        "garbage_collector": callbacks.GarbageCollectorCallback(),
        "checkpointer": olmo_core_utils.build_checkpointer_callback(
            args.checkpoint.checkpointing_steps, args.checkpoint.ephemeral_save_interval
        ),
        "beaker": olmo_core_callbacks.BeakerCallbackV2(config=config_dict),
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

    load_strategy = LoadStrategy.never if not use_hf_ckpt else LoadStrategy.if_available

    trainer = TrainerConfig(
        save_folder=args.checkpoint.output_dir,
        load_strategy=load_strategy,
        max_duration=max_duration,
        metrics_collect_interval=args.logging.logging_steps,
        callbacks=trainer_callbacks,
        save_overwrite=True,
        checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=32, throttle_uploads=True),
    ).build(train_module, data_loader)

    if not use_hf_ckpt:
        logger.info(f"Loading olmo-core checkpoint from {args.model.model_name_or_path}...")
        trainer.load_checkpoint(args.model.model_name_or_path, load_trainer_state=False)

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
            dataset_transformation.TokenizerConfig,
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
        target_columns=list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS),
    )
    tracking, model, training, dataset, logging_cfg, checkpoint, tc = parser.parse()  # ty: ignore[invalid-assignment, not-iterable]
    args = SFTArguments(
        tracking=tracking, model=model, training=training, dataset=dataset, logging=logging_cfg, checkpoint=checkpoint
    )
    main(args, tc)
