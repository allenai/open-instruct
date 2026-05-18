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
import glob
import hashlib
import os
import pathlib
import re
from typing import Any

import torch
import torch.distributed as dist
from olmo_core import data as oc_data
from olmo_core import optim
from olmo_core.config import DType
from olmo_core.distributed import parallel
from olmo_core.distributed.utils import is_distributed
from olmo_core.train import Duration, LoadStrategy, TrainerConfig, callbacks, teardown_training_environment
from olmo_core.train import train_module as train_module_lib
from olmo_core.train.checkpoint import CheckpointerConfig

from open_instruct import dataset_transformation, logger_utils, numpy_dataset_conversion, olmo_core_utils, utils

logger = logger_utils.setup_logger(__name__)


_DEFAULT_EPHEMERAL_SAVE_INTERVAL = 500

_TOKENIZE_BARRIER_TIMEOUT_HOURS = 24

_NUMPY_SFT_SUBDIR = "numpy_sft"

_PART_INDEX_RE = re.compile(r"_part_(\d+)\.")


def _chunk_indices(numpy_dir: str, pattern: str) -> set[int]:
    indices = set()
    for path in glob.glob(os.path.join(numpy_dir, pattern)):
        match = _PART_INDEX_RE.search(os.path.basename(path))
        if match is not None:
            indices.add(int(match.group(1)))
    return indices


def _numpy_dir_is_populated(numpy_dir: str) -> bool:
    """Return True only if every chunk has token_ids, labels_mask, and metadata."""
    token_chunks = _chunk_indices(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_NPY_GLOB)
    if not token_chunks:
        return False
    labels_chunks = _chunk_indices(numpy_dir, numpy_dataset_conversion.LABELS_MASK_NPY_GLOB)
    metadata_chunks = _chunk_indices(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_METADATA_GLOB)
    return token_chunks == labels_chunks == metadata_chunks


def _seed_cache_suffix(seed: int, max_seq_length: int) -> str:
    return hashlib.sha256(f"{seed}:{max_seq_length}".encode()).hexdigest()[:8]


def _tokenize_to_numpy_dir(
    numpy_dir: str,
    args: "SFTArguments",
    tc: dataset_transformation.TokenizerConfig,
    transform_fn_args: list[dict],
    visualize: bool,
) -> None:
    logger.info(f"Tokenizing dataset into numpy format at {numpy_dir}")
    numpy_dataset_conversion.convert_hf_to_numpy_sft(
        output_dir=pathlib.Path(numpy_dir),
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
    use_hf_ckpt = olmo_core_utils.is_hf_checkpoint(args.model.model_name_or_path)

    olmo_core_utils.setup_tokenizer_and_cache(args.model, args.dataset, tc)
    transform_fn_args = [{"max_seq_length": args.training.max_seq_length}, {}]

    dcs = dataset_transformation.load_dataset_configs(
        dataset_mixer_list=args.dataset.mixer_list,
        dataset_mixer_list_splits=args.dataset.mixer_list_splits,
        dataset_transform_fn=args.dataset.transform_fn,
        transform_fn_args=transform_fn_args,
        target_columns=list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE),
    )
    cache_hash = dataset_transformation.compute_config_hash(dcs, tc)
    seed_suffix = _seed_cache_suffix(args.tracking.seed, args.training.max_seq_length)
    numpy_dir = os.path.join(args.dataset.local_cache_dir, _NUMPY_SFT_SUBDIR, f"{cache_hash}-{seed_suffix}")

    if args.dataset.cache_dataset_only:
        pre_init_rank = int(os.environ.get("RANK", 0))
        if pre_init_rank == 0:
            if _numpy_dir_is_populated(numpy_dir):
                logger.info(f"Numpy SFT files already present at {numpy_dir}; nothing to do.")
            else:
                _tokenize_to_numpy_dir(numpy_dir, args, tc, transform_fn_args, visualize=True)
            logger.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    if not _numpy_dir_is_populated(numpy_dir):
        mixer = " ".join(args.dataset.mixer_list)
        raise FileNotFoundError(
            "Pre-tokenized numpy SFT dataset not found.\n"
            f"  expected: {numpy_dir}\n"
            f"  hash:     {cache_hash}\n\n"
            "Launch a CPU-only tokenization job on a Weka-mounted cluster first, e.g.:\n\n"
            "  uv run python mason.py \\\n"
            "      --cluster ai2/jupiter --workspace ai2/open-instruct-dev \\\n"
            '      --priority urgent --image "$BEAKER_IMAGE" --budget ai2/oe-adapt \\\n'
            "      --gpus 0 --num_nodes 1 --no_auto_dataset_cache \\\n"
            "      -- uv run python open_instruct/olmo_core_finetune.py \\\n"
            f"      --model_name_or_path {args.model.model_name_or_path} \\\n"
            f"      --tokenizer_name_or_path {tc.tokenizer_name_or_path} \\\n"
            f"      --max_seq_length {args.training.max_seq_length} \\\n"
            f"      --mixer_list {mixer} \\\n"
            f"      --local_cache_dir {args.dataset.local_cache_dir} \\\n"
            "      --cache_dataset_only\n\n"
            "Re-launch training once the tokenization job has completed."
        )

    global_rank, world_size, is_main_process = olmo_core_utils.setup_distributed_env(
        seed=args.tracking.seed, timeout=datetime.timedelta(hours=_TOKENIZE_BARRIER_TIMEOUT_HOURS)
    )

    if is_main_process:
        os.makedirs(args.checkpoint.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config = olmo_core_utils.setup_model(args.model, tc, init_device="meta")

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

    ac_config = olmo_core_utils.build_ac_config(args.training.activation_memory_budget, args.training.compile_model)

    rank_batch_size_seqs = args.training.per_device_train_batch_size * args.training.gradient_accumulation_steps
    rank_microbatch_size = args.training.per_device_train_batch_size * args.training.max_seq_length
    dp_world_size = world_size // cp_degree

    oc_tokenizer_config = olmo_core_utils.to_oc_tokenizer_config(tc)
    np_dataset_config = oc_data.NumpyPackedFSLDatasetConfig(
        tokenizer=oc_tokenizer_config,
        work_dir=args.checkpoint.output_dir,
        paths=[os.path.join(numpy_dir, numpy_dataset_conversion.TOKEN_IDS_NPY_GLOB)],
        expand_glob=True,
        label_mask_paths=[os.path.join(numpy_dir, numpy_dataset_conversion.LABELS_MASK_NPY_GLOB)],
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
    scheduler = olmo_core_utils.build_scheduler(
        args.training.lr_scheduler_type, args.training.warmup_ratio, effective_steps
    )

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
        max_grad_norm=args.training.max_grad_norm
        if args.training.max_grad_norm and args.training.max_grad_norm > 0
        else None,
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

    if use_hf_ckpt:
        olmo_core_utils.reload_hf_checkpoint_after_parallelization(
            train_module, args.model.model_name_or_path, args.checkpoint.output_dir
        )

    if args.training.max_train_steps is not None:
        max_duration = Duration.steps(args.training.max_train_steps)
    else:
        max_duration = Duration.epochs(args.training.num_epochs)

    run_name = args.tracking.run_name or f"sft-{os.path.basename(args.model.model_name_or_path)}"
    config_dict = dataclasses.asdict(args)

    trainer_callbacks: dict[str, Any] = olmo_core_utils.build_base_callbacks(
        config_dict=config_dict,
        run_name=run_name,
        checkpointing_steps=args.checkpoint.checkpointing_steps,
        ephemeral_save_interval=args.checkpoint.ephemeral_save_interval,
        with_tracking=args.logging.with_tracking,
        wandb_project=args.logging.wandb_project,
        wandb_entity=args.logging.wandb_entity or "ai2-llm",
    )
    trainer_callbacks["config_saver"] = callbacks.ConfigSaverCallback(_config=config_dict)
    trainer_callbacks["garbage_collector"] = callbacks.GarbageCollectorCallback()

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
