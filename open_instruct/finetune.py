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

# TODO: Add LoRA support via peft integration
# TODO: Add QLoRA support (4-bit quantization)
# TODO: Add LigerKernel support for fused operations
# TODO: Add 8-bit optimizer support
# TODO: Add load balancing loss for OLMoE models

"""
SFT script for OLMo models using OLMo-core training infrastructure.

This script supports three modes:
- `cache_dataset_only`: Convert HuggingFace datasets to numpy format for training
- `launch`: Configure and launch training on Beaker
- `train`: Execute training
- `dry_run`: Validate configuration without training

Usage:
    # Step 1: Prepare dataset
    python open_instruct/finetune.py cache_dataset_only \\
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \\
        --output_dir /path/to/dataset \\
        --max_seq_length 4096

    # Step 2: Launch training
    python open_instruct/finetune.py launch run_name /path/to/checkpoint ai2/cluster \\
        --dataset_path /path/to/dataset \\
        --seq_len 4096 \\
        --num_nodes 2
"""

import argparse
import gzip
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

import numpy as np
from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyPackedFSLDatasetConfig, TokenizerConfig
from olmo_core.data.types import LongDocStrategy
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_local_rank, get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import (
    CLUSTER_TO_GPU_TYPE,
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.io import copy_dir, dir_is_empty, get_parent, join_path
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
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
from olmo_core.utils import prepare_cli_environment, seed_all
from rich import print
from tqdm import tqdm

from open_instruct.dataset_transformation import (
    ATTENTION_MASK_KEY,
    DATASET_ORIGIN_KEY,
    INPUT_IDS_KEY,
    LABELS_KEY,
    TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE,
    get_cached_dataset_tulu_with_statistics,
    remove_dataset_source_field,
    visualize_token,
)
from open_instruct.dataset_transformation import TokenizerConfig as OITokenizerConfig
from open_instruct.utils import is_beaker_job

log = logging.getLogger(__name__)

DEFAULT_SEQUENCE_LENGTH = 4096
DEFAULT_NUM_NODES = 1
GPUS_PER_NODE = 8
MAX_RANK_MICROBATCH_SIZE_TOKENS = 16_384


@dataclass
class BatchSizeConfig:
    """Automatically calculates microbatch size and gradient accumulation steps."""

    global_batch_size_tokens: int
    sequence_length: int
    world_size: int
    gpu_type: str
    rank_microbatch_size_tokens: int = field(init=False)
    rank_microbatch_size_sequences: int = field(init=False)
    grad_accum_steps: int = field(init=False)

    def __post_init__(self):
        assert self.global_batch_size_tokens > 0
        assert self.sequence_length > 0
        assert self.world_size > 0

        max_tokens_per_rank = MAX_RANK_MICROBATCH_SIZE_TOKENS
        if "B200" in self.gpu_type:
            max_tokens_per_rank *= 2

        dp_world_size = self.world_size
        rank_batch_size_tokens = self.global_batch_size_tokens // dp_world_size

        if rank_batch_size_tokens > max_tokens_per_rank:
            self.grad_accum_steps = 1
            while rank_batch_size_tokens // self.grad_accum_steps > max_tokens_per_rank:
                self.grad_accum_steps *= 2

            self.rank_microbatch_size_tokens = rank_batch_size_tokens // self.grad_accum_steps
            log.info(
                f"Rank batch size ({rank_batch_size_tokens} tokens) exceeds "
                f"max tokens per rank ({max_tokens_per_rank} tokens). "
                f"Using grad_accum_steps={self.grad_accum_steps}"
            )
        else:
            self.rank_microbatch_size_tokens = rank_batch_size_tokens
            self.grad_accum_steps = 1

        assert self.rank_microbatch_size_tokens % self.sequence_length == 0
        self.rank_microbatch_size_sequences = self.rank_microbatch_size_tokens // self.sequence_length

        total_tokens = self.rank_microbatch_size_tokens * dp_world_size * self.grad_accum_steps
        assert self.global_batch_size_tokens == total_tokens


@dataclass
class CacheDatasetArguments:
    """Arguments for caching HuggingFace datasets to numpy format."""

    output_dir: str = field(metadata={"help": "Output directory for numpy files"})
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-olmo-2-mixture", "1.0"])
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE)
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: str | None = None
    dataset_skip_cache: bool = False
    max_seq_length: int | None = None
    num_examples: int = 0
    visualize: bool = False


def build_sft_dataset(
    root_dir: str, tokenizer_config: TokenizerConfig, sequence_length: int, dataset_path: str
) -> NumpyPackedFSLDatasetConfig:
    """Build SFT dataset configuration for pre-tokenized numpy data."""
    clean_path = dataset_path.rstrip("/")
    token_id_paths = [f"{clean_path}/token_ids_part_*.npy"]
    label_mask_paths = [f"{clean_path}/labels_mask_part_*.npy"]

    dataset = NumpyPackedFSLDatasetConfig(
        tokenizer=tokenizer_config,
        work_dir=get_work_dir(root_dir),
        paths=token_id_paths,
        expand_glob=True,
        label_mask_paths=label_mask_paths,
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
        sequence_length=sequence_length,
    )

    return dataset


@dataclass
class SFTConfig(Config):
    """Configuration for SFT training."""

    run_name: str
    launch: BeakerLaunchConfig
    model: TransformerConfig
    dataset: NumpyPackedFSLDatasetConfig | None
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int

    @classmethod
    def build(
        cls,
        *,
        script: str,
        cmd: str,
        run_name: str,
        seq_len: int,
        num_nodes: int,
        global_batch_size: int,
        checkpoint: str,
        cluster: str,
        overrides: list[str],
        workspace: str,
        budget: str,
        init_seed: int = 42,
        dataset_path: str,
        learning_rate: float = 8e-5,
        warmup_ratio: float = 0.03,
        num_epochs: int = 3,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
    ) -> "SFTConfig":
        root_dir = get_root_dir(cluster)
        user_name = get_beaker_username()

        tokenizer_config = TokenizerConfig.dolma2()
        dataset_config = build_sft_dataset(
            root_dir=root_dir, tokenizer_config=tokenizer_config, sequence_length=seq_len, dataset_path=dataset_path
        )
        gpu_type = CLUSTER_TO_GPU_TYPE[cluster]

        bs_config = BatchSizeConfig(
            sequence_length=seq_len,
            world_size=num_nodes * GPUS_PER_NODE,
            global_batch_size_tokens=global_batch_size,
            gpu_type=gpu_type,
        )
        if get_local_rank() == 0:
            print("Batch size config (before overrides):")
            print(bs_config)

        dp_shard_degree = GPUS_PER_NODE
        if not dp_shard_degree > 0:
            raise OLMoConfigurationError(f"dp_shard_degree ({dp_shard_degree}) must be positive.")

        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules, modules=["blocks.*.feed_forward"]
        )

        dp_config = TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            shard_degree=dp_shard_degree,
        )

        model = TransformerConfig.olmo2_7B(vocab_size=tokenizer_config.padded_vocab_size())
        model.block.attention.use_flash = True

        config = SFTConfig(
            run_name=run_name,
            launch=build_launch_config(
                name=run_name,
                root_dir=root_dir,
                cmd=[
                    script,
                    cmd,
                    run_name,
                    checkpoint,
                    cluster,
                    f"--seq_len={seq_len}",
                    f"--num_nodes={num_nodes}",
                    f"--global_batch_size={global_batch_size}",
                    f"--budget={budget}",
                    f"--workspace={workspace}",
                    f"--dataset_path={dataset_path}",
                    *overrides,
                ],
                cluster=cluster,
                num_nodes=num_nodes,
                budget=budget,
                workspace=workspace,
            ),
            model=model,
            dataset=None,
            data_loader=NumpyDataLoaderConfig(
                global_batch_size=bs_config.global_batch_size_tokens, seed=init_seed + 1000, num_workers=4
            ),
            train_module=TransformerTrainModuleConfig(
                rank_microbatch_size=bs_config.rank_microbatch_size_tokens,
                max_sequence_length=bs_config.sequence_length,
                z_loss_multiplier=None,
                compile_model=True,
                optim=SkipStepAdamWConfig(lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95), compile=False),
                dp_config=dp_config,
                ac_config=ac_config,
                scheduler=LinearWithWarmup(warmup_fraction=warmup_ratio, alpha_f=0.0),
                max_grad_norm=1.0,
            ),
            trainer=TrainerConfig(
                save_folder=f"{root_dir}/checkpoints/{user_name}/olmo-sft/{run_name}",
                load_strategy=LoadStrategy.never,
                checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=32, throttle_uploads=True),
                save_overwrite=True,
                metrics_collect_interval=10,
                cancel_check_interval=10,
                max_duration=Duration.epochs(num_epochs),
            )
            .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
            .with_callback("config_saver", ConfigSaverCallback())
            .with_callback("garbage_collector", GarbageCollectorCallback())
            .with_callback(
                "checkpointer", CheckpointerCallback(save_interval=1000, ephemeral_save_interval=500, save_async=True)
            )
            .with_callback(
                "wandb",
                WandBCallback(
                    name=run_name,
                    entity=wandb_entity or "ai2-llm",
                    project=wandb_project or f"{user_name}-sft",
                    enabled=wandb_project is not None,
                    cancel_check_interval=10,
                ),
            ),
            init_seed=init_seed,
        ).merge(overrides)

        config.dataset = dataset_config

        print(config)

        return config


def write_memmap_chunked(base_filename: str, data: list, dtype, max_size_gb: float = 1) -> tuple[list, list]:
    """Write data to multiple memmap files if size exceeds max_size_gb."""
    item_size = np.dtype(dtype).itemsize
    max_size_bytes = max_size_gb * 1024**3

    chunk_size = int(max_size_bytes // item_size)
    chunks = []
    chunk_boundaries = []

    for i in range(0, len(data), chunk_size):
        chunk_data = data[i : i + chunk_size]
        filename = f"{base_filename}_part_{i // chunk_size:04d}.npy"
        mmap = np.memmap(filename, mode="w+", dtype=dtype, shape=(len(chunk_data),))
        mmap[:] = chunk_data
        mmap.flush()
        chunks.append(mmap)
        chunk_boundaries.append((i, i + len(chunk_data)))
        print(f"Written {filename} ({len(chunk_data) * item_size / 1024**3:.2f} GB)")

    return chunks, chunk_boundaries


def write_metadata_for_chunks(base_filename: str, document_boundaries: list, chunk_boundaries: list) -> None:
    """Write metadata files for each chunk with document boundaries."""
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_boundaries):
        metadata_filename = f"{base_filename}_part_{chunk_idx:04d}.csv.gz"

        with gzip.open(metadata_filename, "wt") as f:
            for doc_start, doc_end in document_boundaries:
                if doc_end > chunk_start and doc_start < chunk_end:
                    adjusted_start = max(0, doc_start - chunk_start)
                    adjusted_end = min(chunk_end - chunk_start, doc_end - chunk_start)

                    if adjusted_end > adjusted_start:
                        f.write(f"{adjusted_start},{adjusted_end}\n")

        print(f"Written metadata {metadata_filename}")


def write_dataset_statistics(
    output_dir: str,
    dataset_statistics: dict[str, Any],
    total_instances: int,
    total_tokens: int,
    total_trainable_tokens: int,
    num_samples_skipped: int,
    tokenizer_name: str,
    max_seq_length: int | None,
    chat_template_name: str | None,
    per_dataset_counts: dict[str, int],
    per_dataset_tokens: dict[str, int],
    per_dataset_trainable_tokens: dict[str, int],
    per_dataset_filtered: dict[str, int],
) -> None:
    """Write dataset statistics to JSON and text files."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    merged_stats = []
    pre_transform_stats = {stat["dataset_name"]: stat for stat in dataset_statistics.get("per_dataset_stats", [])}

    for dataset_name in per_dataset_counts:
        pre_stat = pre_transform_stats.get(dataset_name, {})
        merged_stat = {
            "dataset_name": dataset_name,
            "dataset_split": pre_stat.get("dataset_split", "unknown"),
            "initial_instances": pre_stat.get("initial_instances", "N/A"),
            "instances_after_transformation": pre_stat.get("final_instances", "N/A"),
            "final_instances_in_output": per_dataset_counts[dataset_name],
            "final_tokens_in_output": per_dataset_tokens[dataset_name],
            "final_trainable_tokens_in_output": per_dataset_trainable_tokens[dataset_name],
            "instances_filtered_after_tokenization": per_dataset_filtered[dataset_name],
            "avg_tokens_per_instance": per_dataset_tokens[dataset_name] / per_dataset_counts[dataset_name]
            if per_dataset_counts[dataset_name] > 0
            else 0,
            "percentage_of_total_tokens": (per_dataset_tokens[dataset_name] / total_tokens * 100)
            if total_tokens > 0
            else 0,
        }
        merged_stats.append(merged_stat)

    stats_data = {
        "timestamp": timestamp,
        "output_directory": output_dir,
        "configuration": {
            "tokenizer": tokenizer_name,
            "max_sequence_length": max_seq_length,
            "chat_template": chat_template_name,
        },
        "per_dataset_statistics": merged_stats,
        "overall_statistics": {
            "total_datasets": len(per_dataset_counts),
            "total_instances": total_instances,
            "total_tokens": total_tokens,
            "trainable_tokens": total_trainable_tokens,
            "trainable_percentage": (total_trainable_tokens / total_tokens * 100) if total_tokens > 0 else 0,
            "instances_filtered": num_samples_skipped,
            "average_sequence_length": total_tokens / total_instances if total_instances > 0 else 0,
        },
    }

    json_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"Written dataset statistics to {json_path}")

    text_path = os.path.join(output_dir, "dataset_statistics.txt")
    with open(text_path, "w") as f:
        f.write("Dataset Statistics Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {timestamp}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        f.write(f"Total instances: {total_instances:,}\n")
        f.write(f"Total tokens: {total_tokens:,}\n")
        f.write(
            f"Trainable tokens: {total_trainable_tokens:,} ({stats_data['overall_statistics']['trainable_percentage']:.1f}%)\n"
        )
    print(f"Written human-readable statistics to {text_path}")


def cache_dataset_only(args: CacheDatasetArguments, tc: OITokenizerConfig) -> None:
    """Convert HuggingFace dataset to numpy format for OLMo-core training."""
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        beaker_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        if os.path.exists(beaker_cache_dir):
            args.dataset_local_cache_dir = beaker_cache_dir

    print("Tokenizer configuration:")
    print(f"  vocab_size: {tc.tokenizer.vocab_size}")
    print(f"  bos_token_id: {tc.tokenizer.bos_token_id}")
    print(f"  pad_token_id: {tc.tokenizer.pad_token_id}")
    print(f"  eos_token_id: {tc.tokenizer.eos_token_id}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_output_dir, exist_ok=True)
    print(f"Saving tokenizer to {tokenizer_output_dir}...")
    tc.tokenizer.save_pretrained(tokenizer_output_dir)

    chat_template_path = os.path.join(tokenizer_output_dir, "chat_template.jinja")
    tokenizer_config_path = os.path.join(tokenizer_output_dir, "tokenizer_config.json")
    if os.path.exists(chat_template_path) and os.path.exists(tokenizer_config_path):
        with open(chat_template_path) as f:
            chat_template_content = f.read()
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        if "chat_template" not in tokenizer_config:
            tokenizer_config["chat_template"] = chat_template_content
            with open(tokenizer_config_path, "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            print(f"Added chat_template from {chat_template_path} to tokenizer_config.json")

    transform_functions_and_args = [
        ("sft_tulu_tokenize_and_truncate_v1", {"max_seq_length": args.max_seq_length}),
        ("sft_tulu_filter_v1", {}),
    ]

    train_dataset, dataset_statistics = get_cached_dataset_tulu_with_statistics(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=[func for func, _ in transform_functions_and_args],
        transform_fn_args=[args for _, args in transform_functions_and_args],
        target_columns=args.dataset_target_columns,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
        drop_dataset_source=False,
    )

    train_dataset = train_dataset.shuffle()

    if args.visualize:
        print("Visualizing first example...")
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tc.tokenizer)

    if args.num_examples > 0:
        print(f"Selecting {args.num_examples} examples for debugging")
        train_dataset = train_dataset.select(range(args.num_examples))

    print("Collecting tokens from dataset...")
    token_ids = []
    labels_mask = []
    num_samples_skipped = 0
    document_boundaries = []
    current_position = 0

    per_dataset_counts: dict[str, int] = {}
    per_dataset_tokens: dict[str, int] = {}
    per_dataset_trainable_tokens: dict[str, int] = {}
    per_dataset_filtered: dict[str, int] = {}

    for sample in tqdm(train_dataset, desc="Collecting tokens", bar_format="{l_bar}{bar}{r_bar}\n"):
        sample_length = len(sample[INPUT_IDS_KEY])
        sample_tokens = sample[INPUT_IDS_KEY]
        sample_labels = sample[LABELS_KEY]
        dataset_source = sample.get(DATASET_ORIGIN_KEY, "unknown")

        if dataset_source not in per_dataset_counts:
            per_dataset_counts[dataset_source] = 0
            per_dataset_tokens[dataset_source] = 0
            per_dataset_trainable_tokens[dataset_source] = 0
            per_dataset_filtered[dataset_source] = 0

        per_dataset_counts[dataset_source] += 1
        per_dataset_tokens[dataset_source] += sample_length
        trainable_tokens_in_sample = sum(1 for label in sample_labels if label != -100)
        per_dataset_trainable_tokens[dataset_source] += trainable_tokens_in_sample

        token_ids.extend(sample_tokens)
        labels_mask.extend([1 if label != -100 else 0 for label in sample_labels])

        document_boundaries.append((current_position, current_position + sample_length))
        current_position += sample_length

        if all(label == -100 for label in sample_labels):
            num_samples_skipped += 1
            per_dataset_filtered[dataset_source] += 1

        assert all(mask == 1 for mask in sample[ATTENTION_MASK_KEY]), "Expected all attention mask values to be 1"

    train_dataset = remove_dataset_source_field(train_dataset)

    total_instances = len(train_dataset)
    total_tokens = len(token_ids)
    total_trainable_tokens = sum(labels_mask)

    print(f"Total sequences: {total_instances}")
    print(f"Total tokens: {total_tokens}")
    print(f"Trainable tokens: {total_trainable_tokens}")

    vocab_size = tc.tokenizer.vocab_size
    token_dtype = None
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if (vocab_size - 1) <= np.iinfo(dtype).max:
            token_dtype = dtype
            print(f"Using dtype '{dtype}' for token_ids based on vocab size {vocab_size}")
            break
    if token_dtype is None:
        raise ValueError(f"Vocab size {vocab_size} is too big for any numpy integer dtype!")

    print(f"Writing converted data to {output_dir}")
    _, token_chunk_boundaries = write_memmap_chunked(f"{output_dir}/token_ids", token_ids, token_dtype)
    write_metadata_for_chunks(f"{output_dir}/token_ids", document_boundaries, token_chunk_boundaries)

    for i, (start, end) in enumerate(token_chunk_boundaries):
        chunk_data = labels_mask[start:end]
        filename = f"{output_dir}/labels_mask_part_{i:04d}.npy"
        mmap = np.memmap(filename, mode="w+", dtype=np.bool_, shape=(len(chunk_data),))
        mmap[:] = chunk_data
        mmap.flush()
        print(f"Written {filename}")

    write_dataset_statistics(
        output_dir=output_dir,
        dataset_statistics=dataset_statistics,
        total_instances=total_instances,
        total_tokens=total_tokens,
        total_trainable_tokens=total_trainable_tokens,
        num_samples_skipped=num_samples_skipped,
        tokenizer_name=tc.tokenizer_name_or_path,
        max_seq_length=args.max_seq_length,
        chat_template_name=tc.chat_template_name,
        per_dataset_counts=per_dataset_counts,
        per_dataset_tokens=per_dataset_tokens,
        per_dataset_trainable_tokens=per_dataset_trainable_tokens,
        per_dataset_filtered=per_dataset_filtered,
    )

    print("Dataset caching completed successfully!")


def train(checkpoint: str, config: SFTConfig, no_save_tokenizer: bool = False) -> None:
    """Execute SFT training using OLMo-core Trainer."""
    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)

    if config.dataset is not None:
        dataset = config.dataset.build()
        data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = config.trainer.build(train_module, data_loader)

        if not no_save_tokenizer and get_rank() == 0:
            tokenizer_path = join_path(get_parent(dataset.paths[0]), "tokenizer")
            if not dir_is_empty(tokenizer_path):
                log.info("Saving tokenizer...")
                destination_path = join_path(trainer.save_folder, "tokenizer")
                if not dir_is_empty(destination_path):
                    log.info(f"Tokenizer already exists: {destination_path}")
                else:
                    log.info(f"Saving tokenizer to {destination_path}")
                    copy_dir(tokenizer_path, destination_path)

        config_dict = config.as_config_dict()
        cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

        log.info("Loading checkpoint...")
        if not trainer.maybe_load_checkpoint(trainer.save_folder):
            log.info(f"No checkpoint found in save folder '{trainer.save_folder}', loading from '{checkpoint}'")
            trainer.load_checkpoint(checkpoint, load_trainer_state=False)
        else:
            log.info(f"Loaded checkpoint from save folder '{trainer.save_folder}'")

        trainer.fit()
    else:
        log.error(f"Config dataset is None: {config}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SFT for OLMo models using OLMo-core.", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="cmd", help="Subcommand to run")

    # cache_dataset_only subcommand
    cache_parser = subparsers.add_parser("cache_dataset_only", help="Convert HF datasets to numpy format")
    cache_parser.add_argument("--output_dir", required=True, help="Output directory for numpy files")
    cache_parser.add_argument("--dataset_mixer_list", nargs="+", default=["allenai/tulu-3-sft-olmo-2-mixture", "1.0"])
    cache_parser.add_argument("--dataset_mixer_list_splits", nargs="+", default=["train"])
    cache_parser.add_argument("--max_seq_length", type=int, default=4096)
    cache_parser.add_argument("--num_examples", type=int, default=0)
    cache_parser.add_argument("--visualize", action="store_true")
    cache_parser.add_argument("--tokenizer_name_or_path", default="allenai/OLMo-2-1124-7B")
    cache_parser.add_argument("--chat_template_name", default="olmo")

    # launch subcommand
    launch_parser = subparsers.add_parser("launch", help="Launch training on Beaker")
    launch_parser.add_argument("run_name", help="Name of the run")
    launch_parser.add_argument("pretrain_checkpoint", help="Path to pretrain checkpoint")
    launch_parser.add_argument("cluster", help="Beaker cluster")
    launch_parser.add_argument("--dataset_path", required=True, help="Path to pre-tokenized dataset")
    launch_parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    launch_parser.add_argument("--num_nodes", type=int, default=DEFAULT_NUM_NODES)
    launch_parser.add_argument("--global_batch_size", type=int, default=64 * DEFAULT_SEQUENCE_LENGTH)
    launch_parser.add_argument("--budget", required=True, help="Beaker budget")
    launch_parser.add_argument("--workspace", required=True, help="Beaker workspace")
    launch_parser.add_argument("--learning_rate", type=float, default=8e-5)
    launch_parser.add_argument("--warmup_ratio", type=float, default=0.03)
    launch_parser.add_argument("--num_epochs", type=int, default=3)
    launch_parser.add_argument("--wandb_project", default=None)
    launch_parser.add_argument("--wandb_entity", default=None)
    launch_parser.add_argument("--follow", action="store_true")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Execute training")
    train_parser.add_argument("run_name", help="Name of the run")
    train_parser.add_argument("pretrain_checkpoint", help="Path to pretrain checkpoint")
    train_parser.add_argument("cluster", help="Beaker cluster")
    train_parser.add_argument("--dataset_path", help="Path to pre-tokenized dataset")
    train_parser.add_argument("--dataset_mixer_list", nargs="+", help="HF datasets to mix (e.g., dataset1 1.0)")
    train_parser.add_argument("--cache_dataset_only", action="store_true", help="Cache dataset and exit")
    train_parser.add_argument("--output_dir", default="output/", help="Output dir for cached dataset")
    train_parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    train_parser.add_argument("--num_nodes", type=int, default=DEFAULT_NUM_NODES)
    train_parser.add_argument("--global_batch_size", type=int, default=64 * DEFAULT_SEQUENCE_LENGTH)
    train_parser.add_argument("--budget", default="ai2/oe-training")
    train_parser.add_argument("--workspace", default="ai2/tulu-3")
    train_parser.add_argument("--learning_rate", type=float, default=8e-5)
    train_parser.add_argument("--warmup_ratio", type=float, default=0.03)
    train_parser.add_argument("--num_epochs", type=int, default=3)
    train_parser.add_argument("--wandb_project", default=None)
    train_parser.add_argument("--wandb_entity", default=None)
    train_parser.add_argument("--no_save_tokenizer", action="store_true")

    # dry_run subcommand
    dry_run_parser = subparsers.add_parser("dry_run", help="Validate configuration")
    dry_run_parser.add_argument("run_name", help="Name of the run")
    dry_run_parser.add_argument("pretrain_checkpoint", help="Path to pretrain checkpoint")
    dry_run_parser.add_argument("cluster", help="Beaker cluster")
    dry_run_parser.add_argument("--dataset_path", required=True, help="Path to pre-tokenized dataset")
    dry_run_parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    dry_run_parser.add_argument("--num_nodes", type=int, default=DEFAULT_NUM_NODES)
    dry_run_parser.add_argument("--global_batch_size", type=int, default=64 * DEFAULT_SEQUENCE_LENGTH)
    dry_run_parser.add_argument("--budget", default="ai2/oe-training")
    dry_run_parser.add_argument("--workspace", default="ai2/tulu-3")

    args, overrides = parser.parse_known_args()

    if args.cmd == "cache_dataset_only":
        tc = OITokenizerConfig(
            tokenizer_name_or_path=args.tokenizer_name_or_path, chat_template_name=args.chat_template_name
        )
        cache_args = CacheDatasetArguments(
            output_dir=args.output_dir,
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            max_seq_length=args.max_seq_length,
            num_examples=args.num_examples,
            visualize=args.visualize,
        )
        cache_dataset_only(cache_args, tc)
        return

    if args.cmd == "train" and getattr(args, "cache_dataset_only", False):
        dataset_mixer_list = getattr(args, "dataset_mixer_list", None)
        if dataset_mixer_list is None:
            raise ValueError("--dataset_mixer_list required with --cache_dataset_only")
        tc = OITokenizerConfig(tokenizer_name_or_path=args.pretrain_checkpoint, chat_template_name="olmo")
        cache_args = CacheDatasetArguments(
            output_dir=getattr(args, "output_dir", "output/"),
            dataset_mixer_list=dataset_mixer_list,
            dataset_mixer_list_splits=["train"],
            max_seq_length=args.seq_len,
            num_examples=0,
            visualize=False,
        )
        cache_dataset_only(cache_args, tc)
        return

    if args.cmd in ("launch", "dry_run"):
        prepare_cli_environment()
    elif args.cmd == "train":
        prepare_training_environment()
    else:
        parser.print_help()
        return

    dataset_path = getattr(args, "dataset_path", None)
    if dataset_path is None:
        raise ValueError("--dataset_path required for training (use --cache_dataset_only first)")

    config = SFTConfig.build(
        script=sys.argv[0],
        cmd="train",
        run_name=args.run_name,
        checkpoint=args.pretrain_checkpoint,
        cluster=args.cluster,
        seq_len=args.seq_len,
        num_nodes=args.num_nodes,
        global_batch_size=args.global_batch_size,
        overrides=overrides,
        budget=args.budget,
        workspace=args.workspace,
        dataset_path=dataset_path,
        learning_rate=getattr(args, "learning_rate", 8e-5),
        warmup_ratio=getattr(args, "warmup_ratio", 0.03),
        num_epochs=getattr(args, "num_epochs", 3),
        wandb_project=getattr(args, "wandb_project", None),
        wandb_entity=getattr(args, "wandb_entity", None),
    )

    if get_local_rank() == 0:
        print(config)

    if args.cmd == "dry_run":
        print("Dry run completed. Configuration is valid.")
    elif args.cmd == "launch":
        config.launch.launch(follow=args.follow)
    elif args.cmd == "train":
        no_save_tokenizer = getattr(args, "no_save_tokenizer", False)
        try:
            train(args.pretrain_checkpoint, config, no_save_tokenizer)
        finally:
            teardown_training_environment()


if __name__ == "__main__":
    main()
