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
import json
import os
import pathlib
import re
import shlex
from typing import Any, cast

import torch
import torch.distributed as dist
from olmo_core import data as oc_data
from olmo_core import optim
from olmo_core.config import DType
from olmo_core.distributed import parallel
from olmo_core.distributed.utils import is_distributed
from olmo_core.nn.ddp import model as ddp_model_lib
from olmo_core.train import Duration, LoadStrategy, TrainerConfig, callbacks, teardown_training_environment
from olmo_core.train import train_module as train_module_lib
from olmo_core.train.checkpoint import CheckpointerConfig

from open_instruct import (
    atomic_think,
    atomic_warmup,
    dataset_transformation,
    logger_utils,
    numpy_dataset_conversion,
    olmo_core_utils,
    packed_sample_cache,
    resolved_sft_dataset,
    utils,
)

logger = logger_utils.setup_logger(__name__)


# Must stay strictly below CheckpointConfig.checkpointing_steps (500), which olmo-core
# requires;
_DEFAULT_EPHEMERAL_SAVE_INTERVAL = 250

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
class SFTConfig:
    """Settings read only by this script.

    They stay out of the shared configs because DPO and GRPO inherit those and
    would advertise flags neither trainer reads.
    """

    dist_timeout_hours: float = 24
    """Timeout for distributed collectives, in hours."""
    save_async: bool = True
    """Whether olmo-core saves checkpoints asynchronously."""
    tracking_url: str | None = None
    """Optional URL (GitHub issue, ticket, experiment log) recorded in the run
    directory's provenance README so any copy of a checkpoint traces back to it."""
    atomic_audit_mode: str | None = None
    """H077 only: control, rounding-only, atomic-safe, or atomic-historical."""
    warmup_manifest: str | None = None
    """Pinned FP32 warmup matrices for fresh SFT initialization; never reapplied on resume."""
    warmup_manifest_sha256: str | None = None
    scheduler_horizon: int | None = None
    """Fix the LR horizon independently of the stopping step for partial SFT."""
    atomic_state_steps: list[int] = dataclasses.field(default_factory=list)
    """Expensive complete state hashes for the short checkpoint/noninterference gate only."""
    atomic_probe_manifest: str | None = None
    """Frozen, tokenized H077 probe JSON, shared by all arms."""
    atomic_probe_steps: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 10, 100, 353, 512])
    atomic_probe_offline: bool = False
    """Save native measurement checkpoints and model hashes instead of probing inside training."""
    atomic_verify_resume: bool = False
    """Require exact native model/optimizer/trainer restoration before H077 continuation."""
    atomic_batch_steps: list[int] | None = None
    """Explicit bounded batch audit steps; None preserves the original short-run audit."""
    sft_dataset_manifest: str | None = None
    """Read CPU-resolved raw-dataset metadata; still verify the exact tokenizer/cache identity."""
    write_sft_dataset_manifest: str | None = None
    """Write resolved metadata during --cache_dataset_only; never permitted during training."""
    packed_sample_cache: str | None = None
    """Optional exact packed-sample cache; preserves native dataset identity and loader order."""


@dataclasses.dataclass
class SFTArguments:
    tracking: olmo_core_utils.ExperimentConfig
    model: olmo_core_utils.ModelConfig
    training: olmo_core_utils.TrainingConfig
    dataset: olmo_core_utils.DatasetConfig
    logging: olmo_core_utils.LoggingConfig
    checkpoint: olmo_core_utils.CheckpointConfig
    sft: SFTConfig


def main(args: SFTArguments, tc: dataset_transformation.TokenizerConfig) -> None:
    if args.sft.write_sft_dataset_manifest and (not args.dataset.cache_dataset_only or args.sft.sft_dataset_manifest):
        raise ValueError("Writing a dataset manifest requires cache-only mode without an input manifest")
    atomic_think.validate_config(args.sft, args.training, tc)
    use_hf_ckpt = olmo_core_utils.is_hf_checkpoint(args.model.model_name_or_path)

    olmo_core_utils.setup_tokenizer_and_cache(args.model, args.dataset, tc)
    if args.sft.atomic_probe_manifest:
        manifest = json.loads(pathlib.Path(args.sft.atomic_probe_manifest).read_text())
        if tc.tokenizer_files_hash != manifest["tokenizer_files_hash"]:
            raise ValueError("Training tokenizer differs from the frozen probe tokenizer")
    transform_fn_args = [
        dataset_transformation.sft_tokenize_fn_args(args.training.max_seq_length, args.training.over_length_strategy),
        {},
    ]

    dataset_request = {
        "dataset_mixer_list": args.dataset.mixer_list,
        "dataset_mixer_list_splits": args.dataset.mixer_list_splits,
        "dataset_transform_fn": args.dataset.transform_fn,
        "transform_fn_args": transform_fn_args,
        "target_columns": list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE),
        "dataset_config_seed": 42,
    }
    dcs = None
    if args.sft.sft_dataset_manifest:
        cache_hash, manifest_hash = resolved_sft_dataset.read(args.sft.sft_dataset_manifest, dataset_request, tc)
        logger.info(
            "Verified resolved SFT dataset manifest %s (SHA256 %s)", args.sft.sft_dataset_manifest, manifest_hash
        )
    else:
        dcs = dataset_transformation.load_dataset_configs(
            dataset_mixer_list=args.dataset.mixer_list,
            dataset_mixer_list_splits=args.dataset.mixer_list_splits,
            dataset_transform_fn=args.dataset.transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=list(dataset_transformation.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE),
            dataset_config_seed=42,
        )
        cache_hash = dataset_transformation.compute_config_hash(dcs, tc)
    seed_suffix = _seed_cache_suffix(args.tracking.seed, args.training.max_seq_length)
    numpy_dir = os.path.join(args.dataset.local_cache_dir, _NUMPY_SFT_SUBDIR, f"{cache_hash}-{seed_suffix}")
    # A launch that must reuse one specific cache names it, and dies here -- before tokenizing,
    # distributed init or a GPU -- if its arguments resolve to any other (H015).
    expected_cache = os.environ.get("EXPECTED_NUMPY_CACHE")
    if expected_cache and os.path.basename(numpy_dir) != expected_cache:
        raise RuntimeError(f"Resolved numpy cache {numpy_dir}, but EXPECTED_NUMPY_CACHE={expected_cache}")

    if args.dataset.cache_dataset_only:
        pre_init_rank = int(os.environ.get("RANK", 0))
        if pre_init_rank == 0:
            if _numpy_dir_is_populated(numpy_dir):
                logger.info(f"Numpy SFT files already present at {numpy_dir}; nothing to do.")
            else:
                if args.sft.sft_dataset_manifest:
                    raise FileNotFoundError("Resolved metadata requires an already populated NumPy cache")
                _tokenize_to_numpy_dir(numpy_dir, args, tc, transform_fn_args, visualize=True)
            if args.sft.write_sft_dataset_manifest:
                assert dcs is not None
                resolved_sft_dataset.write(args.sft.write_sft_dataset_manifest, dataset_request, dcs, tc)
                logger.info("Saved resolved SFT dataset manifest to %s", args.sft.write_sft_dataset_manifest)
            logger.info("Dataset cached successfully. Exiting because --cache_dataset_only was set.")
        return

    if not _numpy_dir_is_populated(numpy_dir):
        cache_args = [
            f"--model_name_or_path {args.model.model_name_or_path}",
            f"--tokenizer_name_or_path {tc.tokenizer_name_or_path}",
            f"--max_seq_length {args.training.max_seq_length}",
            f"--mixer_list {' '.join(args.dataset.mixer_list)}",
            f"--mixer_list_splits {' '.join(args.dataset.mixer_list_splits)}",
            f"--seed {args.tracking.seed}",
        ]
        if tc.chat_template_name is not None:
            cache_args.append(f"--chat_template_name {tc.chat_template_name}")
        if tc.add_bos:
            cache_args.append("--add_bos")
        if args.dataset.transform_fn:
            cache_args.append(f"--transform_fn {' '.join(args.dataset.transform_fn)}")
        # Part of the cache hash.
        if args.training.over_length_strategy != dataset_transformation.DEFAULT_OVER_LENGTH_STRATEGY:
            cache_args.append(f"--over_length_strategy {args.training.over_length_strategy}")
        # Also part of the cache hash, and the values contain shell metacharacters: an
        # unquoted `<think>` would be a redirection, and a command that dropped the flag
        # would tokenize to a different hash than the one this job is looking for.
        if tc.reserved_slot_tokens:
            quoted = " ".join(shlex.quote(token) for token in tc.reserved_slot_tokens)
            cache_args.append(f"--reserved_slot_tokens {quoted}")
        cache_args += [f"--local_cache_dir {args.dataset.local_cache_dir}", "--cache_dataset_only"]
        cache_cmd = " \\\n      ".join(cache_args)
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
            f"      {cache_cmd}\n\n"
            "Re-launch training once the tokenization job has completed."
        )

    global_rank, world_size, is_main_process = olmo_core_utils.setup_distributed_env(
        seed=args.tracking.seed, timeout=datetime.timedelta(hours=args.sft.dist_timeout_hours)
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

    ac_config = olmo_core_utils.build_ac_config(
        args.training.activation_memory_budget,
        args.training.compile_model,
        args.training.activation_checkpointing_mode,
        args.training.activation_checkpointing_modules,
    )

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
    if args.sft.packed_sample_cache:
        np_dataset = packed_sample_cache.CachedPackedFSLDataset(np_dataset, args.sft.packed_sample_cache)

    global_batch_size_seqs = rank_batch_size_seqs * dp_world_size

    num_training_steps = len(np_dataset) // global_batch_size_seqs * args.training.num_epochs
    effective_steps = (
        args.training.max_train_steps if args.training.max_train_steps is not None else num_training_steps
    )
    logger.info(f"Total training steps: {effective_steps} (epochs={args.training.num_epochs})")
    if effective_steps < 1:
        raise ValueError(
            f"Computed {effective_steps} training steps from {len(np_dataset)} packed instances "
            f"// global batch {global_batch_size_seqs} * {args.training.num_epochs} epochs. "
            "Use more data, or lower --gradient_accumulation_steps / --per_device_train_batch_size."
        )
    scheduler = olmo_core_utils.build_scheduler(
        args.training.lr_scheduler_type, args.training.warmup_ratio, args.sft.scheduler_horizon or effective_steps
    )
    if args.sft.scheduler_horizon is not None:
        scheduler.t_max = args.sft.scheduler_horizon

    max_grad_norm = (
        args.training.max_grad_norm if args.training.max_grad_norm and args.training.max_grad_norm > 0 else None
    )
    if isinstance(model, ddp_model_lib.OLMoDDPModel):
        # MoE v2 models refuse FSDP; they train through the branch's own DDP train
        # module, configured from the checkpoint's train_module section. Requires
        # --config_name to be the checkpoint config json.
        assert args.model.config_name is not None and args.model.config_name.endswith(".json"), (
            "OLMoDDPModel requires --config_name pointing at the (migrated) checkpoint config json"
        )
        assert cp_config is None, "context parallelism is not wired for the DDP train module"
        train_module_config = olmo_core_utils.build_ddp_train_module_config(
            args.model.config_name,
            rank_microbatch_size=rank_microbatch_size,
            max_sequence_length=args.training.max_seq_length,
            learning_rate=args.training.learning_rate,
            weight_decay=args.training.weight_decay,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
            compile_model=args.training.compile_model,
            ac_config=ac_config,
        )
    else:
        train_module_config = train_module_lib.TransformerTrainModuleConfig(
            rank_microbatch_size=rank_microbatch_size,
            max_sequence_length=args.training.max_seq_length,
            z_loss_multiplier=None,
            compile_model=args.training.compile_model,
            optim=optim.SkipStepAdamWConfig(
                lr=args.training.learning_rate,
                weight_decay=args.training.weight_decay,
                betas=(0.9, 0.95),
                compile=False,
            ),
            dp_config=dp_config,
            cp_config=cp_config,
            ac_config=ac_config,
            scheduler=scheduler,
            max_grad_norm=max_grad_norm,
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
    if args.sft.atomic_audit_mode is not None:
        config_dict["tokenizer"] = dataclasses.asdict(tc)
        if is_main_process:
            tc.tokenizer.save_pretrained(os.path.join(args.checkpoint.output_dir, "tokenizer"))

    if is_main_process:
        olmo_core_utils.write_provenance_readme(
            output_dir=args.checkpoint.output_dir,
            run_name=run_name,
            model_name_or_path=args.model.model_name_or_path,
            tracking_url=args.sft.tracking_url,
            wandb_project=args.logging.wandb_project if args.logging.with_tracking else None,
            wandb_entity=args.logging.wandb_entity,
        )

    trainer_callbacks: dict[str, Any] = olmo_core_utils.build_base_callbacks(
        config_dict=config_dict,
        run_name=run_name,
        checkpointing_steps=args.checkpoint.checkpointing_steps,
        ephemeral_save_interval=args.checkpoint.ephemeral_save_interval,
        with_tracking=args.logging.with_tracking,
        wandb_project=args.logging.wandb_project,
        wandb_entity=args.logging.wandb_entity or "ai2-llm",
        max_checkpoints=args.checkpoint.keep_last_n_checkpoints,
        save_async=args.sft.save_async,
    )
    trainer_callbacks["config_saver"] = callbacks.ConfigSaverCallback(_config=config_dict)
    trainer_callbacks["garbage_collector"] = callbacks.GarbageCollectorCallback()
    if args.sft.atomic_audit_mode is not None:
        assert args.sft.scheduler_horizon is not None
        trainer_callbacks["atomic_audit"] = atomic_think.AtomicAuditCallback(
            mode=args.sft.atomic_audit_mode,
            horizon=args.sft.scheduler_horizon,
            manifest=args.sft.atomic_probe_manifest,
            steps=[] if args.sft.atomic_probe_offline else args.sft.atomic_probe_steps,
            state_steps=args.sft.atomic_state_steps,
            model_state_steps=args.sft.atomic_probe_steps if args.sft.atomic_probe_offline else [],
            offline=args.sft.atomic_probe_offline,
            verify_resume=args.sft.atomic_verify_resume,
            batch_steps=args.sft.atomic_batch_steps,
            output_dir=os.path.join(args.checkpoint.output_dir, "atomic-audit"),
        )
        if args.sft.atomic_probe_offline:
            checkpointer = trainer_callbacks["checkpointer"]
            checkpointer.fixed_steps = sorted({step for step in args.sft.atomic_probe_steps if step > 0})

    if bool(args.sft.warmup_manifest) != bool(args.sft.warmup_manifest_sha256):
        raise ValueError("Warmup initialization requires both manifest and SHA256")
    if args.sft.warmup_manifest:
        if use_hf_ckpt or args.sft.atomic_audit_mode not in ("control", "atomic-safe"):
            raise ValueError("Warmup handoff requires native control/atomic-safe SFT")
        if not re.fullmatch(r"[0-9a-f]{64}", args.sft.warmup_manifest_sha256):
            raise ValueError("Invalid warmup manifest SHA256")
        trainer_callbacks["warmup_identity"] = atomic_warmup.WarmupIdentity(args.sft.warmup_manifest_sha256)

    trainer = TrainerConfig(
        save_folder=args.checkpoint.output_dir,
        load_strategy=LoadStrategy.never,
        max_duration=max_duration,
        metrics_collect_interval=args.logging.logging_steps,
        callbacks=trainer_callbacks,
        save_overwrite=True,
        checkpointer=CheckpointerConfig(save_thread_count=1, load_thread_count=32, throttle_uploads=True),
    ).build(train_module, data_loader)

    # Loaded here rather than by fit(), which skips its own load once anything has
    # been loaded: putting the base weights in first would suppress it. Precedence
    # is fit()'s own -- an interrupted run in output_dir, then an explicit resume,
    # then the base weights.
    resumed = trainer.maybe_load_checkpoint(args.checkpoint.output_dir, load_trainer_state=True, load_optim_state=True)
    if args.checkpoint.resume_from_checkpoint is not None:
        if resumed:
            logger.warning(
                f"Ignoring --resume_from_checkpoint ({args.checkpoint.resume_from_checkpoint}) "
                f"since a checkpoint was found in {args.checkpoint.output_dir}"
            )
        else:
            logger.info(f"Resuming from {args.checkpoint.resume_from_checkpoint}...")
            # Raises when the path holds no checkpoint: a typo here must not
            # silently fall back to an expensive restart from the base weights.
            trainer.load_checkpoint(args.checkpoint.resume_from_checkpoint)
            resumed = True
    if not resumed and not use_hf_ckpt:
        logger.info(f"Loading olmo-core checkpoint from {args.model.model_name_or_path}...")
        # model_name_or_path is weight initialization, not a resume: skip the optimizer
        # state (SFT starts a fresh optimizer) as well as the trainer state (step counter,
        # LR schedule position, data order).
        trainer.load_checkpoint(args.model.model_name_or_path, load_trainer_state=False, load_optim_state=False)

    # Last point at which the base weights are settled: the HF path loaded them before the
    # trainer was built, the olmo-core path just above, and the trainer is configured with
    # LoadStrategy.never so fit() loads nothing of its own. Skipped when resuming, since those
    # rows have been trained since they were seeded.
    if resumed and args.sft.warmup_manifest and not trainer_callbacks["warmup_identity"].restored:
        raise ValueError("Checkpoint did not restore the requested warmup identity")
    if not resumed and args.sft.warmup_manifest:
        manifest, matrices = atomic_warmup.load_manifest(
            args.sft.warmup_manifest,
            args.sft.warmup_manifest_sha256,
            args.model.model_name_or_path,
            tc.tokenizer,
            args.sft.atomic_audit_mode,
        )
        atomic_warmup.inject(
            train_module,
            manifest,
            matrices,
            os.path.join(args.checkpoint.output_dir, "atomic-audit"),
            args.sft.warmup_manifest_sha256,
        )
        del matrices
    elif not resumed:
        row_check = olmo_core_utils.PromotedRowStepCheck.before_seeding(train_module, tc.tokenizer)
        if args.sft.atomic_audit_mode is None:
            olmo_core_utils.initialize_promoted_token_embeddings(train_module, tc.tokenizer)
        else:
            atomic_think.initialize_arm(
                train_module,
                tc.tokenizer,
                args.sft.atomic_audit_mode,
                os.path.join(args.checkpoint.output_dir, "atomic-audit"),
                olmo_core_utils.sync_promoted_token_optimizer_rows,
            )
        if row_check is not None:
            row_check.record_seeded()
            trainer.add_callback("promoted_row_step_check", row_check)

    if args.sft.atomic_probe_offline:
        checkpointer = cast(callbacks.CheckpointerCallback, trainer.callbacks["checkpointer"])
        checkpointer.pre_train_checkpoint = not resumed and 0 in args.sft.atomic_probe_steps

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
            SFTConfig,
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
    tracking, model, training, dataset, logging_cfg, checkpoint, sft, tc = parser.parse()  # ty: ignore[invalid-assignment, not-iterable]
    args = SFTArguments(
        tracking=tracking,
        model=model,
        training=training,
        dataset=dataset,
        logging=logging_cfg,
        checkpoint=checkpoint,
        sft=sft,
    )
    main(args, tc)
