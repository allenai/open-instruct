# !/usr/bin/env python
# Copyright 2026 AllenAI. All rights reserved.
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
Multimodal SFT (Molmo stage 2) on OLMo-core (docs/design/multimodal_sft.md).

A thin wrapper over OLMo-core's vision-branch components — ``MultimodalLM``,
``MultimodalTransformerTrainModule``, ``MixtureDataLoader`` — with the mixture
assembled by ``sft_mixture`` so the nlp group can be open-instruct's own SFT mix.

``MOLMO_DATA_DIR`` must point at the multimodal data root (weka:
``/weka/oe-training-default/mm-olmo``) in the launch environment; it is frozen at
import time by ``olmo_core.data.multimodal.paths``.

Usage (1-GPU smoke, debug mixture):
    torchrun --nproc_per_node=1 open_instruct/olmo_core_mixture_finetune.py \
        --mixture debug --max_train_steps 10 \
        --global_batch_instances 1 --rank_microbatch_instances 1 \
        --compile_model false --compile_vision false --compile_connector false

The merged single stage (image mixture + open-instruct text mix):
    torchrun --nproc_per_node=8 open_instruct/olmo_core_mixture_finetune.py \
        --mixture image-only-v9 --nlp_source open_instruct \
        --mixer_list allenai/Dolci-Instruct-SFT 1.0
"""

import dataclasses
import os

import torch
from olmo_core.data.multimodal import MixtureDataLoader, paths
from olmo_core.distributed import utils as dist_utils
from olmo_core.train import Duration, TrainerConfig, teardown_training_environment
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.common import LoadStrategy
from transformers import AutoTokenizer

from open_instruct import (
    logger_utils,
    olmo_core_callbacks,
    olmo_core_multimodal_utils,
    olmo_core_utils,
    sft_mixture,
    sft_text_dataset,  # noqa: F401  (registers the open_instruct_sft source type)
    utils,
)

logger = logger_utils.setup_logger(__name__)

_DEFAULT_EPHEMERAL_SAVE_INTERVAL = 250


@dataclasses.dataclass
class MultimodalSFTArguments:
    tracking: olmo_core_utils.ExperimentConfig
    model: olmo_core_multimodal_utils.MultimodalModelConfig
    training: olmo_core_multimodal_utils.MultimodalTrainingConfig
    mixture: sft_mixture.MixtureConfig
    logging: olmo_core_utils.LoggingConfig
    checkpoint: olmo_core_utils.CheckpointConfig


def main(args: MultimodalSFTArguments) -> None:
    if not os.path.isdir(paths.MOLMO_DATA_DIR):
        raise FileNotFoundError(
            f"MOLMO_DATA_DIR={paths.MOLMO_DATA_DIR} is not a directory. The multimodal datasets live on "
            f"weka (/weka/oe-training-default/mm-olmo); launch on a weka cluster (e.g. ai2/jupiter) or set "
            f"MOLMO_DATA_DIR in the launch environment (mason.py --env MOLMO_DATA_DIR=...)."
        )

    _, world_size, _ = olmo_core_utils.setup_distributed_env(args.tracking.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model.tokenizer_name_or_path or args.model.base_hf_model_id,
        trust_remote_code=args.model.trust_remote_code,
    )

    model, model_config, defer_load = olmo_core_multimodal_utils.setup_multimodal_model(args.model)
    if args.mixture.nlp_source == "open_instruct" and args.mixture.text_base_vocab_size is None:
        args.mixture.text_base_vocab_size = model_config.lm.vocab_size

    train_module = olmo_core_multimodal_utils.build_multimodal_train_module_config(
        args.training, world_size=world_size, gpus_per_node=max(torch.cuda.device_count(), 1)
    ).build(model)

    collator = olmo_core_multimodal_utils.build_multimodal_collator_config(
        tokenizer, args.training.max_seq_length
    ).build()

    dp_pg = train_module.dp_process_group
    dp_world_size = dist_utils.get_world_size(dp_pg)
    dp_rank = dist_utils.get_rank(dp_pg)

    data_seed = args.tracking.data_loader_seed if args.tracking.data_loader_seed is not None else args.tracking.seed
    datasets, weights, dataset_names = sft_mixture.build_mixture(
        tokenizer, args.mixture, max_sequence_length=args.training.max_seq_length, seed=data_seed
    )
    data_loader = MixtureDataLoader(
        datasets,
        weights,
        collator,
        work_dir=args.checkpoint.output_dir,
        global_batch_size=olmo_core_multimodal_utils.global_batch_size_tokens(args.training),
        seed=data_seed,
        pack=args.mixture.pack_sequences,
        pack_max_crops=args.mixture.pack_max_crops if args.mixture.pack_sequences else None,
        est_tokens_per_example=args.mixture.est_tokens_per_example,
        prefetch_workers=args.mixture.prefetch_workers,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        dataset_names=dataset_names,
    )

    run_name = args.tracking.run_name or args.tracking.exp_name
    trainer_config = (
        TrainerConfig(
            save_folder=args.checkpoint.output_dir,
            save_overwrite=True,
            metrics_collect_interval=args.logging.logging_steps,
            cancel_check_interval=5,
            max_duration=Duration.steps(args.training.max_train_steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            olmo_core_utils.build_checkpointer_callback(
                args.checkpoint.checkpointing_steps,
                args.checkpoint.ephemeral_save_interval,
                # Async save with the multimodal train module is unproven; Stage2 uses sync too.
                save_async=False,
                max_checkpoints=args.checkpoint.keep_last_n_checkpoints,
                # HF-init weights are reproducible from the hub, and the step-0 save
                # force-allocates full fp32 Adam states before training — which OOMs
                # the 1-GPU smoke (~34 GB for Molmo2-4B on top of the model copies).
                pre_train_checkpoint=False,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity=args.logging.wandb_entity,
                project=args.logging.wandb_project,
                group=args.logging.wandb_group_name,
                enabled=args.logging.with_tracking,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        # Not OLMo-core's BeakerCallback: that one imports olmo_core.launch.beaker at
        # attach time, whose module top requires beaker-gantry — not shipped in this
        # image. BeakerCallbackV2 is open-instruct's beaker-py 2.x equivalent.
        .with_callback("beaker", olmo_core_callbacks.BeakerCallbackV2(config=dataclasses.asdict(args)))
    )

    # Initial weights / resume (design doc §3.3 step 8):
    # - HF init: weights already loaded in setup_multimodal_model; if_available resumes
    #   a preempted run from save_folder.
    # - Stage-1 init: the trainer loads model weights (only) from load_path, unless a
    #   checkpoint already exists in save_folder (preemption resume).
    if args.checkpoint.resume_from_checkpoint:
        trainer_config.load_path = args.checkpoint.resume_from_checkpoint
    elif defer_load:
        trainer_config.load_path = args.model.model_name_or_path
        trainer_config.load_trainer_state = False
        trainer_config.load_optim_state = False
    else:
        trainer_config.load_strategy = LoadStrategy.if_available

    trainer = trainer_config.build(train_module, data_loader)
    config_saver = trainer.callbacks["config_saver"]
    assert isinstance(config_saver, ConfigSaverCallback)
    config_saver.config = dataclasses.asdict(args)
    trainer.fit()
    teardown_training_environment()


if __name__ == "__main__":
    parser = utils.ArgumentParserPlus(
        (  # ty: ignore[invalid-argument-type]
            olmo_core_utils.ExperimentConfig,
            olmo_core_multimodal_utils.MultimodalModelConfig,
            olmo_core_multimodal_utils.MultimodalTrainingConfig,
            sft_mixture.MixtureConfig,
            olmo_core_utils.LoggingConfig,
            olmo_core_utils.CheckpointConfig,
        )
    )
    parser.set_defaults(
        exp_name="mm_sft", ephemeral_save_interval=_DEFAULT_EPHEMERAL_SAVE_INTERVAL, checkpointing_steps=2000
    )
    tracking, model, training, mixture, logging_cfg, checkpoint = parser.parse()  # ty: ignore[invalid-assignment, not-iterable]
    main(
        MultimodalSFTArguments(
            tracking=tracking,
            model=model,
            training=training,
            mixture=mixture,
            logging=logging_cfg,
            checkpoint=checkpoint,
        )
    )
