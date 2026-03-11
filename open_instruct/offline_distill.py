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
# isort: off
import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed

# isort: on
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

import datasets
import torch
import transformers
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_scheduler,
)
from transformers.training_args import _convert_str_dict

from open_instruct import logger_utils, utils
from open_instruct.collators import DistillationDataCollator
from open_instruct.dataset_transformation import (
    INPUT_IDS_KEY,
    TOKENIZED_DISTILL_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.distillkit import DistillationLossComputer
from open_instruct.model_utils import push_folder_to_hub, save_with_accelerate
from open_instruct.padding_free_collator import TensorDataCollatorWithFlattening
from open_instruct.utils import (
    ArgumentParserPlus,
    clean_last_n_checkpoints,
    get_last_checkpoint_path,
    get_wandb_tags,
    is_beaker_job,
    launch_ai2_evals_on_weka,
    maybe_get_beaker_config,
    maybe_update_beaker_description,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)

logger = get_logger(__name__)


@dataclass
class FlatArguments:
    _VALID_DICT_FIELDS = ["additional_model_arguments"]

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    do_not_randomize_output_dir: bool = False
    model_name_or_path: str | None = field(default=None)
    config_name: str | None = field(default=None)
    use_flash_attn: bool = True
    model_revision: str | None = field(default=None)
    additional_model_arguments: dict | str | None = field(default_factory=dict)
    low_cpu_mem_usage: bool = False
    dataset_name: str | None = field(default=None)
    dataset_mixer: dict | None = field(default=None)
    dataset_mixer_list: list[str] = field(
        default_factory=lambda: ["allenai/tulu-3-sft-personas-algebra", "1.0"]
    )
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: [
            "distill_pretokenized_v1",
            "distill_pretokenized_filter_v1",
        ]
    )
    dataset_target_columns: list[str] = field(
        default_factory=lambda: TOKENIZED_DISTILL_DATASET_KEYS
    )
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: str | None = None
    dataset_skip_cache: bool = False
    dataset_mix_dir: str | None = field(default=None)
    dataset_config_name: str | None = field(default=None)
    max_train_samples: int | None = field(default=None)
    preprocessing_num_workers: int | None = field(default=None)
    max_seq_length: int | None = field(default=None)
    overwrite_cache: bool = False
    clip_grad_norm: float = -1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    logging_steps: int | None = field(default=None)
    lora_rank: int = 64
    lora_alpha: float = 16
    lora_dropout: float = 0.1
    lr_scheduler_type: str = "linear"
    num_train_epochs: int = 2
    output_dir: str = "output/"
    per_device_train_batch_size: int = 8
    use_lora: bool = False
    use_qlora: bool = False
    use_8bit_optimizer: bool = False
    warmup_ratio: float = 0.03
    final_lr_ratio: float | None = field(default=None)
    weight_decay: float = 0.0
    timeout: int = 1800
    resume_from_checkpoint: str | None = field(default=None)
    report_to: str | list[str] = field(default="all")
    save_to_hub: str | None = field(default=None)
    gradient_checkpointing: bool = False
    use_liger_kernel: bool = False
    max_train_steps: int | None = field(default=None)
    seed: int = 42
    checkpointing_steps: str | None = field(default=None)
    keep_last_n_checkpoints: int = 3
    fused_optimizer: bool = True
    load_balancing_loss: bool = False
    load_balancing_weight: float = 0.5
    clean_checkpoints_at_end: bool = True

    with_tracking: bool = False
    wandb_project_name: str = "open_instruct_internal"
    wandb_entity: str | None = None
    push_to_hub: bool = True
    hf_entity: str | None = None
    hf_repo_id: str | None = None
    hf_repo_revision: str | None = None
    hf_repo_url: str | None = None
    try_launch_beaker_eval_jobs: bool = True
    hf_metadata_dataset: str | None = "allenai/tulu-3-evals"
    cache_dataset_only: bool = False

    try_auto_save_to_beaker: bool = True
    gs_bucket_path: str | None = None
    oe_eval_tasks: list[str] | None = None
    oe_eval_max_length: int = 4096
    sync_each_batch: bool = False
    packing: bool = False
    verbose: bool = False

    compression_config: str | None = field(default=None)
    logprob_compressor_config: str | None = field(default=None)
    loss_functions: str | None = field(default=None)

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.dataset_mixer is None
            and self.dataset_mixer_list is None
        ):
            raise ValueError(
                "Need either a dataset name, dataset mixer, or dataset mixer list."
            )
        if (
            (
                self.dataset_name is not None
                and (
                    self.dataset_mixer is not None
                    or self.dataset_mixer_list is not None
                )
            )
            or (self.dataset_name is not None)
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError(
                "Cannot launch Beaker evaluation jobs without pushing to the Hub."
            )
        if self.final_lr_ratio is not None:
            if self.lr_scheduler_type != "linear":
                raise NotImplementedError(
                    "final_lr_ratio only currently implemented for linear schedulers"
                )
            if not (1.0 >= self.final_lr_ratio >= 0.0):
                raise ValueError(
                    f"final_lr_ratio must be between 0 and 1, not {self.final_lr_ratio=}"
                )
        for dict_field in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_field)
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_field, loaded_dict)


def main(args: FlatArguments, tc: TokenizerConfig):
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)

    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=args.sync_each_batch,
        ),
    )

    tc.tokenizer_revision = (
        args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    )
    tc.tokenizer_name_or_path = (
        args.model_name_or_path
        if tc.tokenizer_name_or_path is None
        else tc.tokenizer_name_or_path
    )
    tokenizer = tc.tokenizer

    if not args.do_not_randomize_output_dir:
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logger.info("using the output directory: %s", args.output_dir)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = (
            "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
        )
    beaker_config = None

    if args.push_to_hub and accelerator.is_main_process:
        if args.hf_repo_id is None:
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.exp_name
        args.hf_repo_url = (
            f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"
        )
        if is_beaker_job():
            beaker_config = maybe_get_beaker_config()

    if args.with_tracking:
        experiment_config = vars(args)
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
        if (
            accelerator.is_main_process
            and is_beaker_job()
            and beaker_config is not None
        ):
            experiment_config.update(vars(beaker_config))
        experiment_config.update(vars(tc))
        accelerator.init_trackers(
            args.wandb_project_name,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": args.exp_name,
                    "entity": args.wandb_entity,
                    "tags": [args.exp_name] + get_wandb_tags(),
                }
            },
        )
        wandb_tracker = accelerator.get_tracker("wandb")
        maybe_update_beaker_description(wandb_url=wandb_tracker.run.url)
    else:
        wandb_tracker = None

    if accelerator.is_main_process:
        pprint([args, tc])
    logger_utils.setup_logger()
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_mixer is not None:
        args.dataset_mixer_list = [
            item for pair in args.dataset_mixer.items() for item in pair
        ]
    with accelerator.main_process_first():
        transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
        train_dataset = get_cached_dataset_tulu(
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
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset.set_format(type="pt")
    if accelerator.is_main_process:
        visualize_token(train_dataset[0][INPUT_IDS_KEY], tokenizer)
    if args.cache_dataset_only:
        return

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
            **args.additional_model_arguments,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
            **args.additional_model_arguments,
        )
    else:
        raise ValueError("Instantiating from scratch is not supported.")

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                quantization_config=bnb_config,
                device_map=device_map,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
                if args.use_flash_attn
                else "eager",
            )
        elif args.use_liger_kernel:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM  # noqa: PLC0415

            model = AutoLigerKernelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                attn_implementation="flash_attention_2"
                if args.use_flash_attn
                else "eager",
                fused_linear_cross_entropy=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                revision=args.model_revision,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=tc.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
                if args.use_flash_attn
                else "eager",
            )
    else:
        model = AutoModelForCausalLM.from_config(config)

    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.gradient_checkpointing
            )
        elif args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "o_proj",
                "v_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.packing:
        collate_fn = TensorDataCollatorWithFlattening()
    else:
        collate_fn = DistillationDataCollator(tokenizer=tokenizer, model=model)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )

    if args.loss_functions is None:
        raise ValueError(
            'loss_functions is required, e.g. \'[{"function": "cross_entropy", "weight": 1.0}]\''
        )
    loss_functions_config = (
        json.loads(args.loss_functions)
        if isinstance(args.loss_functions, str)
        else args.loss_functions
    )

    if args.compression_config is not None:
        with open(args.compression_config, encoding="utf-8") as f:
            compressor_config = yaml.safe_load(f)
        if not isinstance(compressor_config, dict):
            raise ValueError(
                f"compression_config must parse to a dict, got {type(compressor_config).__name__}"
            )
    elif args.logprob_compressor_config is None:
        distill_losses_requiring_signal = {
            "kl",
            "jsd",
            "tvd",
            "hinge",
            "logistic_ranking",
        }
        needs_compressor = any(
            cfg.get("function") in distill_losses_requiring_signal
            for cfg in loss_functions_config
        )
        if needs_compressor:
            raise ValueError(
                "logprob_compressor_config is required for distillation losses."
            )
        compressor_config = {}
    elif isinstance(args.logprob_compressor_config, str):
        compressor_config = json.loads(args.logprob_compressor_config)
    else:
        compressor_config = args.logprob_compressor_config
    distill_loss_computer = DistillationLossComputer(
        loss_functions=loss_functions_config,
        compressor_config=compressor_config,
        vocab_size=model.config.vocab_size,
    )

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW  # noqa: PLC0415

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            fused=args.fused_optimizer,
        )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    num_training_steps_for_scheduler = (
        args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes
    )
    num_warmup_steps = int(num_training_steps_for_scheduler * args.warmup_ratio)
    if args.final_lr_ratio is not None and args.lr_scheduler_type == "linear":
        num_training_steps_for_scheduler = (
            num_training_steps_for_scheduler - args.final_lr_ratio * num_warmup_steps
        ) / (1 - args.final_lr_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=num_warmup_steps,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    last_checkpoint_path = get_last_checkpoint_path(args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_batch_idx = 0
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_batch_idx = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_batch_idx // len(train_dataloader)
            completed_steps = resume_batch_idx // args.gradient_accumulation_steps
            resume_batch_idx -= starting_epoch * len(train_dataloader)
    else:
        resume_batch_idx = 0

    progress_bar.update(completed_steps)
    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    local_pred_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    local_total_tokens_this_log_period = torch.tensor(
        0, dtype=torch.int64, device=accelerator.device
    )
    local_pred_tokens_this_log_period = torch.tensor(
        0, dtype=torch.int64, device=accelerator.device
    )
    total_token_including_padding = torch.tensor(
        0, dtype=torch.int64, device=accelerator.device
    )
    start_time = time.perf_counter()
    skipped_batches = False
    fwd_passes_since_last_log = 0

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_loss_components = {}
        if last_checkpoint_path and resume_batch_idx and not skipped_batches:
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_batch_idx
            )
            skipped_batches = True
        else:
            active_dataloader = train_dataloader

        for batch in active_dataloader:
            pred_tokens_in_batch = (batch["labels"] != -100).sum()
            tokens_in_batch = batch["attention_mask"].sum()
            total_token_including_padding += batch["attention_mask"].numel()
            local_total_tokens += tokens_in_batch
            local_total_tokens_this_log_period += tokens_in_batch
            local_pred_tokens += pred_tokens_in_batch
            local_pred_tokens_this_log_period += pred_tokens_in_batch

            with accelerator.accumulate(model):
                model_inputs = {
                    k: v
                    for k, v in batch.items()
                    if k in {"input_ids", "attention_mask", "labels", "position_ids"}
                }
                outputs = model(**model_inputs, use_cache=False)
                loss, loss_dict = distill_loss_computer.compute_loss(
                    student_logits=outputs.logits,
                    model_loss=outputs.loss,
                    labels=batch["labels"],
                    batch=batch,
                )
                del outputs
                total_loss += loss.detach().float()
                for key, value in loss_dict.items():
                    if key != "total_loss":
                        total_loss_components[key] = (
                            total_loss_components.get(key, 0.0) + value
                        )
                fwd_passes_since_last_log += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    sum_loss = accelerator.gather(total_loss).sum().item()
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()
                    total_pred_tokens = (
                        accelerator.gather(local_pred_tokens).sum().item()
                    )
                    total_tokens_including_padding = (
                        accelerator.gather(total_token_including_padding).sum().item()
                    )
                    total_tokens_this_log_period = (
                        accelerator.gather(local_total_tokens_this_log_period)
                        .sum()
                        .item()
                    )
                    local_total_tokens_this_log_period.zero_()
                    accelerator.gather(local_pred_tokens_this_log_period).sum().item()
                    local_pred_tokens_this_log_period.zero_()

                    avg_tokens_per_batch = (
                        total_tokens
                        / accelerator.num_processes
                        / args.per_device_train_batch_size
                        / args.gradient_accumulation_steps
                        / completed_steps
                    )
                    avg_tokens_per_batch_including_padding = (
                        total_tokens_including_padding
                        / accelerator.num_processes
                        / args.per_device_train_batch_size
                        / args.gradient_accumulation_steps
                        / completed_steps
                    )
                    avg_pred_tokens_per_batch = (
                        total_pred_tokens
                        / accelerator.num_processes
                        / args.per_device_train_batch_size
                        / args.gradient_accumulation_steps
                        / completed_steps
                    )
                    metrics_to_log = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "total_tokens": total_tokens,
                        "total_tokens_including_padding": total_tokens_including_padding,
                        "total_pred_tokens": total_pred_tokens,
                        "total_tokens_this_log_period": total_tokens_this_log_period,
                        "avg_tokens_per_batch": avg_tokens_per_batch,
                        "avg_tokens_per_batch_including_padding": avg_tokens_per_batch_including_padding,
                        "avg_pred_tokens_per_batch": avg_pred_tokens_per_batch,
                        "per_device_tps": total_tokens
                        / accelerator.num_processes
                        / (time.perf_counter() - start_time),
                        "per_device_tps_including_padding": total_tokens_including_padding
                        / accelerator.num_processes
                        / (time.perf_counter() - start_time),
                        "reserved_mem_GiB": torch.cuda.max_memory_reserved(
                            device=torch.cuda.current_device()
                        )
                        / 2**30,
                        "allocated_mem_GiB": torch.cuda.max_memory_allocated(
                            device=torch.cuda.current_device()
                        )
                        / 2**30,
                    }
                    total_fwd_passes = (
                        accelerator.gather(
                            torch.tensor(
                                fwd_passes_since_last_log, device=accelerator.device
                            )
                        )
                        .sum()
                        .item()
                    )
                    avg_loss = sum_loss / total_fwd_passes
                    metrics_to_log["train_loss"] = avg_loss
                    for key, value in total_loss_components.items():
                        metrics_to_log[f"train_{key}"] = (
                            value / fwd_passes_since_last_log
                        )
                    if args.with_tracking:
                        accelerator.log(metrics_to_log, step=completed_steps)
                    elif accelerator.is_main_process:
                        # Keep local debugging usable without trackers (e.g., no W&B access).
                        scalar_metrics = {
                            k: v
                            for k, v in metrics_to_log.items()
                            if k.startswith("train_")
                            or k in {"learning_rate", "per_device_tps"}
                        }
                        metric_str = ", ".join(
                            f"{key}={value:.6f}"
                            if isinstance(value, float)
                            else f"{key}={value}"
                            for key, value in scalar_metrics.items()
                        )
                        logger.info(f"step={completed_steps} | {metric_str}")
                    maybe_update_beaker_description(
                        current_step=completed_steps,
                        total_steps=args.max_train_steps,
                        start_time=start_time,
                        wandb_url=wandb_tracker.run.url
                        if wandb_tracker is not None
                        else None,
                    )
                    total_loss = 0
                    total_loss_components = {}
                    fwd_passes_since_last_log = 0

                if (
                    isinstance(checkpointing_steps, int)
                    and completed_steps % checkpointing_steps == 0
                ):
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    with open(
                        os.path.join(
                            get_last_checkpoint_path(args, incomplete=True), "COMPLETED"
                        ),
                        "w",
                    ) as f:
                        f.write("COMPLETED")
                    if accelerator.is_local_main_process:
                        clean_last_n_checkpoints(
                            args.output_dir, args.keep_last_n_checkpoints
                        )
                    accelerator.wait_for_everyone()

                if completed_steps >= args.max_train_steps:
                    break

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            with open(
                os.path.join(
                    get_last_checkpoint_path(args, incomplete=True), "COMPLETED"
                ),
                "w",
            ) as f:
                f.write("COMPLETED")
            if accelerator.is_local_main_process:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            accelerator.wait_for_everyone()

    if args.output_dir is not None:
        save_with_accelerate(
            accelerator,
            model,
            tokenizer,
            args.output_dir,
            args.use_lora,
            chat_template_name=tc.chat_template_name,
        )

    if args.clean_checkpoints_at_end and accelerator.is_local_main_process:
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)

    if (
        args.try_auto_save_to_beaker
        and accelerator.is_main_process
        and is_beaker_job()
        and beaker_config is not None
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if (
        is_beaker_job()
        and accelerator.is_main_process
        and args.try_launch_beaker_eval_jobs
    ):
        launch_ai2_evals_on_weka(
            path=args.output_dir,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=wandb_tracker.run.url if wandb_tracker is not None else None,
            oe_eval_tasks=args.oe_eval_tasks,
            gs_bucket_path=args.gs_bucket_path,
        )
    if args.push_to_hub and accelerator.is_main_process:
        push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    utils.check_oe_eval_internal()
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
