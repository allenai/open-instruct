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
DPO tuning script. Adapted from our finetuning script.
"""

# isort: off
import contextlib
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
with contextlib.suppress(Exception):
    import deepspeed

# isort: on
import math
import pathlib
import random
import shutil
import time
from datetime import timedelta

import datasets
import torch
import torch.utils
import torch.utils.data
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, get_scheduler

from open_instruct import dpo_utils, logger_utils, model_utils, utils
from open_instruct.dataset_transformation import (
    CHOSEN_INPUT_IDS_KEY,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.padding_free_collator import TensorDataCollatorWithFlatteningDPO
from open_instruct.utils import (
    ArgumentParserPlus,
    ModelDims,
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

logger = logger_utils.setup_logger(__name__)


def build_deepspeed_config(
    zero_stage: int, offload_optimizer: bool = False, offload_param: bool = False, zero_hpz_partition_size: int = 8
) -> dict:
    config = {
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 1e5,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }

    if zero_stage == 3:
        config["zero_optimization"].update(
            {
                "sub_group_size": 1e9,
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
                "zero_hpz_partition_size": zero_hpz_partition_size,
            }
        )

    if offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    if offload_param:
        config["zero_optimization"]["offload_param"] = {"device": "cpu", "pin_memory": True}

    return config


def main(args: dpo_utils.ExperimentConfig, tc: TokenizerConfig):
    # ------------------------------------------------------------
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = "wandb"
        accelerator_log_kwargs["project_dir"] = args.output_dir
    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)
    deepspeed_plugin = None
    if args.zero_stage is not None:
        deepspeed_config = build_deepspeed_config(
            zero_stage=args.zero_stage,
            offload_optimizer=args.offload_optimizer,
            offload_param=args.offload_param,
            zero_hpz_partition_size=args.zero_hpz_partition_size,
        )
        deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=deepspeed_config)

    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps, sync_each_batch=args.sync_each_batch
        ),
        deepspeed_plugin=deepspeed_plugin,
    )

    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if tc.tokenizer_revision != args.model_revision and tc.tokenizer_name_or_path != args.model_name_or_path:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{args.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{args.model_name_or_path=}`."""
        logger.warning(warning)
    tokenizer = tc.tokenizer

    # ------------------------------------------------------------
    # Set up runtime variables
    if not args.do_not_randomize_output_dir:
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logger.info("using the output directory: %s", args.output_dir)
    args.local_cache_dir = os.path.abspath(args.local_cache_dir)
    if is_beaker_job():
        args.local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    beaker_config = None
    if is_beaker_job() and accelerator.is_main_process:
        beaker_config = maybe_get_beaker_config()

    if args.push_to_hub and accelerator.is_main_process:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:
            args.hf_repo_revision = args.exp_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    # ------------------------------------------------------------
    # Initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"]

        # (Optional) Ai2 internal tracking
        if args.wandb_entity is None:
            args.wandb_entity = maybe_use_ai2_wandb_entity()
        if accelerator.is_main_process and beaker_config is not None:
            experiment_config.update(vars(beaker_config))
        experiment_config.update(vars(tc))
        accelerator.init_trackers(
            args.wandb_project,
            experiment_config,
            init_kwargs={
                "wandb": {
                    "name": args.exp_name,
                    "entity": args.wandb_entity,
                    "tags": [args.exp_name] + get_wandb_tags(),
                }
            },
        )

    if args.with_tracking:
        wandb_tracker = accelerator.get_tracker("wandb")
        if accelerator.is_main_process:
            maybe_update_beaker_description(wandb_url=wandb_tracker.run.url)
    else:
        wandb_tracker = None

    if accelerator.is_main_process:
        pprint([args, tc])

    init_gpu_memory = None
    if torch.cuda.is_available():
        init_gpu_memory = torch.cuda.mem_get_info()[0]

    # Make one log on every process with the configuration for debugging.
    logger_utils.setup_logger()
    if accelerator.is_main_process:
        logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    if args.dataset_mixer is not None:
        args.mixer_list = [item for pair in args.dataset_mixer.items() for item in pair]
    with accelerator.main_process_first():
        transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
        train_dataset = get_cached_dataset_tulu(
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
        train_dataset = train_dataset.shuffle(seed=args.seed)
        train_dataset.set_format(type="pt")
    if accelerator.is_main_process:
        visualize_token(train_dataset[0][CHOSEN_INPUT_IDS_KEY], tokenizer)

    if args.cache_dataset_only:
        return

    # Load pretrained model and tokenizer
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
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    def load_model():
        if args.model_name_or_path:
            if args.use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                device_index = accelerator.local_process_index
                device_map = {"": device_index}  # force data-parallel training.
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    revision=args.model_revision,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=tc.trust_remote_code,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                )
            elif args.use_liger_kernel:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM  # noqa: PLC0415

                logger.info("Attempting to apply liger-kernel.")

                # Supported models: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py#L948
                model = AutoLigerKernelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    revision=args.model_revision,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=tc.trust_remote_code,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                    # liger-kernel specific args
                    fused_linear_cross_entropy=False,  # don't fuse the linear layer with CE loss, since we want logits
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
                    attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)
        return model

    model = load_model()
    logger.info("=============model loaded")
    print_gpu_stats(init_gpu_memory)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=args.activation_memory_budget < 1
            )

        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.activation_memory_budget < 1:
        model.gradient_checkpointing_enable()

    model_dims = ModelDims(
        num_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        num_attn_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
    )

    # Capture full dataset size by getting it from the dataset. Sharding happens inside the dataloaders, not the dataset, so we're fine to do this.
    # This is used to allocate tensors for the logprobs cache.
    original_dataset_size = len(train_dataset)
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        logger.info(f"Limiting training samples to {max_train_samples} from {len(train_dataset)}.")
        train_dataset = train_dataset.select(range(max_train_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.packing:
        accelerator.print("Using packing/padding-free collation")
        collate_fn = TensorDataCollatorWithFlatteningDPO(return_position_ids=True, return_flash_attn_kwargs=True)
    else:
        collate_fn = dpo_utils.DataCollatorForSeq2SeqDPO(tokenizer=tokenizer, model=model, padding="longest")

    train_sampler = RandomSampler(train_dataset, generator=torch.Generator().manual_seed(args.seed))
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.use_qlora or args.dpo_use_paged_optimizer:
        from bitsandbytes.optim import AdamW  # noqa: PLC0415

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)
    logger.info("=============optimizer loaded")
    print_gpu_stats(init_gpu_memory)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler
    # for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set.
    # In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set.
    # So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the
    # entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of
    # updates matches the num_training_steps.
    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    logger.info("=============accelerate prepared")
    print_gpu_stats(init_gpu_memory)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    resume_step = None
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    logger.info(f"Starting from epoch {starting_epoch} and step {completed_steps}.")

    logger.info("=============before cache logprobs")
    print_gpu_stats(init_gpu_memory)

    # Cache the logprobs
    if args.loss_type.needs_reference_model:
        ref_cache_hash = dpo_utils.compute_reference_cache_hash(args, tc)
        reference_cache_path = pathlib.Path(dpo_utils.REFERENCE_LOGPROBS_CACHE_PATH) / f"{ref_cache_hash}.pt"
        reference_cache = dpo_utils.build_reference_logprobs_cache(
            model=model,
            dataloader=train_dataloader,
            average_log_prob=args.loss_type.is_average_loss,
            forward_fn=args.forward_fn,
            full_dataset_size=original_dataset_size,
            device=accelerator.device,
            cache_path=reference_cache_path,
            is_main_process=accelerator.is_main_process,
            model_dims=model_dims,
            use_lora=args.use_lora,
            disable_adapter_context=None,
        )
        logger.info("=============after cache logprobs")
        print_gpu_stats(init_gpu_memory)
        torch.cuda.empty_cache()
        logger.info("=============after cache logprobs; clear cache")
        print_gpu_stats(init_gpu_memory)

    # Only show the progress bar once on each machine.
    start_time = time.perf_counter()
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process, bar_format="{l_bar}{bar}{r_bar}\n"
    )
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    local_metrics = utils.MetricsTracker(device=accelerator.device)
    episode = 0
    total_tokens_processed = 0
    mfu_interval_start = time.perf_counter()
    for epoch in range(starting_epoch, args.num_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        if last_checkpoint_path and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        # we need to average the log probs for simpo loss
        for batch in active_dataloader:
            episode += len(batch["chosen_input_ids"]) * accelerator.num_processes
            # dpo forward pass & loss
            with accelerator.accumulate(model):
                policy_chosen_logps, policy_rejected_logps, aux_loss = args.forward_fn(
                    model,
                    batch,
                    average_log_prob=args.loss_type.is_average_loss,
                    output_router_logits=args.load_balancing_loss,
                )  # `aux_loss` is only used when `args.load_balancing_loss = True`

                losses, chosen_rewards, rejected_rewards = dpo_utils.compute_loss(
                    args,
                    batch,
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_cache if args.loss_type.needs_reference_model else None,
                )
                loss = losses.mean()
                if args.load_balancing_loss:
                    weighted_aux_loss = args.load_balancing_weight * aux_loss
                    loss += weighted_aux_loss
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # We keep track of the loss at each logged step
                with torch.no_grad():
                    local_metrics["train_loss"] += loss
                    if args.loss_type.computes_reward_metrics:
                        average_rewards = ((chosen_rewards + rejected_rewards) / 2).mean()
                        accuracy = (chosen_rewards > rejected_rewards).float().mean()
                        margin = (chosen_rewards - rejected_rewards).mean()
                        local_metrics["rewards/chosen"] += chosen_rewards.mean()
                        local_metrics["rewards/rejected"] += rejected_rewards.mean()
                        local_metrics["rewards/average"] += average_rewards
                        local_metrics["rewards/accuracy"] += accuracy
                        local_metrics["rewards/margin"] += margin
                    local_metrics["logps/chosen"] += policy_chosen_logps.mean()
                    local_metrics["logps/rejected"] += policy_rejected_logps.mean()
                    if args.load_balancing_loss:
                        local_metrics["aux_loss"] += weighted_aux_loss

                    chosen_lengths = (batch["chosen_labels"] != -100).sum(dim=1)
                    rejected_lengths = (batch["rejected_labels"] != -100).sum(dim=1)
                    local_metrics["token_count"] += chosen_lengths.sum() + rejected_lengths.sum()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    # single all reduce to save time, avoiding per metric all reduce
                    global_metrics_tensor = accelerator.reduce(local_metrics.metrics, reduction="mean")
                    global_metrics_tensor /= args.gradient_accumulation_steps * args.logging_steps
                    global_metrics_tensor[local_metrics.names2idx["token_count"]] *= (
                        accelerator.num_processes * args.gradient_accumulation_steps * args.logging_steps
                    )
                    global_metrics = {
                        name: global_metrics_tensor[index].item() for name, index in local_metrics.names2idx.items()
                    }

                    mfu_interval_end = time.perf_counter()
                    training_time = mfu_interval_end - mfu_interval_start
                    total_tokens_step = int(global_metrics["token_count"])
                    total_tokens_processed += total_tokens_step
                    avg_sequence_length = total_tokens_step / (
                        args.per_device_train_batch_size
                        * accelerator.num_processes
                        * args.gradient_accumulation_steps
                        * args.logging_steps
                        * 2
                    )

                    step_tokens_per_second = total_tokens_step / training_time
                    total_time_elapsed = time.perf_counter() - start_time
                    total_tokens_per_second = total_tokens_processed / total_time_elapsed

                    metrics_to_log = {
                        "training_step": completed_steps,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": episode / len(train_dataset),
                        "train_loss": global_metrics["train_loss"],
                        "logps/chosen": global_metrics["logps/chosen"],
                        "logps/rejected": global_metrics["logps/rejected"],
                    }
                    if args.loss_type.computes_reward_metrics:
                        metrics_to_log.update(
                            {
                                "rewards/chosen": global_metrics["rewards/chosen"],
                                "rewards/rejected": global_metrics["rewards/rejected"],
                                "rewards/average": global_metrics["rewards/average"],
                                "rewards/accuracy": global_metrics["rewards/accuracy"],
                                "rewards/margin": global_metrics["rewards/margin"],
                            }
                        )
                    logger_str = f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {global_metrics['train_loss']}"
                    if args.load_balancing_loss:
                        logger_str += f" Aux Loss: {global_metrics['aux_loss']}"
                        metrics_to_log["aux_loss"] = global_metrics["aux_loss"]

                    metrics_to_log["perf/mfu_step"] = model_dims.approximate_learner_utilization(
                        total_tokens=total_tokens_step,
                        avg_sequence_length=avg_sequence_length,
                        training_time=training_time,
                        num_training_gpus=accelerator.num_processes,
                    )["mfu"]
                    metrics_to_log["perf/tokens_per_second_step"] = step_tokens_per_second
                    metrics_to_log["perf/tokens_per_second_total"] = total_tokens_per_second

                    logger.info(logger_str)
                    if args.with_tracking:
                        accelerator.log(metrics_to_log, step=completed_steps)
                    if accelerator.is_main_process:
                        maybe_update_beaker_description(
                            current_step=completed_steps,
                            total_steps=args.max_train_steps,
                            start_time=start_time,
                            wandb_url=None if wandb_tracker is None else wandb_tracker.run.url,
                        )
                    # Reset the local metrics
                    local_metrics.metrics.zero_()
                    mfu_interval_start = mfu_interval_end

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
                    with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:
                        f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
                    if accelerator.is_main_process:
                        clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
                    accelerator.wait_for_everyone()

                if completed_steps >= args.max_train_steps:
                    break

        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            # use this to mark the checkpoint as completely saved, to avoid restoring from garbled checkpoints
            with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:
                f.write("COMPLETED")  # annoyingly, empty files arent uploaded by beaker.
            if accelerator.is_main_process:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            accelerator.wait_for_everyone()

    if args.output_dir is not None:
        model_utils.save_with_accelerate(
            accelerator, model, tokenizer, args.output_dir, args.use_lora, chat_template_name=tc.chat_template_name
        )

    if accelerator.is_main_process:
        clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)

    if (
        args.try_auto_save_to_beaker
        and accelerator.is_main_process
        and beaker_config is not None
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if is_beaker_job() and accelerator.is_main_process and args.try_launch_beaker_eval_jobs:
        launch_ai2_evals_on_weka(
            path=args.output_dir,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=wandb_tracker.run.url if args.with_tracking else None,
            oe_eval_tasks=args.oe_eval_tasks,
            gs_bucket_path=args.gs_bucket_path,
            eval_workspace=args.eval_workspace,
            eval_priority=args.eval_priority,
            oe_eval_gpu_multiplier=args.oe_eval_gpu_multiplier,
        )
    if args.push_to_hub and accelerator.is_main_process:
        model_utils.push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


def print_gpu_stats(init_gpu_memory: int | None):
    if torch.cuda.is_available():
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = init_gpu_memory - free_gpu_memory
        logger.info(f"Peak memory usage: {peak_memory / 1024**3:.2f} GB")
        logger.info(f"Total memory usage: {total_gpu_memory / 1024**3:.2f} GB")
        logger.info(f"Free memory: {free_gpu_memory / 1024**3:.2f} GB")


if __name__ == "__main__":
    parser = ArgumentParserPlus((dpo_utils.ExperimentConfig, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
