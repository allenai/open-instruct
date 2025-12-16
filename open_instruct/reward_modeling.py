import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import gather_object
from huggingface_hub import HfApi
from rich.pretty import pprint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForSequenceClassification, PreTrainedModel, get_scheduler

from open_instruct.dataset_transformation import (
    CHOSEN_INPUT_IDS_KEY,
    REJECTED_INPUT_IDS_KEY,
    TOKENIZED_PREFERENCE_DATASET_KEYS,
    SimplePreferenceCollator,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.model_utils import (
    ModelConfig,
    disable_dropout_in_model,
    get_reward,
    print_rich_single_line_metrics,
    print_rich_table,
    push_folder_to_hub,
    save_with_accelerate,
)
from open_instruct.reward_modeling_eval import evaluate
from open_instruct.utils import (
    ArgumentParserPlus,
    get_wandb_tags,
    is_beaker_job,
    maybe_get_beaker_config,
    maybe_use_ai2_hf_entity,
    maybe_use_ai2_wandb_entity,
)

api = HfApi()


@dataclass
class Args:
    # Dataset
    dataset_mixer_list: list[str] = field(
        default_factory=lambda: ["allenai/tulu-3-wildchat-reused-on-policy-8b", "1.0"]
    )
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_eval_list: list[str] = field(default_factory=lambda: [])
    """A list of datasets (local or HF) to sample from for evaluation."""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_mixer_eval_list_splits: list[str] = field(default_factory=lambda: [])
    """The dataset splits to use for evaluation"""
    dataset_transform_fn: list[str] = field(default_factory=lambda: ["preference_tokenize_v1", "preference_filter_v1"])
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_PREFERENCE_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: str | None = None
    """The hash of the dataset configuration."""
    dataset_config_eval_hash: str | None = None
    """The hash of the dataset configuration for evaluation."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    max_token_length: int = 512
    """The maximum token length to use for the dataset"""
    max_prompt_token_length: int = 256
    """The maximum prompt token length to use for the dataset"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""

    # Experiment
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    seed: int = 1
    """Seed of the experiment"""
    run_name: str | None = None
    """A unique name of this run"""

    # Optimizer
    eps: float = 1e-5
    """The epsilon value for the optimizer"""
    learning_rate: float = 2e-5
    """The initial learning rate for AdamW optimizer."""
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    ] = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    # Batch sizes
    num_train_epochs: int = 1
    """Number of epochs to train"""
    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""
    per_device_train_batch_size: int | None = 1
    """The forward batch size per device (local_micro_batch_size)"""
    per_device_eval_batch_size: int | None = 1
    """The forward batch size per device for evaluation (local_micro_batch_size)"""
    total_episodes: int | None = None
    """The total number of episodes in the dataset"""
    world_size: int | None = None
    """The number of processes (GPUs) to use"""
    micro_batch_size: int | None = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: int | None = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: int | None = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    num_training_steps: int | None = None
    """The number of training_steps to train"""
    num_evals: int = 1
    """The number of evaluations to run throughout training"""
    eval_freq: int | None = None
    """The frequency of evaluation steps"""

    # Experiment tracking
    with_tracking: bool = False
    """If toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "open_instruct_internal"
    """The wandb's project name"""
    wandb_entity: str | None = None
    """The entity (team) of wandb's project"""
    push_to_hub: bool = True
    """Whether to upload the saved model to huggingface"""
    hf_entity: str | None = None
    """The user or org name of the model repository from the Hugging Face Hub"""
    hf_repo_id: str | None = None
    """The id of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_revision: str | None = None
    """The revision of the saved model in the Hugging Face Hub (can be autoset if not given)"""
    hf_repo_url: str | None = None
    """The url of the saved model in the Hugging Face Hub (will be autoset)"""
    output_dir: str = "output"
    """Where to save the model"""

    # Ai2 specific settings
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""


def layer_init(layer: nn.Module, std: float):
    torch.nn.init.normal_(layer.weight, std=std)
    return layer


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    from open_instruct.olmo_adapter import (
        Olmo2Config,
        Olmo2ForSequenceClassification,
        OlmoeConfig,
        OlmoeForSequenceClassification,
    )

    AutoModelForSequenceClassification.register(Olmo2Config, Olmo2ForSequenceClassification)
    AutoModelForSequenceClassification.register(OlmoeConfig, OlmoeForSequenceClassification)

    # ------------------------------------------------------------
    # Setup tokenizer
    tc.tokenizer_revision = model_config.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        model_config.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if (
        tc.tokenizer_revision != model_config.model_revision
        and tc.tokenizer_name_or_path != model_config.model_name_or_path
    ):
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{model_config.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{model_config.model_name_or_path=}`."""
        print(warning)
    tokenizer = tc.tokenizer

    # ------------------------------------------------------------
    # Set up runtime variables
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    args.world_size = accelerator.num_processes
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
    args.batch_size = int(args.local_batch_size * args.world_size)
    args.run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
    if args.push_to_hub:
        if args.hf_repo_id is None:  # auto-generate one
            args.hf_repo_id = "open_instruct_dev"
        if args.hf_entity is None:  # first try to use AI2 entity
            args.hf_entity = maybe_use_ai2_hf_entity()
        if args.hf_entity is None:  # then try to use the user's entity
            args.hf_entity = HfApi().whoami()["name"]
        args.hf_repo_id = f"{args.hf_entity}/{args.hf_repo_id}"
        if args.hf_repo_revision is None:  # auto-generate one
            args.hf_repo_revision = args.run_name
        args.hf_repo_url = f"https://huggingface.co/{args.hf_repo_id}/tree/{args.hf_repo_revision}"

    if args.with_tracking and accelerator.is_main_process and args.wandb_entity is None:
        args.wandb_entity = maybe_use_ai2_wandb_entity()
    local_seed = args.seed + accelerator.process_index

    # ------------------------------------------------------------
    # Setup experiment tracking and seeds
    all_configs = {}
    if is_beaker_job():
        beaker_config = maybe_get_beaker_config()
        # try saving to the beaker `/output`, which will be uploaded to the beaker dataset
        if len(beaker_config.beaker_dataset_id_urls) > 0:
            args.output_dir = "/output"
        all_configs.update(vars(beaker_config))
    all_configs.update(**asdict(args), **asdict(tc), **asdict(model_config))
    if accelerator.is_main_process:
        if args.with_tracking:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=all_configs,
                name=args.run_name,
                save_code=True,
                tags=[args.exp_name] + get_wandb_tags(),
            )
        writer = SummaryWriter(f"runs/{args.run_name}")
        hyperparams_table = "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        writer.add_text("hyperparameters", f"|param|value|\n|-|-|\n{hyperparams_table}")
    device = accelerator.device
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    if accelerator.is_main_process:
        pprint([args, tc, model_config])

    # ------------------------------------------------------------
    # Set up datasets
    transform_fn_args = [
        {},
        {"max_token_length": args.max_token_length, "max_prompt_token_length": args.max_prompt_token_length},
    ]
    with accelerator.main_process_first():
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
        if len(args.dataset_mixer_eval_list) > 0:
            eval_dataset = get_cached_dataset_tulu(
                args.dataset_mixer_eval_list,
                args.dataset_mixer_eval_list_splits,
                tc,
                args.dataset_transform_fn,
                transform_fn_args,
                hf_entity=args.hf_entity,
                dataset_cache_mode=args.dataset_cache_mode,
                dataset_config_hash=args.dataset_config_eval_hash,
                dataset_local_cache_dir=args.dataset_local_cache_dir,
                dataset_skip_cache=args.dataset_skip_cache,
            )
            eval_dataset = eval_dataset.shuffle(seed=args.seed)
    if accelerator.is_main_process:
        visualize_token(train_dataset[0][CHOSEN_INPUT_IDS_KEY], tokenizer)
    if args.cache_dataset_only:
        return

    # ------------------------------------------------------------
    # Runtime setups and quick logging
    if args.total_episodes is None:
        args.total_episodes = args.num_train_epochs * len(train_dataset)
    args.num_training_steps = args.total_episodes // args.batch_size
    args.eval_freq = max(1, args.total_episodes // args.micro_batch_size // args.num_evals)

    # ------------------------------------------------------------
    # Create the model and optimizer
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, revision=model_config.model_revision, num_labels=1
    )
    # resize does its own gather
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    if model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    disable_dropout_in_model(model)  # see p.3. in https://arxiv.org/pdf/1909.08593
    layer_init(
        model.score, std=1 / np.sqrt(model.config.hidden_size + 1)
    )  # see p. 11 in https://arxiv.org/abs/2009.01325
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_steps,
        num_training_steps=args.num_training_steps * args.num_train_epochs,
    )
    data_collator = SimplePreferenceCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=data_collator
    )
    eval_dataloader = None
    if len(args.dataset_mixer_eval_list) > 0:
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=data_collator
        )

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    torch.manual_seed(local_seed)

    # set up the metrics and initial states
    losses = torch.zeros((args.gradient_accumulation_steps,), device=device)
    accuracies = torch.zeros((args.gradient_accumulation_steps,), device=device)
    chosen_rewards = torch.zeros((args.gradient_accumulation_steps,), device=device)
    rejected_rewards = torch.zeros((args.gradient_accumulation_steps,), device=device)
    reward_margin = torch.zeros((args.gradient_accumulation_steps,), device=device)
    local_metrics = torch.zeros((5,), device=device)
    training_step = 0
    gradient_accumulation_idx = 0
    episode = 0
    model.train()

    # training loop
    for _ in range(args.num_train_epochs):
        for data in dataloader:
            episode += args.micro_batch_size
            training_step += 1
            query_responses = torch.cat((data[CHOSEN_INPUT_IDS_KEY], data[REJECTED_INPUT_IDS_KEY]), dim=0)
            with accelerator.accumulate(model):
                _, predicted_reward, _ = get_reward(model, query_responses, tokenizer.pad_token_id, 0)
                chosen_reward = predicted_reward[: data[CHOSEN_INPUT_IDS_KEY].shape[0]]
                rejected_reward = predicted_reward[data[CHOSEN_INPUT_IDS_KEY].shape[0] :]
                accuracy = (chosen_reward > rejected_reward).float().mean()
                loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                losses[gradient_accumulation_idx] = loss
                accuracies[gradient_accumulation_idx] = accuracy
                chosen_rewards[gradient_accumulation_idx] = chosen_reward.mean()
                rejected_rewards[gradient_accumulation_idx] = rejected_reward.mean()
                reward_margin[gradient_accumulation_idx] = (chosen_reward - rejected_reward).mean()
                gradient_accumulation_idx = (gradient_accumulation_idx + 1) % args.gradient_accumulation_steps
                if training_step % args.gradient_accumulation_steps == 0:
                    scheduler.step()
                    local_metrics[0] = accuracies.mean()
                    local_metrics[1] = losses.mean()
                    local_metrics[2] = chosen_rewards.mean()
                    local_metrics[3] = rejected_rewards.mean()
                    local_metrics[4] = reward_margin.mean()
                    global_metrics = accelerator.reduce(local_metrics, reduction="mean").tolist()

                    metrics = {
                        "episode": episode,
                        "epoch": episode / len(train_dataset),
                        "train/rm/accuracy": global_metrics[0],
                        "train/rm/loss": global_metrics[1],
                        "train/rm/chosen_rewards": global_metrics[2],
                        "train/rm/rejected_rewards": global_metrics[3],
                        "train/rm/reward_margin": global_metrics[4],
                        "train/rm/lr": scheduler.get_last_lr()[0],
                    }
                    if accelerator.is_main_process:
                        print_rich_single_line_metrics(metrics)
                        for key, value in metrics.items():
                            writer.add_scalar(key, value, episode)

            # (optionally) evaluate the model
            if (
                args.num_evals > 0
                and training_step > 1
                and training_step % args.eval_freq == 0
                and eval_dataloader is not None
            ):
                eval_metrics, table = evaluate(model, eval_dataloader, tokenizer, max_sampled_texts=10)
                for key in table:
                    table[key] = gather_object(table[key])
                df = pd.DataFrame(table)
                if accelerator.is_main_process:
                    print_rich_single_line_metrics(eval_metrics)
                    for key, value in eval_metrics.items():
                        writer.add_scalar(key, value, episode)
                    if args.with_tracking:
                        wandb.log({"preference_sample_texts": wandb.Table(dataframe=df)})
                    else:
                        print_rich_table(df)

    # save model
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    save_with_accelerate(accelerator, model, tokenizer, args.output_dir, chat_template_name=tc.chat_template_name)
    if args.push_to_hub and accelerator.is_main_process:
        push_folder_to_hub(args.output_dir, args.hf_repo_id, args.hf_repo_revision)


if __name__ == "__main__":
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    main(*parser.parse())
