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
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Literal

import datasets
import torch
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.accelerator import GradientAccumulationPlugin
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from huggingface_hub import HfApi
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq, get_scheduler
from transformers.training_args import _convert_str_dict

from open_instruct import logger_utils, utils
from open_instruct.dataset_transformation import (
    INPUT_IDS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
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
    """
    Full arguments class for all fine-tuning jobs.
    """

    # Sometimes users will pass in a `str` repr of a dict in the CLI
    # We need to track what fields those can be. Each time a new arg
    # has a dict type, it must be added to this list.
    # Important: These should be typed with Optional[Union[dict,str,...]]
    # Note: the suggested ellipses typing above causes errors on python 3.10, so they are omitted.
    _VALID_DICT_FIELDS = ["additional_model_arguments"]

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this experiment"""
    do_not_randomize_output_dir: bool = False
    """By default the output directory will be randomized"""
    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    use_flash_attn: bool = field(
        default=True, metadata={"help": "Whether to use flash attention in the model training"}
    )
    model_revision: str | None = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    additional_model_arguments: dict | str | None = field(
        default_factory=dict, metadata={"help": "A dictionary of additional model args used to construct the model."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: dict | None = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: list[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-personas-algebra", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list_splits: list[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: list[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: str | None = None
    """The hash of the dataset configuration."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    dataset_mix_dir: str | None = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_seq_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW optimizer."})
    logging_steps: int | None = field(
        default=None, metadata={"help": "Log the training loss and learning rate every logging_steps steps."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: float = field(default=16, metadata={"help": "The alpha parameter of lora."})
    lora_dropout: float = field(default=0.1, metadata={"help": "The dropout rate of lora modules."})
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        },
    )
    num_train_epochs: int = field(default=2, metadata={"help": "Total number of training epochs to perform."})
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    use_8bit_optimizer: bool = field(
        default=False, metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."}
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    final_lr_ratio: float | None = field(
        default=None,
        metadata={
            "help": "Set the final lr value at the end of training to be final_lr_ratio * learning_rate."
            " Only for linear schedulers, currently."
        },
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    resume_from_checkpoint: str | None = field(
        default=None, metadata={"help": "If the training should continue from a checkpoint folder."}
    )
    report_to: str | list[str] = field(
        default="all",
        metadata={
            "help": "The integration(s) to report results and logs to. "
            "Can be a single string or a list of strings. "
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'. "
            "Specify multiple by listing them: e.g., ['tensorboard', 'wandb']"
        },
    )
    save_to_hub: str | None = field(
        default=None, metadata={"help": "Save the model to the Hub under this name. E.g allenai/your-model"}
    )
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."}
    )
    use_liger_kernel: bool = field(default=False, metadata={"help": "Whether to use LigerKernel for training."})
    max_train_steps: int | None = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: str | None = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."
        },
    )
    keep_last_n_checkpoints: int = field(
        default=3, metadata={"help": "How many checkpoints to keep in the output directory. -1 for all."}
    )
    fused_optimizer: bool = field(default=True, metadata={"help": "Whether to use fused AdamW or not."})
    load_balancing_loss: bool = field(
        default=False, metadata={"help": "Whether to include a load balancing loss (for OLMoE) or not."}
    )
    load_balancing_weight: float = field(
        default=0.5, metadata={"help": "Weight for load balancing loss if applicable."}
    )
    clean_checkpoints_at_end: bool = field(
        default=True, metadata={"help": "Whether to clean up all previous checkpoints at the end of the run."}
    )

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
    try_launch_beaker_eval_jobs: bool = True
    """Whether to launch beaker evaluation jobs after training"""
    hf_metadata_dataset: str | None = "allenai/tulu-3-evals"
    """What dataset to upload the metadata to. If unset, don't upload metadata"""
    cache_dataset_only: bool = False
    """Immediately exit after caching the dataset"""
    add_seed_and_date_to_exp_name: bool = True
    """Append the seed and date to exp_name"""

    # Ai2 specific settings
    try_auto_save_to_beaker: bool = True
    """Whether to try to save the model to Beaker dataset `/output` after training"""
    gs_bucket_path: str | None = None
    """The path to the gs bucket to save the model to"""
    oe_eval_tasks: list[str] | None = None
    """The beaker evaluation tasks to launch"""
    oe_eval_max_length: int = 4096
    """the max generation length for evaluation for oe-eval"""

    sync_each_batch: bool = False
    """Optionaly sync grads every batch when using grad accumulation. Can significantly reduce memory costs."""
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to use packing/padding-free collation via TensorDataCollatorWithFlattening"},
    )
    verbose: bool = field(
        default=False, metadata={"help": "Optionally print additional statistics at each reporting period"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixer is None and self.dataset_mixer_list is None:
            raise ValueError("Need either a dataset name, dataset mixer, or dataset mixer list.")
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
            or (self.dataset_name is not None)
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.try_launch_beaker_eval_jobs and not self.push_to_hub:
            raise ValueError("Cannot launch Beaker evaluation jobs without pushing to the Hub.")
        if self.final_lr_ratio is not None:
            if self.lr_scheduler_type != "linear":
                raise NotImplementedError("final_lr_ratio only currently implemented for linear schedulers")
            if not (1.0 >= self.final_lr_ratio >= 0.0):
                raise ValueError(f"final_lr_ratio must be between 0 and 1, not {self.final_lr_ratio=}")

        # Parse in args that could be `dict` sent in from the CLI as a string
        for dict_feld in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_feld)
            # We only want to do this if the str starts with a bracket to indicate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_feld, loaded_dict)


def main(args: FlatArguments, tc: TokenizerConfig):
    # ------------------------------------------------------------
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir
    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)

    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps, sync_each_batch=args.sync_each_batch
        ),
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

    if args.add_seed_and_date_to_exp_name:
        args.exp_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        args.exp_name = args.exp_name
    if not args.do_not_randomize_output_dir:
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
    logger.info("using the output directory: %s", args.output_dir)
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"
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
        if is_beaker_job():
            beaker_config = maybe_get_beaker_config()

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
        if accelerator.is_main_process and is_beaker_job():
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
        maybe_update_beaker_description(wandb_url=wandb_tracker.run.get_url())
    else:
        wandb_tracker = None  # for later eval launching

    if accelerator.is_main_process:
        pprint([args, tc])

    # Make one log on every process with the configuration for debugging.
    logger_utils.setup_logger()
    logger.info(accelerator.state, main_process_only=False)
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
        args.dataset_mixer_list = [item for pair in args.dataset_mixer.items() for item in pair]
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
            from liger_kernel.transformers import AutoLigerKernelForCausalLM

            logger.info("Attempting to apply liger-kernel. fused_linear_cross_entropy=True")

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
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

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
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # DataLoaders creation:
    if args.packing:
        collate_fn = TensorDataCollatorWithFlattening()
    else:
        collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    accelerator.print("Creating dataloader")
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
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
    if args.use_qlora:
        from bitsandbytes.optim import AdamW

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, fused=args.fused_optimizer)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
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

    num_warmup_steps = int(num_training_steps_for_scheduler * args.warmup_ratio)
    if args.final_lr_ratio is not None and args.lr_scheduler_type == "linear":
        # Correct num_training_steps_for_scheduler to respect final_lr_ratio for a linear scheduler
        num_training_steps_for_scheduler = (
            num_training_steps_for_scheduler - args.final_lr_ratio * num_warmup_steps
        ) / (1 - args.final_lr_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=num_warmup_steps,
    )
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and str(checkpointing_steps).lower() != "epoch":
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    last_checkpoint_path = get_last_checkpoint_path(args)
    if last_checkpoint_path:
        accelerator.print(f"Resumed from checkpoint: {last_checkpoint_path}")
        accelerator.load_state(last_checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        last_checkpoint_path = os.path.basename(last_checkpoint_path)
        training_difference = os.path.splitext(last_checkpoint_path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_batch_idx = 0
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_batch_idx = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_batch_idx // len(train_dataloader)
            completed_steps = resume_batch_idx // args.gradient_accumulation_steps
            resume_batch_idx -= starting_epoch * len(train_dataloader)

    else:
        resume_batch_idx = 0

    resume_step = resume_batch_idx // args.gradient_accumulation_steps

    print(f"Starting {starting_epoch=}, {resume_batch_idx=}, {resume_step=}, {completed_steps=}.")
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    local_total_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    local_pred_tokens = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    local_total_tokens_this_log_period = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    local_pred_tokens_this_log_period = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    total_token_including_padding = torch.tensor(0, dtype=torch.int64, device=accelerator.device)
    start_time = time.perf_counter()
    skipped_batches = False
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_dataloader.set_epoch(epoch)
        total_loss = 0
        total_aux_loss = 0
        if last_checkpoint_path and resume_batch_idx and not skipped_batches:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint.
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_batch_idx)
            # Only perform this skip once
            skipped_batches = True
        else:
            active_dataloader = train_dataloader
        for batch in active_dataloader:
            pred_tokens_in_batch = (batch["labels"] != -100).sum()
            if "attention_mask" in batch:
                tokens_in_batch = batch["attention_mask"].sum()
                total_token_including_padding += batch["attention_mask"].numel()
            elif "position_ids" in batch:
                tokens_in_batch = batch["position_ids"].numel()
                total_token_including_padding += tokens_in_batch
            elif "cu_seq_lens_q" in batch:
                tokens_in_batch = batch["cu_seq_lens_q"][-1]
                total_token_including_padding += tokens_in_batch
            else:
                raise ValueError(f"Expected attention_mask or position_ids or cu_seq_lens_q in batch, found {batch=}")
            local_total_tokens += tokens_in_batch
            local_total_tokens_this_log_period += tokens_in_batch
            local_pred_tokens += pred_tokens_in_batch
            local_pred_tokens_this_log_period += pred_tokens_in_batch

            with accelerator.accumulate(model):
                if args.load_balancing_loss:
                    outputs = model(**batch, use_cache=False, output_router_logits=True)
                    total_aux_loss += outputs.aux_loss.detach().float()
                else:
                    # Standard forward pass
                    outputs = model(**batch, use_cache=False)

                loss = outputs.loss
                del outputs

                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    sum_loss = accelerator.gather(total_loss).sum().item()
                    total_tokens = accelerator.gather(local_total_tokens).sum().item()
                    total_pred_tokens = accelerator.gather(local_pred_tokens).sum().item()
                    total_tokens_including_padding = accelerator.gather(total_token_including_padding).sum().item()
                    total_tokens_this_log_period = accelerator.gather(local_total_tokens_this_log_period).sum().item()
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
                        "reserved_mem_GiB": torch.cuda.max_memory_reserved(device=torch.cuda.current_device()) / 2**30,
                        "allocated_mem_GiB": torch.cuda.max_memory_allocated(device=torch.cuda.current_device())
                        / 2**30,
                    }

                    # [Loss Reporting]
                    #
                    # It is useful to handle loss-reporting for the "mean" and "sum" loss cases
                    # differently.  Cases:
                    #
                    # 1) "mean" loss: `sum_loss` takes individual losses which were *averaged* over
                    #    the toks in their sequence and sums them over all fwd passes in the logging
                    #    period.  We instead want the avg over these passes. Report avg_loss =
                    #    sum_loss / total_fwd_passes, which is roughly independent of global batch
                    #    size.
                    #
                    # 2) "sum" loss: `sum_loss` takes individual losses which were *summed* over the
                    #    toks in their sequence and sums them over all fwd passes in the logging
                    #    period.  We want the avg over each optimizer step (which scales with the
                    #    global batch size), and the average loss per token and per prediction
                    #    token (which are roughly independent of global batch size).
                    total_fwd_passes = (
                        args.logging_steps * args.gradient_accumulation_steps * accelerator.num_processes
                    )
                    avg_loss = sum_loss / total_fwd_passes
                    metrics_to_log["train_loss"] = avg_loss
                    if args.verbose:
                        sec_per_step = (time.perf_counter() - start_time) / (completed_steps - resume_step)
                        steps_remaining = args.max_train_steps - completed_steps
                        secs_remaining = steps_remaining * sec_per_step
                        accelerator.print(
                            f"Approx. time remaining: {timedelta(seconds=secs_remaining)}. {args.max_train_steps=}, {completed_steps=}, {steps_remaining=}"
                        )

                    if args.load_balancing_loss:
                        avg_aux_loss = (
                            accelerator.gather(total_aux_loss).mean().item()
                            / args.gradient_accumulation_steps
                            / args.logging_steps
                        )
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, Aux Loss: {avg_aux_loss}, TPS: {total_tokens / (time.perf_counter() - start_time)}"
                        )
                        metrics_to_log["aux_loss"] = avg_aux_loss
                    else:
                        logger.info(
                            f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}, TPS: {total_tokens / (time.perf_counter() - start_time)}"
                        )
                    if args.verbose:
                        accelerator.print(f"{metrics_to_log=}")
                    if args.with_tracking:
                        accelerator.log(metrics_to_log, step=completed_steps)
                    maybe_update_beaker_description(
                        current_step=completed_steps,
                        total_steps=args.max_train_steps,
                        start_time=start_time,
                        wandb_url=wandb_tracker.run.get_url() if wandb_tracker is not None else None,
                    )
                    total_loss = 0
                    total_aux_loss = 0

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    with open(os.path.join(get_last_checkpoint_path(args, incomplete=True), "COMPLETED"), "w") as f:
                        f.write("COMPLETED")
                    if accelerator.is_local_main_process:
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
            if accelerator.is_local_main_process:
                clean_last_n_checkpoints(args.output_dir, args.keep_last_n_checkpoints)
            accelerator.wait_for_everyone()

    if args.output_dir is not None:
        save_with_accelerate(
            accelerator, model, tokenizer, args.output_dir, args.use_lora, chat_template_name=tc.chat_template_name
        )

    # remove all checkpoints to save space
    if args.clean_checkpoints_at_end and accelerator.is_local_main_process:
        clean_last_n_checkpoints(args.output_dir, keep_last_n_checkpoints=0)

    if (
        args.try_auto_save_to_beaker
        and accelerator.is_main_process
        and is_beaker_job()
        and len(beaker_config.beaker_dataset_id_urls) > 0
        and args.output_dir.rstrip("/") != "/output"
    ):
        shutil.copytree(args.output_dir, "/output", dirs_exist_ok=True)

    if is_beaker_job() and accelerator.is_main_process and args.try_launch_beaker_eval_jobs:
        launch_ai2_evals_on_weka(
            path=args.output_dir,
            leaderboard_name=args.hf_repo_revision,
            oe_eval_max_length=args.oe_eval_max_length,
            wandb_url=wandb_tracker.run.get_url() if wandb_tracker is not None else None,
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
