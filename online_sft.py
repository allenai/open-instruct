from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from open_instruct.model_utils import ModelConfig, CHAT_TEMPLATES
from open_instruct.online_sft_trainer import OnlineSFTConfig, OnlineSFTTrainer


"""
python online_sft.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/minimal/online_sft \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --num_generation_per_prompt 10 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-14m \
    --sft_model_path EleutherAI/pythia-14m \
    --reward_model_path EleutherAI/pythia-14m \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --sanity_check
accelerate launch --config_file configs/ds_configs/deepspeed_zero2.yaml \
accelerate launch --num_processes 8 \
    online_sft.py \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --learning_rate 3e-6 \
    --output_dir models/online_sft_test \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 4 \
    --num_epochs 1 \
    --num_mini_batches 1 \
    --total_episodes 1000000 \
    --model_name_or_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr  \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --save_strategy no \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --push_to_hub
"""
@dataclass
class ScriptArguments:
    dataset_name: str = None
    dataset_text_field: str = "prompt"
    dataset_train_split: str = "train"
    dataset_test_split: Optional[str] = "validation"
    max_length: int = 512


def prepare_dataset(dataset, tokenizer, dataset_text_field):
    """pre-tokenize the dataset before training; only collate during training"""

    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],
            padding=False,
        )
        return {"input_ids": outputs["input_ids"]}

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        batched=True,
        num_proc=4,  # multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, OnlineSFTConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATES["simple_concat_with_space"]
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path)
    policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path)
    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1024))
    train_dataset = raw_datasets[args.dataset_train_split]
    train_dataset = prepare_dataset(train_dataset, tokenizer, args.dataset_text_field)
    eval_dataset = raw_datasets[args.dataset_test_split]
    eval_dataset = prepare_dataset(eval_dataset, tokenizer, args.dataset_text_field)

    ################
    # Training
    ################
    trainer = OnlineSFTTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    if not config.sanity_check:
        trainer.save_model(config.output_dir)
        if config.push_to_hub:
            trainer.push_to_hub()
        trainer.generate_completions()
