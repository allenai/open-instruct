#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.model_utils import ModelConfig


def main():
    dataset_config = DatasetConfig(
        dataset_mixer_list=["ai2-adapt-dev/rlvr_gsm8k_zs", "64"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=512,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["ai2-adapt-dev/rlvr_gsm8k_zs", "16"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=512,
    )

    args = Args(
        response_length=512,
        pack_length=1024,
        per_device_train_batch_size=1,
        num_unique_prompts_rollout=16,
        num_samples_per_prompt_rollout=4,
        apply_verifiable_reward=True,
        temperature=0.7,
        learning_rate=3e-7,
        total_episodes=200,
        deepspeed_stage=2,
        num_epochs=1,
        num_learners_per_node=[1],
        vllm_tensor_parallel_size=1,
        beta=0.01,
        seed=3,
        local_eval_every=1,
        vllm_sync_backend="nccl",
        vllm_gpu_memory_utilization=0.8,
        save_traces=True,
        vllm_enforce_eager=True,
        tools=["code"],
        code_tool_api_endpoint="http://0.0.0.0:1212/execute",
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="r1_simple_chat_postpend_think_tools",
        ground_truths_key="ground_truth",
    )

    model_config = ModelConfig(
        model_name_or_path="HuggingFaceTB/SmolLM2-135M",
        gradient_checkpointing=True,
    )

    experiment = ExperimentConfig(
        args=args,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        dataset_config=dataset_config,
        eval_dataset_config=eval_dataset_config,
    )

    experiment.run()


if __name__ == "__main__":
    fire.Fire(main)
