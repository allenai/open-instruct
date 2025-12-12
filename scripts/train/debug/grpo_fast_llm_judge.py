#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.model_utils import ModelConfig


def main():
    dataset_config = DatasetConfig(
        dataset_mixer_list=["hamishivi/virtuoussy_multi_subject_rlvr", "64"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=512,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["hamishivi/virtuoussy_multi_subject_rlvr", "16"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=512,
    )

    args = Args(
        response_length=512,
        pack_length=1024,
        per_device_train_batch_size=1,
        num_unique_prompts_rollout=8,
        num_samples_per_prompt_rollout=4,
        stop_strings=["</answer>"],
        apply_r1_style_format_reward=True,
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
        vllm_sync_backend="gloo",
        vllm_gpu_memory_utilization=0.3,
        save_traces=True,
        vllm_enforce_eager=True,
        single_gpu_mode=True,
        push_to_hub=False,
        llm_judge_model="hosted_vllm/Qwen/Qwen3-32B",
        llm_judge_timeout=600,
        llm_judge_max_tokens=2048,
        llm_judge_max_context_length=32768,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="r1_simple_chat_postpend_think",
        ground_truths_key="ground_truth",
    )

    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
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
