#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.model_utils import ModelConfig


def main():
    dataset_config = DatasetConfig(
        dataset_mixer_list=["faezeb/tulu_3_rewritten_100k-no-math", "20000"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["hamishivi/tulu_3_rewritten_100k", "32"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        max_token_length=4096,
        response_length=512,
        pack_length=4096,
        per_device_train_batch_size=1,
        num_unique_prompts_rollout=32,
        num_samples_per_prompt_rollout=4,
        stop_strings=["</answer>"],
        kl_estimator=2,
        apply_verifiable_reward=True,
        apply_r1_style_format_reward=True,
        non_stop_penalty=False,
        non_stop_penalty_value=0.0,
        temperature=1.0,
        learning_rate=5e-7,
        lr_scheduler_type="constant",
        total_episodes=2048,
        deepspeed_stage=2,
        num_epochs=1,
        num_learners_per_node=[1],
        vllm_tensor_parallel_size=1,
        beta=0.0,
        seed=3,
        local_eval_every=1,
        vllm_sync_backend="gloo",
        vllm_gpu_memory_utilization=0.5,
        vllm_enforce_eager=True,
        single_gpu_mode=True,
        push_to_hub=False,
        llm_judge_model="hosted_vllm/Qwen/Qwen3-32B",
        llm_judge_timeout=600,
        llm_judge_max_tokens=1024,
        llm_judge_max_context_length=8192,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="tulu_thinker_r1_style",
        ground_truths_key="ground_truth",
    )

    model_config = ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
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
