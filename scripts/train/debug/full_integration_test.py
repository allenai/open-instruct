#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "open-instruct-integration-test"):
    print(f"Using Beaker image: {beaker_image}")

    split_int_mix_3 = [
        "hamishivi/omega-combined",
        "63033",
        "allenai/IF_multi_constraints_upto5",
        "63033",
        "saurabh5/rlvr_acecoder_filtered",
        "63033",
        "hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2",
        "63033",
    ]

    eval_dataset_mixer = [
        "hamishivi/omega-combined",
        "8",
        "allenai/IF_multi_constraints_upto5",
        "8",
        "saurabh5/rlvr_acecoder_filtered",
        "8",
        "hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2",
        "4",
        "hamishivi/virtuoussy_multi_subject_rlvr",
        "4",
    ]

    exp_name = "2507rl_qwen2ot2_sft_mix_split_int_mix_3"

    dataset_config = DatasetConfig(
        dataset_mixer_list=split_int_mix_3,
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=eval_dataset_mixer,
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name=exp_name,
        beta=0.0,
        num_samples_per_prompt_rollout=8,
        num_unique_prompts_rollout=32,
        num_mini_batches=4,
        num_epochs=1,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        output_dir="/output",
        kl_estimator=2,
        response_length=8192,
        pack_length=12384,
        stop_strings=["</answer>"],
        non_stop_penalty=False,
        temperature=1.0,
        total_episodes=10000,
        deepspeed_stage=3,
        num_learners_per_node=[8],
        vllm_num_engines=8,
        vllm_tensor_parallel_size=1,
        lr_scheduler_type="constant",
        apply_verifiable_reward=True,
        seed=1,
        local_eval_every=100,
        save_freq=100,
        eval_priority="high",
        try_launch_beaker_eval_jobs_on_weka=True,
        with_tracking=True,
        vllm_enable_prefix_caching=True,
        llm_judge_model="hosted_vllm/Qwen/Qwen3-32B",
        llm_judge_timeout=600,
        llm_judge_max_tokens=2048,
        llm_judge_max_context_length=131072,
        clip_higher=0.272,
        allow_world_padding=False,
        oe_eval_max_length=32768,
        oe_eval_tasks="minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker",
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="tulu_thinker",
        ground_truths_key="ground_truth",
        sft_messages_key="messages",
    )

    model_config = ModelConfig(
        model_name_or_path="ai2-adapt-dev/tulu_3_long_finetune_qwen_7b_reg",
        gradient_checkpointing=True,
    )

    experiment = ExperimentConfig(
        args=args,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        dataset_config=dataset_config,
        eval_dataset_config=eval_dataset_config,
    )

    if run_local:
        return experiment.run()

    launch_config = LaunchConfig(
        cluster=["ai2/augusta"],
        budget="ai2/oe-adapt",
        image=beaker_image,
        description=exp_name,
        workspace="ai2/olmo-instruct",
        priority="high",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=2,
        max_retries=0,
        gpus=8,
        env=[
            {"name": "VLLM_DISABLE_COMPILE_CACHE", "value": "1"},
            {"name": "HOSTED_VLLM_API_BASE", "value": "http://saturn-cs-aus-253.reviz.ai2.in:8001/v1"},
            {"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"},
            {"name": "LITELLM_LOG", "value": "ERROR"},
        ],
    )

    launch_on_beaker(experiment, launch_config)


if __name__ == "__main__":
    fire.Fire(main)
