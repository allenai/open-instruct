#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "hamishivi/open_instruct_testing_1110"):
    print(f"Using Beaker image: {beaker_image}")

    exp_name = "7b_olmo3_thinker_no_pipeline"

    dataset_config = DatasetConfig(
        dataset_mixer_list=["allenai/Dolci-Think-RL-7B", "1.0"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    eval_dataset_config = DatasetConfig(
        dataset_mixer_list=["allenai/Dolci-Think-RL-7B", "8"],
        dataset_mixer_list_splits=["train"],
        max_prompt_token_length=2048,
    )

    args = Args(
        exp_name=exp_name,
        beta=0.0,
        num_samples_per_prompt_rollout=8,
        num_unique_prompts_rollout=64,
        num_mini_batches=1,
        num_epochs=1,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        output_dir="/output",
        kl_estimator=2,
        max_token_length=10240,
        response_length=32768,
        pack_length=35840,
        non_stop_penalty=False,
        mask_truncated_completions=False,
        temperature=1.0,
        total_episodes=10000000,
        deepspeed_stage=3,
        num_learners_per_node=[8, 8],
        vllm_num_engines=56,
        vllm_tensor_parallel_size=1,
        lr_scheduler_type="constant",
        apply_verifiable_reward=True,
        seed=1,
        local_eval_every=50,
        save_freq=25,
        beaker_eval_freq=50,
        eval_priority="urgent",
        try_launch_beaker_eval_jobs_on_weka=True,
        with_tracking=True,
        llm_judge_model="hosted_vllm/Qwen/Qwen3-32B",
        llm_judge_timeout=600,
        llm_judge_max_tokens=2048,
        llm_judge_max_context_length=32768,
        clip_higher=0.272,
        allow_world_padding=False,
        code_api_url="https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program",
        code_pass_rate_reward_threshold=0.99,
        oe_eval_max_length=32768,
        checkpoint_state_freq=100,
        backend_timeout=1200,
        inflight_updates=False,
        async_steps=1,
        oe_eval_beaker_image="oe-eval-beaker/oe_eval_olmo2_retrofit_auto",
        oe_eval_tasks="mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,simpleqa::tulu-thinker_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,gpqa:0shot_cot::hamish_zs_reasoning_deepseek,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,minerva_math::hamish_zs_reasoning_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,omega_500:0-shot-chat_deepseek,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek",
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name="olmo_thinker",
        ground_truths_key="ground_truth",
        sft_messages_key="messages",
    )

    model_config = ModelConfig(
        model_name_or_path="allenai/Olmo-3-7B-Think-DPO",
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
        cluster=["ai2/jupiter"],
        budget="ai2/oe-adapt",
        image=beaker_image,
        workspace="ai2/olmo-instruct",
        priority="urgent",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=9,
        max_retries=0,
        gpus=8,
        env=[
            {"name": "RAY_CGRAPH_get_timeout", "value": "300"},
            {"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"},
            {"name": "HOSTED_VLLM_API_BASE", "value": "http://ceres-cs-aus-447.reviz.ai2.in:8001/v1"},
        ],
    )

    url = launch_on_beaker(experiment, launch_config)
    print(f"Launched: {url}")


if __name__ == "__main__":
    fire.Fire(main)
