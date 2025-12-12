#!/usr/bin/env python
import fire

from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.grpo_fast import Args, DatasetConfig, ExperimentConfig
from open_instruct.launch import LaunchConfig, launch_on_beaker
from open_instruct.model_utils import ModelConfig


def main(run_local: bool = False, beaker_image: str = "nathanl/open_instruct_auto"):
    print(f"Using Beaker image: {beaker_image}")

    nonreasoner_math_mix_decon = [
        "hamishivi/rlvr_acecoder_filtered_filtered",
        "20000",
        "hamishivi/omega-combined-no-boxed_filtered",
        "20000",
        "hamishivi/rlvr_orz_math_57k_collected_filtered",
        "14000",
        "hamishivi/polaris_53k",
        "14000",
        "hamishivi/MathSub-30K_filtered",
        "9000",
        "hamishivi/DAPO-Math-17k-Processed_filtered",
        "7000",
        "allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered",
        "38000",
        "allenai/rlvr_general_mix-keyword-filtered-topic-chars-char-filt-topic-filtered",
        "50000",
    ]

    eval_dataset_mixer = [
        "hamishivi/omega-combined",
        "4",
        "allenai/IF_multi_constraints_upto5",
        "4",
        "saurabh5/rlvr_acecoder_filtered",
        "4",
        "hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2",
        "4",
        "hamishivi/virtuoussy_multi_subject_rlvr",
        "4",
    ]

    model_name_or_path = "/weka/oe-adapt-default/scottg/olmo/merging/ckpts/olmo3-instruct-dpo-1116-vibes/olmo3-7b-DPO-1115-newb-tpc-d5-lbc100-bal-1e-6-1__42__1763293644"
    chat_template = "olmo123"
    gs_model_name = "olmo3-instruct-dpo-hpz1"
    exp_name = f"grpo_math_only_p64_4_8k_{gs_model_name}"

    dataset_config = DatasetConfig(
        dataset_mixer_list=nonreasoner_math_mix_decon,
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
        num_unique_prompts_rollout=64,
        num_mini_batches=4,
        num_epochs=1,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        kl_estimator=2,
        response_length=8192,
        pack_length=11264,
        stop_strings=["</answer>"],
        non_stop_penalty=False,
        temperature=1.0,
        total_episodes=1024000,
        deepspeed_stage=3,
        num_learners_per_node=[8],
        vllm_num_engines=56,
        lr_scheduler_type="constant",
        apply_verifiable_reward=True,
        seed=1,
        local_eval_every=50,
        save_freq=50,
        checkpoint_state_freq=50,
        beaker_eval_freq=50,
        with_tracking=True,
        vllm_enable_prefix_caching=True,
        clip_higher=0.272,
        mask_truncated_completions=False,
        llm_judge_model="hosted_vllm/Qwen/Qwen3-32B",
        llm_judge_timeout=600,
        llm_judge_max_tokens=2048,
        llm_judge_max_context_length=32768,
        oe_eval_max_length=32768,
        try_launch_beaker_eval_jobs_on_weka=True,
        oe_eval_tasks="gpqa:0shot_cot::qwen3-instruct,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,omega_500:0-shot-chat_deepseek,minerva_math_500::hamish_zs_reasoning_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags_lite,aime:zs_cot_r1::pass_at_32_2024_deepseek,aime:zs_cot_r1::pass_at_32_2025_deepseek,zebralogic::hamish_zs_reasoning_deepseek,bbh:cot::hamish_zs_reasoning_deepseek_v2,mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek",
        eval_priority="urgent",
        code_pass_rate_reward_threshold=0.99,
        inflight_updates=True,
        async_steps=8,
        active_sampling=True,
        advantage_normalization_type="centered",
        no_resampling_pass_rate=0.875,
    )

    tokenizer_config = TokenizerConfig(
        chat_template_name=chat_template,
    )

    model_config = ModelConfig(
        model_name_or_path=model_name_or_path,
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
        task_name=exp_name,
        workspace="ai2/olmo-instruct",
        priority="urgent",
        preemptible=True,
        pure_docker_mode=True,
        num_nodes=8,
        max_retries=5,
        gpus=8,
        gs_model_name=gs_model_name,
        env=[
            {"name": "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "value": "1"},
            {"name": "HOSTED_VLLM_API_BASE", "value": ""},
        ],
    )

    url = launch_on_beaker(experiment, launch_config)
    print(f"Launched: {url}")


if __name__ == "__main__":
    fire.Fire(main)
