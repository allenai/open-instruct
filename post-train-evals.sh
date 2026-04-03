# # # need to set budget correctly
# # # can update to not run aime (?)

# # # MODEL_PATH=/weka/oe-adapt-default/sanjaya/flexolmo/checkpoints/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr/test-if-rlvr-flex-olmo__1__1754720424_checkpoints/step_350
# # # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-hf
# # # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-base-hf
# # # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-anneal-no-expert-bias/step95368-hf
# # # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95250-hf
# # # MODEL_PATH=/weka/oe-training-default/sanjaya/flexolmo/checkpoints/OLMo2-7B-from-posttrained-math-pretrainednonFFN-frozen/step11921-hf
# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step594-hf
# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf
# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed-on-base-no-anneal/step1062-hf
# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-on-base-no-anneal/step594-hf
# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-anneal-frozen-router-mixed-sft/step1062-hf
# # # MODEL_NAME=flexolmo-2x7b-5b-math-anneal-frozen-router-mixed-sft
# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-anneal-NO-frozen-router-mixed-sft/step1062-hf
# # # MODEL_NAME=flexolmo-2x7b-5b-math-anneal-NO-frozen-router-mixed-sft
# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-on-base-no-anneal/step150-hf
# # # MODEL_NAME=flexolmo-2x7b-code-sft-on-base-no-anneal
# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft/step150-hf
# # # MODEL_NAME=flexolmo-2x7b-code-sft



# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-base-no-anneal/step620-hf
# # MODEL_NAME=flexolmo-2x7b-code-sft-mixed-on-base-no-anneal
# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed/step620-hf
# # MODEL_NAME=flexolmo-2x7b-code-sft-mixed

# # MODEL_PATH=/weka/oe-training-default/jacobm/flexolmo/checkpoints/code-anneal-no-expert-bias/step95250-hf
# # MODEL_NAME=flex_olmo_2x7b_code_anneal

# # # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-hf/
# # # MODEL_NAME=FlexOlmo-3x7B-sft-only-test

# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-test-router-hf
# # MODEL_NAME=FlexOlmo-3x7B-sft-only-test-router

# # MODEL_PATH=/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-math_base-code_mixed_no_ann-math_mixed-hf
# # MODEL_NAME=FlexOlmo-3x7B-math_base-code_mixed_no_ann-math_mixed
# # uv run python scripts/submit_eval_jobs.py \
# #     --model_name $MODEL_NAME \
# #     --location $MODEL_PATH \
# #     --cluster ai2/saturn \
# #     --is_tuned \
# #     --workspace ai2/flex2 \
# #     --priority urgent \
# #     --preemptible \
# #     --use_hf_tokenizer_template \
# #     --run_oe_eval_experiments \
# #     --evaluate_on_weka \
# #     --run_id placeholder \
# #     --oe_eval_max_length 4096 \
# #     --process_output r1_style \
# #     --skip_oi_evals \
# #     --beaker_image jacobm/oe-eval-flex-olmo-9-29-5 


# # # 32768

# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_5b-router_sft_all_mixed/step1128-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_20b-router_sft_all_mixed/step1128-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-router_test-math_base-math_5b_sft-code_5b_sft-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-router_test_frozen_router-math_base-math_5b_sft-code_5b_sft-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-NO-frozen-router-mixed-sft-router/step1062-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-math-frozen-router-mixed-sft-router/step1062-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-code-frozen-router-mixed-sft-router/step620-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-3x7B-router_test-math_base-math_50b_sft-code_50b_sft-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert/grpo_math_only_flexolmo-2x7b-math-expert__1__1768451642_checkpoints/step_50/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert/grpo_math_only_flexolmo-2x7b-math-expert__1__1768451642_checkpoints/step_100/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert/grpo_math_only_flexolmo-2x7b-math-expert__1__1768451642_checkpoints/step_150/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert/grpo_math_only_flexolmo-2x7b-math-expert__1__1768451642_checkpoints/step_200/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_50/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_100/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_150/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_200/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_250/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-code-sft-mixed-on-olmo3-code-anneal-no-eb-50B/step620-hf/grpo_code_only_flexolmo-2x7b-code-expert/grpo_code_only_flexolmo-2x7b-code-expert__1__1768452402_checkpoints/step_300/"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-50b_olmo3_code_anneal-tool_use_only/step422-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-tool_use_general_mix/step888-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-tool_use_general_mix/step888-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_anneal-general_sft/step470-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix/step888-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b-olmo3_math-mixed-sft/step1062-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal_mixed_SFT_TEST/step620-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_50"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_100"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_150"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_200"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_250"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_300"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_350"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_400"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_450"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze/grpo_math_only_flexolmo-2x7b-math-expert-no-freeze__1__1770083026_checkpoints/step_500"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_general_only/step394-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_50"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_100"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_150"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_200"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_250"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_300"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_350"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test-high-lr__1__1770186458_checkpoints/step_400"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_50"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_100"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_150"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_200"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_250"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_300"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_350"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test/grpo_math_only_flexolmo-2x7b-math-expert-freeze-test__1__1770173615_checkpoints/step_400"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-4k-test/step888-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_anneal-general-olmo3_math-mix/step966-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_0.25_mix/step536-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_math_code_mix/step1224-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3-reasoning_sft_0.75/step842-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-reasoning_anneal-general-olmo3_reasoning-mix/step784-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-unf-lm-head/step888-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_code_anneal-olmo3_code-general-mix-unf-lm-head/step782-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed/step888-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head__1__1771484873_checkpoints/step_500"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code/step1128-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-unf-lm-head-embed/step1128-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed-1-active/step888-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-5b_code_1a-mix_sft/step782-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-unf-rt-4-domain/step1128-hf"
# #     "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/router-sft-only-newcode-sft-experts/step1212-hf"
# #     "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/router-sft-newcode-pretrained/step1212-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3-reasoning_sft_0.75/step842-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.05/step56-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.1/step112-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.25/step280-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.5/step562-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.75/step842-hf"
# # "/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/general-model-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head__1__1771484873_checkpoints/step_500"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_code_anneal-olmo3_code-general-mix-unf-lm-head/step782-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool-mix-unf-lm-head-embed-1-active/step888-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_all_mixed/step1128-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code/step1128-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-unf-lm-head-embed/step1128-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-unf-rt-4-domain/step1128-hf"
# # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_1.0/step1128-hf"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf__1__1771994781_checkpoints/step_10"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf__1__1771994781_checkpoints/step_20"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf__1__1771994781_checkpoints/step_30"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf__1__1771994781_checkpoints/step_50"
# #     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.50-redux/step562-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-reasoning_anneal-FIXED-general-olmo3_reasoning-mix/step784-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-20b_olmo3_math_anneal-math-mixed-sft/step1062-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-olmo3_code_50b_sft-router_sft_0.50-old-seeds/step562-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_50"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_100"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_150"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_200"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_250"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_300"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_350"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_400"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_450"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_500"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_rl-olmo3_code-tool-rt-4-domain/step1128-hf/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer/grpo_math_only_flex-4x7b-4-domain-RLRT-6e-7-unf-longer__1__1772074378_checkpoints/step_550"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_50"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_100"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_150"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_200"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_250"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_300"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_350"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_400"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_450"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_500"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_550"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_600"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_650"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_700"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_750"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_800"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_850"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_900"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-math_sft-olmo3_code-tool-fixed_rt_sft/step1534-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-tool_use-router_sft-lr_1e-3/step1534-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-tool_use-router_sft-lr_5e-3/step1534-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-tool_use-router_sft-lr_1e-3-old_seeds/step1534-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-DPO-math-1e-6"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-DPO-code-1e-6/flexolmo-2x7b-DPO-code-1e-6/"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-DPO-math-and-general-1e-6/"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-DPO-code-and-general-1e-6/"


#     "/weka/oe-training-default/sanjaya/flexolmo/checkpoints/flexolmo-4x7b-tool_use-router_sft-lr_5e-5/step1534-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_sft-olmo3_code-tool_use-NO_RT-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.1-1e-4/step152-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.05-1e-4/step76-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-full-1e-3/step1534-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-full-1e-4/step1534-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.25-1e-4/step382-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.5-1e-4/step766-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-1.0-5e-4/step1534-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-1.0-5e-5/step1534-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-test-fixed-rt-0.75-1e-4/step1150-hf"
#     "/weka/oe-adapt-default/sanjaya/flexolmo/checkpoints/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr/test-if-rlvr-flex-olmo__1__1754720424_checkpoints/step_350"
#     "/weka/oe-training-default/jacobm/flexolmo/checkpoints/math-base-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-no_anneal-tool_use_general_mix-unf-lm-head/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math_base-olmo3_safety/step62-hf"

#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-0.25-mix-tool-use-1.0/step686-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-0.25-mix-code-1.0/step494-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-0.25-mix-general-1.0/step680-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-0.25-mix-math-1.0/step822-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-code-only/step148-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-general-only/step398-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-math-only/step588-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/flex2-7B-sft/flexolmo-4x7b-router_sft-ablation-tool-use-only/step406-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router__1__1772922838_checkpoints/step_50"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router__1__1772922838_checkpoints/step_100"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router__1__1772922838_checkpoints/step_150"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router__1__1772922838_checkpoints/step_200"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7-froz-router__1__1772922838_checkpoints/step_250"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_base-math_rl-olmo3_code-tool_use-average_all-no_rt-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-fixed-rt-merge-all-0.05-4dom-1e-4/step76-hf"
    # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7b-fixed-rt-merge-all-0.25-4dom-1e-4/step382-hf"
    # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head/grpo_math_only_flex-2x7b-math_rl_froz-6e-7-unf-lm-head__1__1771484873_checkpoints/step_500"
    # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-4x7B-math_rl_x4-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-0.05-1e-4/step80-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-0.25-1e-4/step400-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-50b_math2_anneal-olmo3_math_mix-attm2/step966-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-1.0-1e-4/step1602-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-0.05-1e-4/step80-hf/grpo_math_only_flex-4x7b-olm3_doms-0.05-RLRT-6e-7-unf-all/grpo_math_only_flex-4x7b-olm3_doms-0.05-RLRT-6e-7-unf-all__1__1772996052_checkpoints/step_100"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-50b_math2_anneal-olmo3_math_mix-unf-lm-emb/step966-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-olmo3_3x_domain-0.01-1e-4/step16-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-olmo3_50b_code_anneal-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7/grpo_code_only_flex-2x7b-olmo3_code_sft-6e-7__1__1772261343_checkpoints/step_200"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-merge-all-math_rl-code_rl-tool_use_sft-0.05-1e-4/step80-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4/step66-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-kevin_med_anneal-10b-general-olmo3_science-biomed-mix/step694-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-olmo2_code_sft-tool_use_sft-safety_sft-0.05-1e-4/step60-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-20b_olmo3_math_anneal-math-mixed-sft/step1062-hf/grpo_math_only_flex-2x7b-20b_ol3_ann-ol2_sft_math-6e-7-unf/grpo_math_only_flex-2x7b-20b_ol3_ann-ol2_sft_math-6e-7-unf__1__1773120635_checkpoints/step_100"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-20b_olmo3_math_anneal-math-mixed-sft/step1062-hf/grpo_math_only_flex-2x7b-20b_ol3_ann-ol2_sft_math-6e-7-unf/grpo_math_only_flex-2x7b-20b_ol3_ann-ol2_sft_math-6e-7-unf__1__1773120635_checkpoints/step_150"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-20b_olmo3_math_anneal-math-mixed-sft/step1062-hf/grpo_math_only_flex-2x7b-20b_ol3_ann-ol2_sft_math-6e-7-unf/grpo_math_only_flex-2x7b-20b_ol3_ann-ol2_sft_math-6e-7-unf__1__1773120635_checkpoints/step_200"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-20b_olmo3_math_anneal-math-mixed-sft/step1062-hf/grpo_math_only_flex-2x7b-20b_ol3_ann-ol2_sft_math-6e-7-unf/grpo_math_only_flex-2x7b-20b_ol3_ann-ol2_sft_math-6e-7-unf__1__1773120635_checkpoints/step_250"
# # "/weka/oe-training-default/jacobm/flexolmo/checkpoints/olmo3-code-anneal-50B/step95368-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_all-0.05-1e-4/step66-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_4-math_rl-0.05-1e-4/step66-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_4-code_rl-0.05-1e-4/step66-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_3-math_code_rl-0.05-1e-4/step66-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_3-olmo2_code-olmo3_math-0.05-1e-4/step60-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-olmo3_sft_3-olmo3_code_rl-olmo2_math-0.05-1e-4/step70-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4/step66-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-test/step594-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-math-sft-mixed/step1062-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-olmo2_code_sft-tool_use_sft-safety_sft-0.05-1e-4/step60-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-1-active/step66-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-2-active/step66-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-3-active/step66-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-1e-4-4-active/step66-hf"
# "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_reasoning-mix/step784-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_math_code_tool_use_safety-1.0/step1764-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_tool_use-mix/step888-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_safety-mix/step534-hf"
#     "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/olmo2-7B-sft/math_expert_sft_mixed/step1062-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-DPO-math-1e-6-unfrozen"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-DPO-math-and-general-1e-6-unfrozen"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-DPO-code-1e-6-unfrozen"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-2x7b-DPO-code-and-general-1e-6-unfrozen"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_50/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_100/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_150/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_200/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_250/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_300/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_350/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_400/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_450/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_500/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_550/"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flex-base-7b-DPO-olmo2-1e-6/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7/grpo_math_only_flex-base-7b-mixed-all-sft-6e-7__1__1773965057_checkpoints/step_600/"
"/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/7b-merge-model-baseline-5-domains-2"
    # "/weka/oe-adapt-default/sanjaya/flexolmo/checkpoints/olmo2_flex_base-tulu3-no_code-no_math-dpo-rlvr/test-if-rlvr-flex-olmo__1__1754720424_checkpoints/step_350",
    # "/weka/oe-training-default/ai2-llm/checkpoints/sanjaya/olmo2-7B-sft/math_expert_sft_mixed/step1062-hf/grpo_math_only_flex-base-7b-final-6e-7/grpo_math_only_flex-base-7b-final-6e-7__1__1773891947_checkpoints/step_400",
    # "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-50b_ol3_code_ann-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-base-7b-ol3_code-6e-7-unf/grpo_code_only_flex-base-7b-ol3_code-6e-7-unf__1__1773949222_checkpoints/step_100",
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_tool_use-mix/step888-hf",
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_safety-mix/step534-hf",
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-4-active/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-3-active/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-2-active/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.05-5e-4-1-active/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-final-sft-only-0.05-1e-4/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.25-1e-4-4-active/step332-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-0.25-1e-4-3-active/step332-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-1.0-1e-4-4-active/step1328-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-5x7B-math_rl-code_rl-tool_use_sft-safety_sft-1.0-1e-4-3-active/step1328-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-0.05-1e-4/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-4x7B-math_rl-code_rl-tool_use_sft-0.05-1e-4/step62-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first-0.05-1e-4/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/flexolmo-3x7B-math_rl-code_rl-0.05-1e-4/step78-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-1.0-1e-4/step1328-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo2_math-mix/step1062-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_code-mix/step782-hf"
    "/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/7b-merge-model-baseline-5-domains-sft-only"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo2_math-mix/step1062-hf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf__1__1774160548_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo2_math-mix/step1062-hf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf__1__1774160548_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo2_math-mix/step1062-hf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf__1__1774160548_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo2_math-mix/step1062-hf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf__1__1774160548_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo2_math-mix/step1062-hf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf__1__1774160548_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo2_math-mix/step1062-hf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf/grpo_math_only_flex-base-7b-ol2_math-no-anneal-6e-7-unf__1__1774160548_checkpoints/step_300"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf__1__1774160535_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf__1__1774160535_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf__1__1774160535_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf__1__1774160535_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf__1__1774160535_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-BASE-general-olmo3_code-mix/step782-hf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf/grpo_code_only_flex-base-7b-ol3_code-no-anneal-6e-7-unf__1__1774160535_checkpoints/step_300"
"/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/7b-merge-model-baseline-5-domains-test-mergekit"
    "/weka/oe-adapt-default/jacobm/flexolmo/checkpoints/7b-merge-model-baseline-5-domains-test-manual-latest"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4-4-active/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4-3-active/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4-2-active/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/BTX-5x7B-Test-5-Domains-tool-first-FIXED-0.05-1e-4-1-active/step66-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-CONTINUED-mixed_all_sft-fixed/step1856-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-math_sft-0.05-1e-4/step66-hf"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-CONTINUED-mixed_all_sft-fixed/step1856-hf/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7__1__1774407287_checkpoints/step_50"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-CONTINUED-mixed_all_sft-fixed/step1856-hf/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7__1__1774407287_checkpoints/step_100"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-CONTINUED-mixed_all_sft-fixed/step1856-hf/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7__1__1774407287_checkpoints/step_150"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-CONTINUED-mixed_all_sft-fixed/step1856-hf/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7__1__1774407287_checkpoints/step_200"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-CONTINUED-mixed_all_sft-fixed/step1856-hf/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7__1__1774407287_checkpoints/step_250"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-CONTINUED-mixed_all_sft-fixed/step1856-hf/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7__1__1774407287_checkpoints/step_300"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7b-CONTINUED-mixed_all_sft-fixed/step1856-hf/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7/grpo_math_only_olmo2-7b-CONTINUED-mixed_all_sft-6e-7__1__1774407287_checkpoints/step_350"
"/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-math_sft-FINAL-0.05-1e-4/step70-hf"
MODEL_PATHS=(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-0.01-1e-4/step12-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-0.1-1e-4/step132-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-0.25-1e-4/step332-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-0.5-1e-4/step664-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-0.75-1e-4/step996-hf"
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/flex2-7B-sft/FlexOlmo-5x7B-final-1.0-1e-4/step1328-hf"
)
all_but_safety="mmlu:cot::hamish_zs_reasoning_deepseek,popqa::hamish_zs_reasoning_deepseek,simpleqa::tulu-thinker_deepseek,bbh:cot::hamish_zs_reasoning,gpqa:0shot_cot::qwen3-instruct,zebralogic::hamish_zs_reasoning_deepseek,agi_eval_english:0shot_cot::hamish_zs_reasoning_deepseek,gsm8k::zs_cot_latex_deepseek,omega_500:0-shot-chat_deepseek,codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek,bfcl_all::std"
# coding_tasks="codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,livecodebench_codegeneration::tulu-thinker_deepseek_no_think_tags"
# all_but_safety="codex_humanevalplus:0-shot-chat::tulu-thinker_deepseek,mbppplus:0-shot-chat::tulu-thinker_deepseek,alpaca_eval_v3::hamish_zs_reasoning_deepseek,ifeval::hamish_zs_reasoning_deepseek"

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")
    
    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi

    # MODEL_NAME=$MODEL_NAME-2
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks $all_but_safety \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5
done

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")
    
    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi
    # MODEL_NAME=$MODEL_NAME-2
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks "minerva_math::hamish_zs_reasoning_deepseek" \
        --gpu_multiplier 2 \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5
done


safety_tasks="harmbench::default,do_anything_now::default,wildguardtest::default,wildjailbreak::benign,trustllm_jailbreaktrigger::default"
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")
    
    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi
    # MODEL_NAME=$MODEL_NAME-2
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks $safety_tasks \
        --beaker_image maliam/flexolmo-libraries-safety \
        --gpu_multiplier 2
done

if_ood="ifeval_ood::tulu-thinker"
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    BASENAME=$(basename "$MODEL_PATH")
    
    if [[ "$BASENAME" =~ ^step_[0-9]+$ ]]; then
        # RL checkpoint case: extract experiment name and step
        STEP_NUM="$BASENAME"
        CHECKPOINTS_DIR=$(dirname "$MODEL_PATH")
        EXPERIMENT_DIR=$(dirname "$CHECKPOINTS_DIR")
        EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")
        MODEL_NAME="${EXPERIMENT_NAME}_${STEP_NUM}"
    elif [[ "$BASENAME" =~ ^step[0-9]+-hf$ ]]; then
        # SFT checkpoint with step number
        MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
    else
        # Direct model path (no step directory)
        MODEL_NAME=$(echo "$BASENAME" | sed 's/-hf$//')
    fi
    # MODEL_NAME=$MODEL_NAME-2
    
    echo "Submitting eval for: $MODEL_NAME"
    uv run python scripts/submit_eval_jobs.py \
        --model_name "${MODEL_NAME}" \
        --location "$MODEL_PATH" \
        --cluster ai2/saturn ai2/ceres \
        --is_tuned \
        --workspace ai2/flex2 \
        --priority urgent \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id placeholder \
        --oe_eval_max_length 4096 \
        --process_output r1_style \
        --skip_oi_evals \
        --oe_eval_tasks $if_ood \
        --beaker_image jacobm/oe-eval-flex-olmo-9-29-5 
done