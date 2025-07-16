# IN OLMO-CORE

gantry run \
    --cluster ai2/phobos-cirrascale \
    --allow-dirty --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/jacobm \
    --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
    --weka=oe-training-default:/weka/oe-training-default \
    --gpus 1 \
    -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
        --tokenizer_name_or_path /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft-tokenizer-olmo-chat-template/ \
        --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/usable-tulu-16k/tulu3-olmo2-mix/ \
        --visualize True \
        --chat_template_name olmo \
        --max_seq_length 16384

gantry run \
    --cluster ai2/phobos-cirrascale \
    --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/jacobm \
    --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync" \
    --weka=oe-training-default:/weka/oe-training-default \
    -- /root/.local/bin/uv run python scripts/data/convert_sft_data_for_olmocore.py \
        --dataset_mixer_list allenai/tulu-3-sft-olmo-2-mixture 1.0 \
        --tokenizer_name_or_path /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921-hf \
        --output_dir /weka/oe-training-default/ai2-llm/jacobm/data/sft/test/jacobmorrison/tulu-3-sft-olmo-2-mixture-jacobtest-2-shuffled \
        --chat_template_name jacobtest2



python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo2-7B-sft-8gpu-tulu-3-sft-16k tulu-3-sft-olmo-2 /weka/oe-training-default/ai2-llm/checkpoints/dustins/lc_7b_cont_pretrain_final_anneal/step11921 ai2/titan-cirrascale \
    --trainer.callbacks.wandb.enabled=True \
    --launch.num_gpus=8 \
    --launch.priority=high



MAX_LENGTH=16384
python src/scripts/train/sft/OLMo2-7B-sft.py launch \
    olmo2-7B-original-tulu-3-sft-16k-jacobtest2_16k-packed-2_epoch-5e-5_lr-test-if-explode-16_gpu-seed_13034431 \
        tulu-3-olmo-2-sft-mixture-jacobtest2-shuffled-16k \
        /weka/oe-training-default/ai2-llm/checkpoints/dustins/OLMo-2-1124-7B \
        ai2/titan-cirrascale \
    --trainer.callbacks.wandb.enabled=True \
    --trainer.max_duration.value=2 \
    --train_module.optim.lr=5e-5 \
    --dataset.sequence_length=$MAX_LENGTH \
    --train_module.max_sequence_length=$MAX_LENGTH \
    --launch.num_gpus=8 \
    --launch.num_nodes=2 \
    --launch.priority=urgent \
    --data_loader.seed=13034431

gantry run --cluster ai2/phobos-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/olmo-instruct \
        --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmo2-7B-sft/olmo2-7B-original-tulu-3-sft-16k-jacobtest2_16k-packed-2_epoch-5e-5_lr-test-if-explode-seed_13034431/step5450 \
            -o /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo2-7B-sft/olmo2-7B-original-tulu-3-sft-16k-jacobtest2_16k-packed-2_epoch-5e-5_lr-test-if-explode-seed_13034431/step5450-hf \
            --max-sequence-length 65536

cp /weka/oe-adapt-default/jacobm/rl-sft/checkpoints/olmo-2-tokenizer-jacobtest2-template/* \
    /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo2-7B-sft/olmo2-7B-original-tulu-3-sft-16k-jacobtest2_16k-packed-2_epoch-5e-5_lr-test-if-explode-seed_13034431/step5450-hf
# IN OPEN-INSTRUCT

# NON-REASONING

EXP_NAME=olmo2-7B-original-tulu-3-sft-16k-jacobtest2_16k-packed-2_epoch-5e-5_lr-test-if-explode-seed_13034431
MODEL_PATH=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo2-7B-sft/olmo2-7B-original-tulu-3-sft-16k-jacobtest2_16k-packed-2_epoch-5e-5_lr-test-if-explode-seed_13034431/step5450-hf
WANDB_RUN=placeholder
python scripts/submit_eval_jobs.py \
        --model_name $EXP_NAME \
        --location $MODEL_PATH \
        --cluster ai2/saturn-cirrascale ai2/jupiter-cirrascale-2 \
        --is_tuned \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --run_id $WANDB_RUN \
        --oe_eval_max_length 4096 \
        --workspace tulu-3-results \
        --skip_oi_evals 
        # \
        # --oe_eval_tasks gsm8k::tulu 

# REASONING:
EXP_NAME=olmo2-7B-sft-16k-jacobm-test-non-reasoning-evals
MODEL_PATH=/weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo2-7B-original-sft-tulu-3-sft-4k-jacobtest2-chat-template-shuffled-packed-2_epoch-1e-5_lr/step4960-hf
WANDB_RUN=https://wandb.ai/ai2-llm/tylerr-7B-sft/runs/jbu70nxv

python scripts/submit_eval_jobs.py \
    --model_name $EXP_NAME \
    --location $MODEL_PATH \
    --cluster ai2/saturn-cirrascale ai2/jupiter-cirrascale-2 \
    --is_tuned --workspace "tulu-3-results" --priority high --preemptible \
    --use_hf_tokenizer_template --run_oe_eval_experiments --skip_oi_evals \
    --oe_eval_max_length 32768 \
    --run_id $WANDB_RUN \
    --evaluate_on_weka \
    --oe_eval_tasks minerva_math::hamish_zs_reasoning,gsm8k::zs_cot_latex,minerva_math_500::hamish_zs_reasoning,zebralogic::hamish_zs_reasoning,aime::hamish_zs_reasoning,agi_eval_english:0shot_cot::hamish_zs_reasoning,gpqa:0shot_cot::hamish_zs_reasoning,ifeval::hamish_zs_reasoning,popqa::hamish_zs_reasoning,mmlu:cot::hamish_zs_reasoning,alpaca_eval_v3::hamish_zs_reasoning,bbh:cot::hamish_zs_reasoning,mbppplus:0-shot-chat::tulu-thinker,codex_humanevalplus:0-shot-chat-v1::tulu-thinker \
    --oe_eval_stop_sequences '</answer>' \
    --process_output r1_style


# python scripts/submit_eval_jobs.py \
#     --model_name $EXP_NAME \
#     --location $MODEL_PATH \
#     --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
#     --is_tuned --priority high --preemptible --skip_oi_evals \
#     --use_hf_tokenizer_template --run_oe_eval_experiments --workspace tulu-3-results \
#     --evaluate_on_weka \
#     --run_id $WANDB_RUN \
#     --oe_eval_max_length 32768


python scripts/submit_eval_jobs.py \
    --model_name olmo2-7B-original-sft-tulu-3-sft-4k \
    --location /weka/oe-adapt-default/jacobm/checkpoints/olmo2-7B-sft/olmo2-7B-original-sft-tulu-3-sft-4k/step5444-hf \
    --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
    --is_tuned \
    --priority high \
    --preemptible \
    --use_hf_tokenizer_template \
    --run_oe_eval_experiments \
    --evaluate_on_weka \
    --run_id https://wandb.ai/ai2-llm/jacobm-7B-sft/runs/oxs74w8z \
    --oe_eval_max_length 16384 \
    --workspace tulu-3-results \
    --skip_oi_evals