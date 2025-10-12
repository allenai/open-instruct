# python mason.py \
#     --cluster ai2/jupiter \
#     --task_name upload_distill_judge_qwen3-8b_sft_v2 \
#     --description "Convert and upload checkpoints for distill_judge_qwen3-8b_sft" \
#     --workspace ai2/oe-data \
#     --priority normal \
#     --image nathanl/open_instruct_auto \
#     --preemptible \
#     --num_nodes 1 \
#     --budget ai2/oe-base \
#     --gpus 1 -- accelerate launch --num_processes 1 \
#     scripts/convert_and_upload_checkpoints.py \
#     --model_name_or_path Qwen/Qwen3-8B \
#     --checkpoints_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/distill_judge_qwen3-8b_sft_v2__8__1759945132 \
#     --hf_repo_id yapeichang/distill_judge_qwen3-8b_sft_v2 \
#     --trust_remote_code

python mason.py \
    --cluster ai2/jupiter \
    --task_name upload_distill_judge_qwen3-8b_sft_v2_fixed_data \
    --description "Convert and upload checkpoints for distill_judge_qwen3-8b_sft_v2_fixed_data" \
    --workspace ai2/oe-data \
    --priority high \
    --image nathanl/open_instruct_auto \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-base \
    --gpus 1 -- accelerate launch --num_processes 1 \
    scripts/convert_and_upload_checkpoints.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --checkpoints_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/distill_judge_qwen3-8b_sft_v2_fixed_data__8__1760158162 \
    --hf_repo_id yapeichang/distill_judge_qwen3-8b_sft_v2_fixed_data \
    --upload_dtype bf16 \
    --max_shard_size 5GB \
    --save_format safetensors \
    --trust_remote_code