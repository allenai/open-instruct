python mason.py \
    --cluster ai2/saturn \
    --task_name upload_distill_judge_qwen3-8b_sft \
    --description "Convert and upload checkpoints for distill_judge_qwen3-8b_sft" \
    --workspace ai2/oe-data \
    --priority high \
    --image nathanl/open_instruct_auto \
    --preemptible \
    --num_nodes 1 \
    --budget ai2/oe-base \
    --gpus 1 -- accelerate launch --num_processes 1 \
    scripts/convert_and_upload_checkpoints.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --checkpoints_dir /weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/distill_judge_qwen3-8b_sft__8__1759472419 \
    --hf_repo_id yapeichang/distill_judge_qwen3-8b_sft \
    --trust_remote_code