export CUDA_VISIBLE_DEVICES=0

python -m eval.superni.run_eval \
    --data_dir data/eval/superni/splits/default/ \
    --task_dir data/eval/tasks/ \
    --max_num_instances_per_task 1 \
    --max_num_instances_per_eval_task 10 \
    --max_source_length 1024 \
    --max_target_length 256 \
    --num_pos_examples 0 \
    --add_task_definition True \
    --output_dir results/superni/llama-7B-superni-def-only-batch-gen/ \
    --model /net/nfs.cirrascale/allennlp/yizhongw/checkpoints/superni_7B_def_only \
    --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/checkpoints/superni_7B_def_only \
    --eval_batch_size 8

# python -m eval.superni.run_eval \
#     --data_dir data/eval/superni/splits/default/ \
#     --task_dir data/eval/tasks/ \
#     --max_num_instances_per_task 1 \
#     --max_num_instances_per_eval_task 10 \
#     --max_source_length 1024 \
#     --max_target_length 1024 \
#     --num_pos_examples 2 \
#     --add_task_definition True \
#     --output_dir results/superni/llama-7B-superni-def-pos-2/ \
#     --model /net/nfs.cirrascale/allennlp/yizhongw/checkpoints/superni_7B_def_2_pos \
#     --tokenizer /net/nfs.cirrascale/allennlp/yizhongw/checkpoints/superni_7B_def_2_pos \
#     --eval_batch_size 1


