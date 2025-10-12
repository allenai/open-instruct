target_dir="/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/distill_judge_qwen3-8b_sft_v2_fixed_data__8__1760158162"
dir_to_move="/weka/oe-adapt-default/allennlp/deletable_checkpoint/yapeic/distill_judge_qwen3-8b_sft_v2_fixed_data__8__1760158163"

for epoch_n in $dir_to_move/epoch_*; do
    epoch_base=$(basename "$epoch_n")
    cp -rvx $epoch_n/*.pkl $target_dir/$epoch_base/
    cp -rvx $epoch_n/pytorch_model/* $target_dir/$epoch_base/pytorch_model/
done
