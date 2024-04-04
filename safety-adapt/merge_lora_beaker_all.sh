
# beaker_ids=(01HS25NJTYV256RQ16ENKPGP39 01HS6M49BVPP3ZAYQMRJH27KTJ 01HS25QXBRZK4S3NEFQKG3JZP9)
# beaker_ids=(01HS25T8ZBV5J8VG8W17GK99Y3 01HS25T8T5ZHM8SXCQFQP80ZB4 01HS25T8G0TMJGCEWC3WCVXXHZ)
beaker_ids=(01HS6Z17ZEET9DQQW96MYH38H1)
e=0
for id in ${beaker_ids[@]}; do
    e=$(( e + 1 ))
    # DESC="Merging_Lora_Checkpoints_tulu2-7b_uncensored_all_safety_adapt_v0_contrastive_${e}epochs"
    DESC="Merging_Lora_Checkpoints_tulu2-7b_uncensored_only_contrastive_${e}epochs"
    mason --cluster ai2/general-cirrascale-a100-80g-ib --budget ai2/oe-adapt --gpus 2 --priority high --workspace ai2/safety-adapt --description $DESC --beaker_datasets /model:$id  --  \
    python open_instruct/merge_lora_v2.py \
        --base_model_name_or_path allenai/tulu-2-7b \
        --lora_model_name_or_path /model \
        --output_dir /output \
        --save_tokenizer
done
