

# declare -a beaker_ids=(
#     "01HVMQRWXY34SEGYWTZJ3NSSQD refusal_lora_tunning_tulu-2-7b_tulu_match_refusal_adapt_v0.1_filtered_epoch1"
#     "01HVMQST443QZAX6PHKQBCN1A4 refusal_lora_tunning_tulu-2-7b_tulu_none_refusal_adapt_v0.1_filtered_epoch1"
#     "01HVMQTQDJSJ29ME9FY1081DJ8 refusal_lora_tunning_tulu-2-7b_refusal_adapt_v0.1_filtered_bottom25_epoch1"
#     "01HVMQWVST4ZGBDXQPRXB74NK0 refusal_lora_tunning_tulu-2-7b_refusal_adapt_v0.1_filtered_top25_epoch1"
#     "01HVMQXFAJC01YAH5YGBSKYB79 refusal_lora_tunning_tulu-2-7b_refusal_adapt_v0.1_filtered_contrastive_safe_epoch1"
#     "01HVMQY2VGZDP63M374T1K2RY6 refusal_lora_tunning_tulu-2-7b_refusal_adapt_v0.1_filtered_contrastive_refusal_epoch1"
#     "01HVMQYCKV5EP48W8B5S5VXG0V refusal_lora_tunning_tulu-2-7b_refusal_adapt_v0.1_filtered_contrastive_mix_safety_refusal_epoch1"
#     "01HVMQYPCYHY6ZGFP9DG7Q8P1M refusal_lora_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_filtered_contrastive_mix_safety_refusal_epoch1"
#     "01HVMQZ05M1WFCJEAXX0Y40BPQ refusal_lora_tunning_tulu-2-7b_uncensored_tulu_match_refusal_adapt_v0.1_filtered_epoch1"
#     "01HVMR1E7K2B5663Y46156GBGK refusal_lora_tunning_tulu-2-7b_uncensored_tulu_none_refusal_adapt_v0.1_filtered_epoch1"
#     "01HVMR1EDQ4BP17MHV2SZHB9JW refusal_lora_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_filtered_top25_epoch1"
#     "01HVMR1R1ATWQZ2WX9T4AQQCD3 refusal_lora_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_filtered_bottom25_epoch1"
#     "01HVMR21S2ZX7V3YGFY18YR4B6 refusal_lora_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_filtered_contrastive_safety_epoch1"
#     "01HVMR2BGV82729MM3KV12MYZA refusal_lora_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_filtered_contrastive_refusal_epoch1"
# )
declare -a beaker_ids=(
    # "01HVSE2M2FANR8YC071N7CY445 lora_tulu_none-safety_v2_10_epoch2"
    # "01HVSE37K9EWVW733J9VF3P3NF lora_tulu_none-safety_v2_20_epoch2"
    # "01HVSE3HC8H2MECR0FXJJ9BY35 lora_tulu_none-safety_v2_60_epoch2"
    # "01HVSE3V40X1BF9765XPHGPFCE lora_tulu_none-safety_v2_100_epoch2"
)

declare -a beaker_ids=(
    "01HVY7EGP1MAVSCYSY0F14ZPNQ refusal_lora_tunning_tulu-2-7b_refusal_adapt_v0.1_filtered_remove_amb_contrastive_mix_epoch1_"
    "01HVY7FQS1MXBPXNG4WS2N3CPG refusal_lora_tunning_tulu-2-7b_refusal_adapt_v0.1_filtered_remove_amb_contrastive_safety_epoch1_"
    "01HVY7DKE7MTFZAW76T06HVFZE refusal_lora_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_filtered_remove_amb_contrastive_mix_epoch1_"
    "01HVY7GB9C8WV9SPJ58XF714BA refusal_lora_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_filtered_remove_amb_contrastive_safety_epoch1_"
)

#allenai/tulu-2-7b
# hamishivi/tulu_2_7b_no_refusals
for elem in "${beaker_ids[@]}"; do
    # DESC="Merging_Lora_Checkpoints_tulu2-7b_uncensored_all_safety_adapt_v0_contrastive_${e}epochs"
    read -a strarr <<< "$elem"  # uses default whitespace IFS
    DESC="Merging_${strarr[1]}_w_tulu"
    echo $DESC
    mason --cluster ai2/mosaic-cirrascale ai2/general-cirrascale-a100-80g-ib --budget ai2/oe-adapt --gpus 1 --priority high --workspace ai2/safety-adapt --description $DESC --beaker_datasets /model:${strarr[0]} /model_base:hamishivi/tulu_2_7b_no_refusals  --  \
    python open_instruct/merge_lora_v2.py \
        --base_model_name_or_path allenai/tulu-2-7b \
        --lora_model_name_or_path /model \
        --output_dir /output \
        --save_tokenizer
done