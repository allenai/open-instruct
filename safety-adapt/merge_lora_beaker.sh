MODEL_SIZE=7B
NUM_GPUS=2
DESC="Merging_refusal_tunning_tulu-2-7b_uncensored_refusal_adapt_v0.1_filtered_combined_contrastive_safety_epoch1"
#"Merging_Lora_tulu-2-7b_safety_adapt_v0.1_contrastive_v0.1_augmented_filterv2_epoch1" #"Merging_Lora_Checkpoints_tulu2-7b_safety_top25_match_tulu_v2"
BEAKER_ID=01HSF6KG3PRXYKH128D7CQ5F1Z #(top 25) #01HRFFSH9HYQ0N322DVMKFVJWM (bottom 25) # 01HRG3BEP8C5V4AASXYBKQMZ2M (bottom25_match_tulu) # 01HRDW66NZFFNTM61GDH070CNG (safety_match_tulu) # 01HRMJV0FHS8H31AVXCFHN4ET4 (safety_tulu_match_top25)
# 01HRNB2DGNG7WWYVGJ1X19RSJ8 (safetytuned llama redteaming) # 01HRQFE3T72G4EW0YB2934GBJ5 (all safety_adapt v0.1)
# 01HRWTBM4VFANMWJXME6ER77P2 (tulu2-7b-uncensored all_safety adapt v0.1)

# ai2/mosaic-cirrascale 
# ai2/general-cirrascale-a100-80g-ib
mason --cluster ai2/mosaic-cirrascale ai2/general-cirrascale-a100-80g-ib --budget ai2/oe-adapt --gpus 2 --priority high --workspace ai2/safety-adapt --description $DESC --beaker_datasets /model:01HV50FCS5036F81SRHT1XA81W  --  \
python open_instruct/merge_lora_v2.py \
    --base_model_name_or_path allenai/tulu-2-7b \
    --lora_model_name_or_path /model \
    --output_dir /output \
    --save_tokenizer
