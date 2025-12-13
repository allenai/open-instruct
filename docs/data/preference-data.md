# Current preference datasets

To build all the datasets at once (use this carefully), run:
```
sh scripts/data/preferences/prepare_all.sh
```

## Chat

### Maintained here

First, older popular datasets. Build these datasets (a subset only) with:
```
python scripts/data/preferences/webgpt.py --push_to_hub --hf_entity=ai2-adapt-dev
python scripts/data/preferences/hh-harmless.py --push_to_hub --hf_entity=ai2-adapt-dev
python scripts/data/preferences/hh-helpful.py --push_to_hub --hf_entity=ai2-adapt-dev
```
* [ai2-adapt-dev/webgpt-binarized](https://huggingface.co/datasets/ai2-adapt-dev/webgpt-binarized)
* [ai2-adapt-dev/hh-rlhf-harmless](https://huggingface.co/datasets/ai2-adapt-dev/hh-rlhf-harmless)
* [ai2-adapt-dev/hh-rlhf-helpful](https://huggingface.co/datasets/ai2-adapt-dev/hh-rlhf-helpful)

Next, Nvidia's recent HelpSteer 2.
They are created with:
```
python scripts/data/preferences/helpsteer2.py --push_to_hub --min_score 2.5 --hf_entity=ai2-adapt-dev
python scripts/data/preferences/helpsteer2.py --push_to_hub --min_score 2 --hf_entity=ai2-adapt-dev
python scripts/data/preferences/helpsteer2.py --push_to_hub --min_score 2 --hf_entity=ai2-adapt-dev --aspects_to_ignore verbosity
python scripts/data/preferences/helpsteer2.py --push_to_hub --hf_entity=ai2-adapt-dev --aspects_to_ignore verbosity
```
The binarization weighting that Nvidia recommends can be used with:
```
python scripts/data/preferences/helpsteer2_nvidia.py --push_to_hub --hf_entity ai2-adapt-dev
```
Some examples include:
* [ai2-adapt-dev/helpsteer-2-binarized-above-2.0-margin-0.5-ignore-verbosity](https://huggingface.co/datasets/ai2-adapt-dev/helpsteer-2-binarized-above-2.0-margin-0.5-ignore-verbosity)
* [ai2-adapt-dev/helpsteer-2-binarized-ignore-verbosity](https://huggingface.co/datasets/ai2-adapt-dev/helpsteer-2-binarized-ignore-verbosity): This ignores verbosity aspect, which is unclear in the paper.
* [ai2-adapt-dev/helpsteer2-binarized-nvidia-spec](https://huggingface.co/datasets/ai2-adapt-dev/helpsteer2-binarized-nvidia-spec): This uses the specific weighting that Nvidia converged on in their HelpSteer2 paper training multiple types of reward models.

Also, specific splits of Nectar (randomly binarized from top 3 completions and a bottom completion) are included with:
```
python scripts/data/preferences/nectar.py --push_to_hub --hf_entity ai2-adapt-dev
python scripts/data/preferences/nectar.py --push_to_hub --subset anthropic-hh --hf_entity ai2-adapt-dev
python scripts/data/preferences/nectar.py --push_to_hub --deduplication --hf_entity ai2-adapt-dev
```
The default split is `lmsys-chat-1m`.
The last example is called "deduplication" due to potential overlap with UltraFeedback, given they source from the same underlying dataset. Basic tests showed they did not use the same prompts, but slight modifications could've occured.
* [ai2-adapt-dev/nectar_binarized-anthropic-hh](https://huggingface.co/datasets/ai2-adapt-dev/nectar_binarized-anthropic-hh)
* [ai2-adapt-dev/nectar_binarized-lmsys-chat-1m](https://huggingface.co/datasets/ai2-adapt-dev/nectar_binarized-lmsys-chat-1m)
* [ai2-adapt-dev/nectar_binarized-dedup-ultrafeedback](https://huggingface.co/datasets/ai2-adapt-dev/nectar_binarized-dedup-ultrafeedb)ack

### Stored on HF
* [allenai/ultrafeedback_binarized_cleaned_train](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned_train)
* [ai2-adapt-dev/summarize_from_feedback](https://huggingface.co/datasets/ai2-adapt-dev/summarize_from_feedback)
* [ai2-adapt-dev/DaringAnteater-prefs](https://huggingface.co/datasets/ai2-adapt-dev/DaringAnteater-prefs)
* [ai2-adapt-dev/DaringAnteater-prefs-RM-filter](https://huggingface.co/datasets/ai2-adapt-dev/DaringAnteater-prefs-RM-filter)
* [ai2-adapt-dev/WildChat-prefs-280824](https://huggingface.co/datasets/ai2-adapt-dev/WildChat-prefs-280824)
* [ai2-adapt-dev/helpsteer2-binarized-mean-aspects](https://huggingface.co/datasets/ai2-adapt-dev/helpsteer2-binarized-mean-aspects): Similar to our other HelpSteer splits, less processing.
* [ai2-adapt-dev/Skywork-Magpie](https://huggingface.co/datasets/ai2-adapt-dev/Skywork-Magpie): Subset of the [Skywork Preference Dataset](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.1) for only the [Magpie](https://arxiv.org/abs/2406.08464) splits.

### UltraFeedback Replication

**The current replications have fewer prompts than the original. These are built by splitting the original and recreating completions. We are working on merging them.**

Build these datasets with:
```
python scripts/data/preferences/ultrafeedback.py --push_to_hub --hf_entity=ai2-adapt-dev
```
The master version of the UltraFeedback pipeline replication can be found here:
[ai2-adapt-dev/ultrafeedback-pipeline-replication](https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-pipeline-replication)

UltraFeedback variants explore different combinations of prompt sources, model diversity, sampling methods, and prompt templates:

- Setup 0: Replication of original UltraFeedback
- Setup 1-2: Custom prompts with UltraFeedback methodology
- Setup 3-4: Custom prompts with varied model diversity and principle sampling
- Setup 5: Custom prompts with UltraFeedback template
- Setup 6: Increased model diversity

- [ai2-adapt-dev/ultrafeedback-replication-p0](https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-replication-p0)
- [ai2-adapt-dev/ultrafeedback-replication-p1](https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-replication-p1)
- [ai2-adapt-dev/ultrafeedback-replication-p2](https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-replication-p2)
- [ai2-adapt-dev/ultrafeedback-replication-p3](https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-replication-p3)
- [ai2-adapt-dev/ultrafeedback-replication-p4](https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-replication-p4)
- [ai2-adapt-dev/ultrafeedback-replication-p5](https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-replication-p5)
- [ai2-adapt-dev/ultrafeedback-replication-p6](https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-replication-p6)

## UltraInteract Variants
Build these datasets with:
```
python scripts/data/preferences/ultrainteract.py --push_to_hub --hf_entity=ai2-adapt-dev
```
Split by category and by selecting the longest conversations per prompt or a random length per prompt.
From [UltraInteract_pair](https://huggingface.co/datasets/openbmb/UltraInteract_pair).

* [ai2-adapt-dev/UltraInteract_pair_maxlen_Coding](https://huggingface.co/datasets/ai2-adapt-dev/UltraInteract_pair_maxlen_Coding)
* [ai2-adapt-dev/UltraInteract_pair_randomlen_Coding](https://huggingface.co/datasets/ai2-adapt-dev/UltraInteract_pair_randomlen_Coding)
* [ai2-adapt-dev/UltraInteract_pair_maxlen_Math_CoT](https://huggingface.co/datasets/ai2-adapt-dev/UltraInteract_pair_maxlen_Math_CoT)
* [ai2-adapt-dev/UltraInteract_pair_randomlen_Math_CoT](https://huggingface.co/datasets/ai2-adapt-dev/UltraInteract_pair_randomlen_Math_CoT)
* [ai2-adapt-dev/UltraInteract_pair_maxlen_Math_PoT](https://huggingface.co/datasets/ai2-adapt-dev/UltraInteract_pair_maxlen_Math_PoT)
* [ai2-adapt-dev/UltraInteract_pair_randomlen_Math_PoT](https://huggingface.co/datasets/ai2-adapt-dev/UltraInteract_pair_randomlen_Math_PoT)
* [ai2-adapt-dev/UltraInteract_pair_maxlen_Logic](https://huggingface.co/datasets/ai2-adapt-dev/UltraInteract_pair_maxlen_Logic)
* [ai2-adapt-dev/UltraInteract_pair_randomlen_Logic](https://huggingface.co/datasets/ai2-adapt-dev/UltraInteract_pair_randomlen_Logic)

## Tulu 2.5 Data
Build these datasets with:
```
python scripts/data/preferences/split_tulu2.5_prefs.py --push_to_hub --hf_entity=ai2-adapt-dev

```
Split from [this dataset](https://huggingface.co/datasets/allenai/tulu-2.5-preference-data) for easier mixing:
* [ai2-adapt-dev/tulu-2.5-prefs-alpaca_farm_gpt4_pref](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-alpaca_farm_gpt4_pref)
* [ai2-adapt-dev/tulu-2.5-prefs-alpaca_farm_human_pref](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-alpaca_farm_human_pref)
* [ai2-adapt-dev/tulu-2.5-prefs-argilla_dpo_mix](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-argilla_dpo_mix)
* [ai2-adapt-dev/tulu-2.5-prefs-capybara](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-capybara)
* [ai2-adapt-dev/tulu-2.5-prefs-chatbot_arena_2023](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-chatbot_arena_2023)
* [ai2-adapt-dev/tulu-2.5-prefs-chatbot_arena_2024](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-chatbot_arena_2024)
* [ai2-adapt-dev/tulu-2.5-prefs-helpsteer](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-helpsteer)
* [ai2-adapt-dev/tulu-2.5-prefs-hh_rlhf](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-hh_rlhf)
* [ai2-adapt-dev/tulu-2.5-prefs-hh_rlhf_60k](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-hh_rlhf_60k)
* [ai2-adapt-dev/tulu-2.5-prefs-nectar](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-nectar)
* [ai2-adapt-dev/tulu-2.5-prefs-nectar_60k](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-nectar_60k)
* [ai2-adapt-dev/tulu-2.5-prefs-orca_dpo_pairs](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-orca_dpo_pairs)
* [ai2-adapt-dev/tulu-2.5-prefs-preference_big_mixture](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-preference_big_mixture)
* [ai2-adapt-dev/tulu-2.5-prefs-prm800k_pairs_phase2](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-prm800k_pairs_phase2)
* [ai2-adapt-dev/tulu-2.5-prefs-shp_2](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-shp_2)
* [ai2-adapt-dev/tulu-2.5-prefs-stack_exchange_60k](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-stack_exchange_60k)
* [ai2-adapt-dev/tulu-2.5-prefs-stack_exchange_paired](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-stack_exchange_paired)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_evol_instruct](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_evol_instruct)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_false_qa](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_false_qa)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_flan_v2](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_flan_v2)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_lower_10k](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_lower_10k)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_mean_aspects](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_mean_aspects)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_middle_10k](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_middle_10k)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_overall](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_overall)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_sharegpt](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_sharegpt)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_top_10k](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_top_10k)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_truthful_qa](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_truthful_qa)
* [ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_ultrachat](https://huggingface.co/datasets/ai2-adapt-dev/tulu-2.5-prefs-ultrafeedback_ultrachat)
