# Can our model generate better responses than human?

This repository contains some experimental script to do gpt 4 as a judge to see if gpt 4 prefers our model response over human-written responses. We use https://huggingface.co/datasets/HuggingFaceH4/no_robots which contains only human-written responses.


## Get started

First navigate to this directory in order to run the scripts. Then run 


```bash
huggingface-cli download --repo-type dataset --local-dir . ai2-adapt-dev/hf_no_robots_judged
# run this if you want to sync
# huggingface-cli upload --private --repo-type dataset ai2-adapt-dev/hf_no_robots_judged csvs csvs
```


```bash
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_finetune_tulu3_8b_norobot__meta-llama_Meta-Llama-3.1-8B__42__1725559869 \
    --n 500
# preferred
# response1    0.682
# response0    0.318
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=198.402

python -i generate_and_eval.py \
    --model_name_or_path allenai/llama-3.1-tulu-2-dpo-8b \
    --n 500
# preferred
# response0    0.71
# response1    0.29
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=320.09

python -i generate_and_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --n 500
# preferred
# response0    0.816
# response1    0.184
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=309.322

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/qminkg08/overview
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/online_dpo_vllm_thread_beta_0.03_episode_42000__allenai_open_instruct_dev \
    --model_revision online_dpo_vllm_thread_beta_0.03_episode_42000__3__1726546910 \
    --n 500
# preferred
# response1    0.524
# response0    0.476
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=222.886

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/do4nuqhh
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/online_dpo_vllm_thread_beta_0.03__allenai_open_instruct_dev \
    --model_revision online_dpo_vllm_thread_beta_0.03__3__1726200312 \
    --n 500
# preferred
# response0    0.606
# response1    0.394
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=303.176

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/2mtdilmj
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/online_dpo_vllm_thread_beta_0.03_episode_126000__allenai_open_instruct_dev \
    --model_revision online_dpo_vllm_thread_beta_0.03_episode_126000__3__1726559254 \
    --n 500
# referred
# response0    0.64
# response1    0.36
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=314.074

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/twxhblyw
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/ppo_vllm_thread_beta_0.03_episode_42000__allenai_open_instruct_dev \
    --model_revision ppo_vllm_thread_beta_0.03_episode_42000__3__1726543996 \
    --n 500
# preferred
# response1    0.554
# response0    0.446
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=238.528

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/jvjegpcq/
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/ppo_vllm_thread_beta_0.03__allenai_open_instruct_dev \
    --model_revision ppo_vllm_thread_beta_0.03__3__1726244716 \
    --n 500
# preferred
# response0    0.548
# response1    0.452
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=233.058

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/i2ubt8zy
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/ppo_vllm_thread_beta_0.03_episode_126000__allenai_open_instruct_dev \
    --model_revision ppo_vllm_thread_beta_0.03_episode_126000__3__1726543993 \
    --n 500
# preferred
# response0    0.562
# response1    0.438
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=226.956


# https://wandb.ai/ai2-llm/open_instruct_internal/runs/97i9hdk3
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair__allenai_open_instruct_dev__42__1726174531 \
    --n 500
# preferred
# response0    0.578
# response1    0.422
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=273.732

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/w21rugjl
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_2peoch__allenai_open_instruct_dev__42__1726241871 \
    --n 500
# preferred
# response0    0.668
# response1    0.332
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=215.136

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/g70g1coa/overview
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_3peoch__allenai_open_instruct_dev__42__1726354080 \
    --n 500
# preferred
# response0    0.688
# response1    0.312
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=186.848


# https://wandb.ai/ai2-llm/open_instruct_internal/runs/9boxjt9g
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_1peoch_dpo__allenai_open_instruct_dev__42__1726598473 \
    --n 500
# preferred
# response1    0.654
# response0    0.346
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=209.43

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/l3sx2vy3
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_2peoch_dpo__allenai_open_instruct_dev__42__1726598454 \
    --n 500
# preferred
# response1    0.622
# response0    0.378
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=208.1

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/scc7ic9o
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_3peoch_dpo__allenai_open_instruct_dev__42__1726598454 \
    --n 500
# preferred
# response1    0.626
# response0    0.374
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=223.876

## beta 0.1
# https://wandb.ai/ai2-llm/open_instruct_internal/runs/5ro3trre
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_1peoch_dpo_beta_0.1__allenai_open_instruct_dev__42__1726616746 \
    --n 500
# preferred
# response0    0.528
# response1    0.472
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=227.462

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/jjudzmie
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_2peoch_dpo_beta_0.1__allenai_open_instruct_dev__42__1726624662 \
    --n 500
# preferred
# response0    0.558
# response1    0.442
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=205.354

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/p1g5p9f1
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_3peoch_dpo_beta_0.1__allenai_open_instruct_dev__42__1726616662 \
    --n 500
# preferred
# response0    0.574
# response1    0.426
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=205.498

## beta 0.03
# https://wandb.ai/ai2-llm/open_instruct_internal/runs/8hmgusle
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_1peoch_dpo_beta_0.03__allenai_open_instruct_dev__42__1726616748 \
    --n 500
# preferred
# response0    0.58
# response1    0.42
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=206.512

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/i3a3oodn
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_2peoch_dpo_beta_0.03__allenai_open_instruct_dev__42__1726616747 \
    --n 500
# preferred
# response0    0.578
# response1    0.422
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=207.07

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/r50e84fb
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_3peoch_dpo_beta_0.03__allenai_open_instruct_dev__42__1726624622 \
    --n 500
# preferred
# response0    0.612
# response1    0.388
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=204.078

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/qzxfu3bi
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision ppo_vllm_thread_beta_0.01_episode_42000__3__1726693093 \
    --n 500
# preferred
# response1    0.522
# response0    0.478
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=264.274

https://wandb.ai/ai2-llm/open_instruct_internal/runs/l0tjy2w7
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision ppo_vllm_thread_beta_0.01_episode_84000__3__1726693099 \
    --n 500
# preferred
# response0    0.578
# response1    0.422
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=262.338

https://wandb.ai/ai2-llm/open_instruct_internal/runs/kmy7xdjv
python generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision ppo_vllm_thread_beta_0.01_episode_126000__3__1726693097 \
    --n 500
# preferred
# response0    0.592
# response1    0.408
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=264.802


python plot_winrate.py


python -m open_instruct.hf_viz \
    --preference vwxyzjn/norobot_3pair_pref_11166 \
    --split train \
    --preference_chosen_column_name chosen \
    --preference_rejected_column_name rejected
python -m open_instruct.hf_viz \
    --preference vwxyzjn/norobot_3pair_test_pref_8412 \
    --split train \
    --preference_chosen_column_name chosen \
    --preference_rejected_column_name rejected
python -m open_instruct.hf_viz \
    --preference ai2-adapt-dev/numina_math_gsm8k_minerva_RM \
    --split test \
    --preference_chosen_column_name chosen \
    --preference_rejected_column_name rejected
```

![](winrate_plot.png)


It is basically the same thing as the win rate in https://arxiv.org/abs/2009.01325

![alt text](image.png)


## Measure agreemnt rate

```bash
python measure_agreement_rate.py \
    --reward_model_path vwxyzjn/reward_modeling__allenai_open_instruct_dev \
    --reward_model_revision reward_modeling__1__1725760619 \
    --input_path csvs/allenai/open_instruct_dev_costa_offline_dpo_norobot_3pair_3peoch__allenai_open_instruct_dev__42__1726354080_judged.csv \
    --n -1
# count: 500, agreement_rate: 45.20%

python measure_agreement_rate.py \
    --reward_model_path vwxyzjn/reward_modeling__allenai_open_instruct_dev \
    --reward_model_revision reward_modeling__1__1725760619 \
    --input_path csvs/vwxyzjn/online_dpo_vllm_thread_beta_0.03_episode_126000__allenai_open_instruct_dev_online_dpo_vllm_thread_beta_0.03_episode_126000__3__1726559254_judged.csv \
    --n -1
# count: 500, agreement_rate: 56.20%

python measure_agreement_rate.py \
    --reward_model_path vwxyzjn/reward_modeling__allenai_open_instruct_dev \
    --reward_model_revision reward_modeling__1__1725760619 \
    --input_path csvs/vwxyzjn/ppo_vllm_thread_beta_0.03_episode_126000__allenai_open_instruct_dev_ppo_vllm_thread_beta_0.03_episode_126000__3__1726543993_judged.csv \
    --n -1
# count: 500, agreement_rate: 59.60%

python measure_agreement_rate.py \
    --reward_model_path allenai/llama-3-tulu-2-8b-uf-mean-rm \
    --input_path csvs/allenai/open_instruct_dev_costa_offline_dpo_norobot_3pair_3peoch__allenai_open_instruct_dev__42__1726354080_judged.csv \
    --n -1
# count: 500, agreement_rate: 61.20%

python measure_agreement_rate.py \
    --reward_model_path allenai/llama-3-tulu-2-8b-uf-mean-rm \
    --input_path csvs/vwxyzjn/online_dpo_vllm_thread_beta_0.03_episode_126000__allenai_open_instruct_dev_online_dpo_vllm_thread_beta_0.03_episode_126000__3__1726559254_judged.csv \
    --n -1
# count: 500, agreement_rate: 64.00%

python measure_agreement_rate.py \
    --reward_model_path allenai/llama-3-tulu-2-8b-uf-mean-rm \
    --input_path csvs/vwxyzjn/ppo_vllm_thread_beta_0.03_episode_126000__allenai_open_instruct_dev_ppo_vllm_thread_beta_0.03_episode_126000__3__1726543993_judged.csv \
    --n -1
# count: 500, agreement_rate: 63.40%


https://huggingface.co/vwxyzjn/reward_modeling__allenai_llama-3-tulu-2-8b/tree/reward_modeling__1__1726175049
python measure_agreement_rate.py \
    --reward_model_path vwxyzjn/reward_modeling__allenai_llama-3-tulu-2-8b \
    --reward_model_revision reward_modeling__1__1726175049 \
    --input_path csvs/allenai/open_instruct_dev_costa_offline_dpo_norobot_3pair_3peoch__allenai_open_instruct_dev__42__1726354080_judged.csv \
    --n -1
# count: 500, agreement_rate: 63.00%

python measure_agreement_rate.py \
    --reward_model_path vwxyzjn/reward_modeling__allenai_llama-3-tulu-2-8b \
    --reward_model_revision reward_modeling__1__1726175049 \
    --input_path csvs/vwxyzjn/online_dpo_vllm_thread_beta_0.03_episode_126000__allenai_open_instruct_dev_online_dpo_vllm_thread_beta_0.03_episode_126000__3__1726559254_judged.csv \
    --n -1
# count: 500, agreement_rate: 63.00%

python measure_agreement_rate.py \
    --reward_model_path vwxyzjn/reward_modeling__allenai_llama-3-tulu-2-8b \
    --reward_model_revision reward_modeling__1__1726175049 \
    --input_path csvs/vwxyzjn/ppo_vllm_thread_beta_0.03_episode_126000__allenai_open_instruct_dev_ppo_vllm_thread_beta_0.03_episode_126000__3__1726543993_judged.csv \
    --n -1
# count: 500, agreement_rate: 58.40%

python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision "ppo_vllm_thread_beta_0.03__3__1732222410" \
    --n 500
# preferred
# response0    263
# response1    237
# Name: count, dtype: int64
# preferred
# response0    0.526
# response1    0.474
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=229.416
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision "ppo_vllm_thread_no_async_beta_0.03__3__1732222411" \
    --n 500
# preferred
# response0    265
# response1    235
# Name: count, dtype: int64
# preferred
# response0    0.53
# response1    0.47
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=220.348
```