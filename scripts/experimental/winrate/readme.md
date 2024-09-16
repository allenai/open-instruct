# Can our model generate better responses than human?

This repository contains some experimental script to do gpt 4 as a judge to see if gpt 4 prefers our model response over human-written responses. We use https://huggingface.co/datasets/HuggingFaceH4/no_robots which contains only human-written responses.


## Get started

First navigate to this directory.


```bash
python -i generate_and_eval.py \
    --model_name_or_path ai2-adapt-dev/sft-norobot \
    --output_path test.csv \
    --n 500

# preferred
# response1    0.69
# response0    0.31
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=121.41
# df['reference_response_len'].mean()=179.726

python -i generate_and_eval.py \
    --model_name_or_path allenai/llama-3.1-tulu-2-dpo-8b \
    --output_path test.csv \
    --n 500

# preferred
# response0    0.504
# response1    0.496
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=172.248
# df['reference_response_len'].mean()=179.726

python -i generate_and_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_path test.csv \
    --n 500

# preferred
# response0    0.566
# response1    0.434
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=151.506

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/do4nuqhh
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/online_dpo_vllm_thread_beta_0.03__allenai_open_instruct_dev \
    --model_revision online_dpo_vllm_thread_beta_0.03__3__1726200312 \
    --output_path test.csv \
    --n 500

# preferred
# response1    0.53
# response0    0.47
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=153.838


# https://wandb.ai/ai2-llm/open_instruct_internal/runs/jvjegpcq/
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/ppo_vllm_thread_beta_0.03__allenai_open_instruct_dev \
    --model_revision ppo_vllm_thread_beta_0.03__3__1726244716 \
    --output_path test.csv \
    --n 500
# preferred
# response1    0.554
# response0    0.446
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=146.928

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/97i9hdk3
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair__allenai_open_instruct_dev__42__1726174531 \
    --output_path test.csv \
    --n 500
# preferred
# response1    0.576
# response0    0.424
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=144.382

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/w21rugjl
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_2peoch__allenai_open_instruct_dev__42__1726241871 \
    --output_path test.csv \
    --n 500
# preferred
# response1    0.502
# response0    0.498
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=140.248

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/g70g1coa/overview
python -i generate_and_eval.py \
    --model_name_or_path allenai/open_instruct_dev \
    --model_revision costa_offline_dpo_norobot_3pair_3peoch__allenai_open_instruct_dev__42__1726354080 \
    --output_path test.csv \
    --n 500
# preferred
# response0    0.536
# response1    0.464
# Name: proportion, dtype: float64
# df['model_response_len'].mean()=136.864

# https://wandb.ai/ai2-llm/open_instruct_internal/runs/z5wfidhb
python -i generate_and_eval.py \
    --model_name_or_path vwxyzjn/online_dpo_vllm_thread_beta_0.5__allenai_open_instruct_dev \
    --model_revision online_dpo_vllm_thread_beta_0.5__3__1726368079 \
    --output_path test.csv \
    --n 500


python plot_winrate.py
```

![](winrate_plot.png)


It is basically the same thing as the win rate in https://arxiv.org/abs/2009.01325

![alt text](image.png)