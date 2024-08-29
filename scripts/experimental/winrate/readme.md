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

python -i generate_and_eval.py \
    --model_name_or_path allenai/llama-3.1-tulu-2-dpo-8b \
    --output_path test.csv \
    --n 500

# preferred
# response0    0.504
# response1    0.496
# Name: proportion, dtype: float64

python -i generate_and_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output_path test.csv \
    --n 500

# preferred
# response0    0.566
# response1    0.434
# Name: proportion, dtype: float64


python plot_winrate.py
```

![](winrate_plot.png)


It is basically the same thing as the win rate in https://arxiv.org/abs/2009.01325

![alt text](image.png)