# Rejection sampling

This is a technique used in the Llama 3 paper. The basic idea is to sample `n` (typically between 10 and 30) outputs from the latest chat model policy (usually
the best performing checkpoint of some kind) and use a reward model to select the best candidate. In the following script, we can vary the `--num_completions` to generate
different number of completions per prompt.


# Debug run (use an interactive session)

This code supports HF models, local models and also API-based models (e.g., `gpt-4`). For generating completions, the code now accepts one model at a time, but we're working on adding an ensemble of models. Stay tuned. 

```bash
# 1. first sample a bunch of completions given prompts
# Here is an example created dataset: https://huggingface.co/datasets/vwxyzjn/generation_1727879425
python open_instruct/rejection_sampling/generation.py \
    --dataset_mixer_list allenai/tulu-v2-sft-mixture 100 \
    --dataset_splits train \
    --model_name_or_path allenai/llama-3-tulu-2-8b \
    --num_completions 3 \
    --save_filename output/completions.jsonl \
    --push_to_hub
```

### Scoring completions
You can use either a single RM to score responses or a list of RMs. In the latter case, we will take the majority vote to compute the final score. The RMs can be models explicitly trained as RMs, HF LMs, or API-based models.

Note that by default we include the reference completion in the list of completions to perform rejection sampling. This can be disabled by setting `--no_include_reference_completion_for_rejection_sampling`

```bash
# 2.1 tokenize them and run a reward model to filter them
# Here is an example created dataset: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_1727887719
# Here is an example created dataset for raw scores: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_scores_1727887719/
python open_instruct/rejection_sampling/rejection_sampling.py \
    --input_filename output/completions.jsonl \
    --model_names_or_paths allenai/llama-3-tulu-2-8b-uf-mean-rm \
    --save_filename_scores output/completions_scores.jsonl \
    --save_filename output/rejection_sampled_completions.jsonl \
    --num_completions 3 \
    --push_to_hub \
    --num_gpus 1 \

# 2.1.2 without reference completion in rejection sampling
# Here is an example created dataset: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_1727887719
# Here is an example created dataset for raw scores: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_scores_1727887719/
python open_instruct/rejection_sampling/rejection_sampling.py \
    --input_filename output/completions.jsonl \
    --model_names_or_paths allenai/llama-3-tulu-2-8b-uf-mean-rm \
    --save_filename_scores output/completions_scores.jsonl \
    --save_filename output/rejection_sampled_completions.jsonl \
    --no_include_reference_completion_for_rejection_sampling \
    --num_completions 3 \
    --push_to_hub \
    --num_gpus 1 \

# 2.2 tokenize them and run llm as a judge
# Note then when using LLM as a judge, it's possible that llm api failed to produce a score in our expected
# format, so score extraction failed and we simply mark the score -1.
# Here is an example created dataset: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_1727889563
# Here is an example created dataset for raw scores: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_scores_1727889563
python open_instruct/rejection_sampling/rejection_sampling.py \
    --input_filename output/completions.jsonl \
    --model_names_or_paths gpt-4o-mini  \
    --save_filename_scores output/completions_scores.jsonl \
    --save_filename output/rejection_sampled_completions.jsonl \
    --num_completions 3 \
    --push_to_hub \
    --num_gpus 1 \

# 2.3 tokenize them and run a combination of reward models / llm as a judge
# Here is an example created dataset: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_1724273702
# Here is an example created dataset for raw scores: https://huggingface.co/datasets/vwxyzjn/rejection_sampling_scores_1724273702
python open_instruct/rejection_sampling/rejection_sampling.py \
    --input_filename output/completions.jsonl \
    --model_names_or_paths allenai/llama-3-tulu-2-8b-uf-mean-rm gpt-4o-mini gpt-4-turbo \
    --save_filename_scores output/completions_scores.jsonl \
    --save_filename output/rejection_sampled_completions.jsonl \
    --num_completions 3 \
    --push_to_hub \
    --num_gpus 1 \
 ```



# Run through the entire dataset run

To run through the entire dataset you would need a lot more GPUs to finish the generation more quickly. 


```bash
# NOTE: the scripts below only generate 400 prompts, so it's for demonstration purposes only. The scripts are highly scalable, and you could modify its `num_prompts=400` to something else like 300000 for the tulu dataset.

# you need to make sure your default beaker workspace has WANDB_API_KEY and HF_TOKEN secrets in them
beaker secret write HF_TOKEN xxxxxxxxxxxx
beaker secret write WANDB_API_KEY xxxxxxxxxxx

# You can use docker to do the job submission
bash scripts/rejection_sampling_tulu_docker.bash

# if you are using mason you can debug with the following command(s), the
# rejection sampling shards should appear in your local foldeer
bash scripts/rejection_sampling_tulu.bash
```

You can see a demo [here](https://drive.google.com/file/d/1dq3KG15ajpOv8tFYEZGS4tlW7G55oOYP/view?usp=sharing)

<img width="1327" alt="image" src="https://github.com/user-attachments/assets/71a15671-e054-4eab-a571-715881958e74">


# Implementation details

Note that it is possible to generate identical completions per prompt, which is not going to be that useful, so we filter them out via

```py
if len(set([item.text for item in output.outputs])) == 1:
    continue
```



## Debug commands

```bash
# debug job submission; you should install your python on NFS and
# make sure `which python` returns the python environment you are using
python mason.py \
    --cluster ai2/allennlp-cirrascale ai2/general-cirrascale-a5000 ai2/general-cirrascale-a5000 ai2/general-cirrascale-a100-80g-ib \
    --priority low \
    --budget ai2/allennlp \
    --gpus 1 -- which python
# sometimes we run into permission issues; need to run the following
python mason.py \
    --cluster ai2/allennlp-cirrascale ai2/general-cirrascale-a5000 ai2/general-cirrascale-a5000 ai2/general-cirrascale-a100-80g-ib \
    --priority low \
    --budget ai2/allennlp \
    --gpus 1 -- chmod -R 777 /net/nfs.cirrascale/allennlp/.cache/
```
