# Rejection sampling

This is a technique used in the Llama 3 paper. The basic idea is to sample `n` (typically between 10 and 30) outputs from the latest chat model policy (usually
the best performing checkpoint of some kind) and use a reward model to select the best candidate. In the following script, we can vary the `--n` to generate
different number of completions per prompt.


# Debug run (use an interactive session)

```bash
## tulu v3 recipe
# 1. first sample a bunch of completions given prompts
python open_instruct/generation.py \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --model_name_or_path allenai/llama-3-tulu-2-8b \
    --n 3 \
    --save_filename output/completions.jsonl \
    --sanity_check \
    
# 2. tokenize them and run a reward model to filter them
python open_instruct/rejection_sampling.py \
    --input_filename output/completions.jsonl \
    --model_names_or_paths allenai/llama-3-tulu-2-8b-uf-mean-rm \
    --save_filename output/rejection_sampled_completions.jsonl \
    --n 3 \
    --push_to_hub \
    --num_gpus 1 \
```

# Implementation details

Note that it is possible to generate identical completions per prompt, which is not going to be that useful, so we filter them out via

```py
if len(set([item.text for item in output.outputs])) == 1:
    continue
```


# Run through the entire dataset run

To run through the entire dataset you would need a lot more GPUs to finish the generation more quickly. 


```bash
# prod generations
# NOTE: the scripts below only generate 400 prompts, so it's for demonstration purposes only. The scripts are highly scalable, and you could modify its `num_prompts=400` to something else like 300000 for the tulu dataset.
bash scripts/rejection_sampling_hello_world.bash
bash scripts/rejection_sampling_tulu.bash
```

You can see a demo [here](https://drive.google.com/file/d/1dq3KG15ajpOv8tFYEZGS4tlW7G55oOYP/view?usp=sharing)

<img width="1327" alt="image" src="https://github.com/user-attachments/assets/71a15671-e054-4eab-a571-715881958e74">



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
