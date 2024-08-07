
# preparation 

Make sure you are in the `rejection_sampling` folder.


# Debug run (use an interactive session)

```bash
## tulu v3 recipe
# 1. first sample a bunch of completions given prompts
python rejection_sampling/generation.py \
    --dataset_name allenai/tulu-v2-sft-mixture \
    --model_name_or_path allenai/llama-3-tulu-2-8b \
    --n 3 \
    --save_filename rejection_sampling/completions.jsonl \
    --sanity_check \
    
# 2. tokenize them and run a reward model to filter them
python rejection_sampling/rejection_sampling.py \
    --input_filename rejection_sampling/completions.jsonl \
    --model_name_or_path allenai/llama-3-tulu-2-8b-uf-mean-rm \
    --save_filename rejection_sampling/rejection_sampled_completions.jsonl \
    --n 3 \
    --push_to_hub \
    --num_gpus 1 \
```



# Run through the entire dataset run

To run through the entire dataset you would need a lot more GPUs to finish the generation more quickly. 


```
# prod generations
# NOTE: the scripts below only generate 400 prompts, so it's for demonstration purposes only. The scripts are highly scalable, and you could modify its `num_prompts=400` to something else like 300000 for the tulu dataset.
bash rejection_sampling/e2e_hello_world.bash
bash rejection_sampling/batch_rejection_sampling.bash
```

You can see a demo [here](https://drive.google.com/file/d/1dq3KG15ajpOv8tFYEZGS4tlW7G55oOYP/view?usp=sharing)

<img width="1327" alt="image" src="https://github.com/user-attachments/assets/71a15671-e054-4eab-a571-715881958e74">


```bash
# debug job submission
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
    --gpus 1 -- chmod -R 777 /net/nfs.cirrascale/allennlp/.cache/hub/
```
