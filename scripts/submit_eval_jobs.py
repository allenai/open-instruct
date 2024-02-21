import copy
import subprocess
import yaml
import re
import itertools
from datetime import date
import argparse

today = date.today().strftime("%m%d%Y")

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", type=str, default="hamishivi")
parser.add_argument("--model_name", type=str, default="hf-opt-7B")
parser.add_argument("--location", type=str, default=None)
parser.add_argument("--beaker_subfolder", type=str, default=None)
parser.add_argument("--cluster", nargs='+', default=["ai2/allennlp-cirrascale", "ai2/general-cirrascale", "ai2/general-cirrascale-a100-80g-ib", "ai2/mosaic-cirrascale-a100", "ai2/s2-cirrascale-l40"])
parser.add_argument("--is_tuned", action="store_true")
parser.add_argument("--use_hf_tokenizer_template", action="store_true")
parser.add_argument("--priority", type=str, default="preemptible")
args = parser.parse_args()


workspace = args.workspace
model_type = "vanilla_lm" if not args.is_tuned else "tuned_lm"

with open("beaker_configs/default_eval.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

cluster = args.cluster
if cluster[0] == "all":
    cluster = []  # empty list means all clusters
d1['tasks'][0]['constraints']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = args.priority
d1['tasks'][0]['resources']['gpuCount'] = 1

# modify here for different set of experiments
experiment_groups = [
    "mmlu_0shot",
    "mmlu_5shot",
    "gsm_direct",
    "gsm_cot",
    "bbh_direct",
    "bbh_cot",
    "tydiqa_goldp_1shot",
    "tydiqa_no_context_1shot",
    "codex_eval_temp_0.1",
    "codex_eval_temp_0.8",
    "trutufulqa",
    "toxigen",
    "alpaca_eval",
]

# format: model name, their beaker id, checkpoint subfolder, tuned or base.
# or: name, path, None, tuned or base
models = [
    # llama1 models
    # ("llama1-7B", "01HCCBK1MYKXKQC0C6CSVW1F22", None, "vanilla_lm"),

    # other causal models
    # ("hf-opt-7B", "facebook/opt-6.7b", None, "vanilla_lm"),
    ("olmo_7b_final_dpo", "/net/nfs.cirrascale/allennlp/hamishi/checkpoints/olmo_7b_finetune_dpo", None, "tuned_lm"),
    # (args.model_name, args.location, args.beaker_subfolder, model_type),
]

#--------------- experiments about number of supervision tasks -------------------------

# for experiment_group, model_info in itertools.product(experiment_groups, models):
for model_info, experiment_group in itertools.product(models, experiment_groups):
    print(f"Submitting {experiment_group} for model: {model_info[0]}")
    d = copy.deepcopy(d1)

    model_name = model_info[0] + f"_{model_info[2]}" if model_info[2] is not None else model_info[0]
    name = f"open_instruct_eval_{experiment_group}_{model_name}_{today}"
    d['description'] = name
    d['tasks'][0]['name'] = name

    if experiment_group == "mmlu_0shot":
        d['tasks'][0]['arguments'][0] = '''
            python -m eval.mmlu.run_eval \
            --ntrain 0 \
            --data_dir /data/mmlu/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 4 \
            --load_in_8bit \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "mmlu_5shot":
        d['tasks'][0]['arguments'][0] = '''
            python -m eval.mmlu.run_eval \
            --ntrain 5 \
            --data_dir /data/mmlu/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 4 \
            --load_in_8bit \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "bbh_direct":
        d['tasks'][0]['arguments'][0] = '''
            python -m eval.bbh.run_eval \
            --data_dir /data/bbh \
            --save_dir /output/ \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --max_num_examples_per_task 40 \
            --no_cot \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "bbh_cot":
        d['tasks'][0]['arguments'][0] = '''
            python -m eval.bbh.run_eval \
            --data_dir /data/bbh \
            --save_dir /output/ \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --max_num_examples_per_task 40 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "gsm_direct":
        d['tasks'][0]['arguments'][0] = '''
            python -m eval.gsm.run_eval \
            --data_dir /data/gsm/ \
            --max_num_examples 200 \
            --save_dir /output/ \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --n_shot 8 \
            --no_cot \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "gsm_cot":
        d['tasks'][0]['arguments'][0] = '''
            python -m eval.gsm.run_eval \
            --data_dir /data/gsm/ \
            --max_num_examples 200 \
            --save_dir /output/ \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --n_shot 8 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        ''' 
    elif experiment_group == "tydiqa_goldp_1shot":
        d["tasks"][0]["arguments"][0] = '''
            python -m eval.tydiqa.run_eval \
            --data_dir /data/tydiqa/ \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir /output/ \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "tydiqa_no_context_1shot":
        d["tasks"][0]["arguments"][0] = '''
            python -m eval.tydiqa.run_eval \
            --data_dir /data/tydiqa/ \
            --no_context \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir /output/ \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "codex_eval_temp_0.1":
        d['tasks'][0]['arguments'][0] = '''
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.1 \
            --save_dir /output/ \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model
        '''
    elif experiment_group == "codex_eval_temp_0.8":
        d['tasks'][0]['arguments'][0] = '''
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.8 \
            --save_dir /output/ \
            --use_vllm \
            --model /model \
            --tokenizer_name_or_path /model
        '''
    elif experiment_group == "trutufulqa":
        d['tasks'][0]['arguments'][0] = '''
        python -m eval.truthfulqa.run_eval \
            --data_dir /data/truthfulqa \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --metrics truth info mc \
            --preset qa \
            --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
            --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
            --eval_batch_size 20 \
            --load_in_8bit \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "toxigen":
        d['tasks'][0]['arguments'][0] = '''
        python -m eval.toxigen.run_eval \
            --data_dir /data/toxigen/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 32 \
            --use_vllm \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "alpaca_eval":
        d['tasks'][0]['arguments'][0] = '''
        python -m eval.alpaca_farm.run_eval \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --save_dir /output/ \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    else:
        raise ValueError("experiment_group not supported")

    if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
    elif model_info[1].startswith("/"):  # if it's a local model, load it from the local directory
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
    else:  # if it's a beaker model, mount the beaker dataset to `/model`
        d['tasks'][0]['datasets'][1]['source']['beaker'] = model_info[1]

    # if a specific checkpoint is specified, load model from that checkpoint
    if model_info[2] is not None:
        # extract existing model path
        model_name_or_path = re.search("--model_name_or_path (\S+)", d['tasks'][0]['arguments'][0]).group(1)
        # replace the model path with the checkpoint subfolder
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(model_name_or_path, model_name_or_path+"/"+model_info[2])]
        # replace the tokenizer path with the checkpoint subfolder
        tokenizer_name_or_path = re.search("--tokenizer_name_or_path (\S+)", d['tasks'][0]['arguments'][0]).group(1)

    # for vanilla_lm, remove the chat formatting function
    if model_info[3] == "vanilla_lm":
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]

    if "13B" in model_info[0]:
        # find the batch size argument, and reduce by 4x
        if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
            original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
            new_batch_size = max(1, int(original_batch_size) // 2)
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]


    if "30B" in model_info[0] or "34B" in model_info[0]:
        # find the batch size argument, and reduce by 4x
        if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
            original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
            new_batch_size = max(1, int(original_batch_size) // 4)
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

        if "codex_eval" in experiment_group:
            # request 2x more GPUs
            d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']
    
    elif "70B" in model_info[0] or "65B" in model_info[0] or "40B" in model_info[0]:
        # find the batch size argument, and reduce by 4x
        if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
            original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
            new_batch_size = max(1, int(original_batch_size) // 4)
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

        if "codex_eval" in experiment_group:
            # request 4x more GPUs
            d['tasks'][0]['resources']['gpuCount'] = 4 * d['tasks'][0]['resources']['gpuCount']
        else:
            # request 2x more GPUs
            d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']

    # if using huggingface tokenizer template, replace the chat formatting function with hf tokenizer one
    if args.use_hf_tokenizer_template:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template")
        ]
    if "llama2-chat" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
        ]
    elif "code_llama_instruct" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
        ]
    elif "zephyr" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_zephyr_chat_format")
        ]
    elif "xwin" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_xwin_chat_format")
        ]
    elif "olmo" in model_info[0]:
        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format")
        ]
        # no vllm for olmo yet
        if "--use_vllm" in d['tasks'][0]['arguments'][0]:
            print(f"Removing --use_vllm for {model_info[0]}")
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_vllm", "")] 


    if any([x in model_info[0] for x in ["opt", "pythia", "falcon"]]):
        if "--use_vllm" in d['tasks'][0]['arguments'][0]:
            print(f"Removing --use_vllm for {model_info[0]}")
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_vllm", "")] 

    # print(d)

    fn = "beaker_configs/auto_created/{}.yaml".format(name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/{}".format(fn, workspace)
    subprocess.Popen(cmd, shell=True)
