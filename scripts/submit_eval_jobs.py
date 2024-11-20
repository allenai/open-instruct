import copy
import subprocess
import yaml
import re
import itertools
from datetime import date
import argparse
import os


########################################

# Helper functions.

def adjust_batch_size(task_spec, model_name, batch_size_reduction):
    "Adjust batch size using heuristics that are good for A100-size GPUs."
    reduce_by_2 = ["13B"]
    reduce_by_4 = ["30B", "34B", "40B", "65B", "70B", "70b", "72B", "72b"]
    # If not given, choose a value based on the model name.
    if batch_size_reduction is None:
        if any([pattern in model_name for pattern in reduce_by_2]):
            batch_size_reduction = 2
        elif any([pattern in model_name for pattern in reduce_by_4]):
            batch_size_reduction = 4
        else:
            batch_size_reduction = 1

    # Reduce accordingly.
    if "--eval_batch_size" in task_spec['arguments'][0]:
        original_batch_size = re.search("--eval_batch_size (\d+)", task_spec['arguments'][0]).group(1)
        new_batch_size = max(1, int(original_batch_size) // batch_size_reduction)
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

    return task_spec


def adjust_gpus(task_spec, experiment_group, model_name, gpu_multiplier):
    "Adjust GPU count using heuristics that are good for A100-size GPUs."
    medium = ["30B", "34B"]
    large = ["40B", "65B", "70B", "70b", "72B", "72b"]
    # If not given, choose a value based on model name. 
    if gpu_multiplier is None:
        if any([pattern in model_name for pattern in medium]):
            default_multiplier = 1
            codex_multiplier = 2
        elif any([pattern in model_name for pattern in large]):
            default_multiplier = 2
            codex_multiplier = 4
        else:
            default_multiplier = codex_multiplier = 1
    else:
        default_multiplier = gpu_multiplier
        # If a gpu multiplier is given, double the gpus for Codex.
        codex_multiplier = 2 * gpu_multiplier

    # Increase accordingly. 
    if "codex_eval" in experiment_group:
        task_spec['resources']['gpuCount'] = codex_multiplier * task_spec['resources']['gpuCount']
    else:
        task_spec['resources']['gpuCount'] = default_multiplier * task_spec['resources']['gpuCount']

    return task_spec

    
########################################
# Launcher

NFS_CLUSTERS = [
    "ai2/allennlp-cirrascale",
    "ai2/aristo-cirrascale",
    "ai2/climate-cirrascale",
    "ai2/general-cirrascale",
    "ai2/general-cirrascale-a5000",
    "ai2/mosaic-cirrascale",
    "ai2/mosaic-cirrascale-a100",
    "ai2/pluto-cirrascale",
    "ai2/prior-cirrascale",
    "ai2/s2-cirrascale",
    "ai2/s2-cirrascale-l40",
]

WEKA_CLUSTERS = [
    "ai2/jupiter-cirrascale-2",
    "ai2/saturn-cirrascale",
    "ai2/neptune-cirrascale",
    "ai2/allennlp-elara-cirrascale",
]
GCP_CLUSTERS = [
    "ai2/augusta-google-1"
]


today = date.today().strftime("%m%d%Y")

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", type=str, default="oe-adapt-general")
parser.add_argument("--model_name", type=str, default="hf-opt-7B")
parser.add_argument("--hf_revision", type=str, default=None)
parser.add_argument("--location", type=str, default=None)
parser.add_argument("--beaker_image", type=str, default="nathanl/open_instruct_auto", help="If given, use this Beaker image.")
parser.add_argument("--beaker_subfolder", type=str, default=None)
parser.add_argument("--cluster", nargs='+', default=[
    "ai2/allennlp-cirrascale",
    "ai2/general-cirrascale",
    "ai2/s2-cirrascale-l40",
    "ai2/allennlp-elara-cirrascale",
    "ai2/pluto-cirrascale",
    "ai2/neptune-cirrascale",
    "ai2/saturn-cirrascale",
    "ai2/jupiter-cirrascale-2",
])
parser.add_argument("--is_tuned", action="store_true")
parser.add_argument("--use_hf_tokenizer_template", action="store_true")
parser.add_argument("--priority", type=str, default="low")
parser.add_argument("--preemptible", action="store_true", default=False, help="for using preemtipble jobs (required on some instances)")
parser.add_argument("--olmo", action="store_true", help="Pass this flag if evaluating an OLMo model and `olmo` isn't in the model name.")
parser.add_argument("--experiments", type=str, nargs="+", default=None, help="Experiments to run, e.g., '--experiments mmlu_5shot gsm_cot'")
parser.add_argument("--batch_size_reduction", type=int, default=None, help="Reduce batch size by this factor.")
parser.add_argument("--gpu_multiplier", type=int, default=None, help="Multiply the number of GPUs by this factor.")
parser.add_argument("--gsm_stop_at_double_newline", action="store_true", help="Stop GSM generation at the first double newline.")
parser.add_argument("--no-nfs", action="store_true", help="Don't mount the NFS.")
parser.add_argument("--add_stop_sequence", type=str, nargs="+", default=[], help="Additional stop sequences to use when generating completions.") # e.g. \"<|eot_id|>\" for llama 3
parser.add_argument("--upload_to_hf", type=str, default=None, help="If given, upload the eval results to the Hugging Face model hub. Provide the HF dataset and path in form <hf dataset>//<hf path>.")
parser.add_argument("--hf_upload_experiments", type=str, nargs="*", default=None, help="Upload given experiment to the Hugging Face model hub.")
parser.add_argument("--run_oe_eval_experiments", action="store_true", help="Run the OE eval tool and experiments too.")
parser.add_argument("--run_safety_evaluations", action="store_true", help="Run the OE safety evaluations too.")
parser.add_argument("--skip_oi_evals", action="store_true", help="Don't run open instruct evals.")
parser.add_argument("--oe_eval_max_length", type=int, default=4096, help="Max length for OE eval.")
parser.add_argument("--oe_eval_unseen_evals", action="store_true", help="Run unseen task evals instead of dev task evals on OE Eval.")
parser.add_argument("--use_alternate_safety_image", type=str, default=None, help="Use a different image for safety eval.")
parser.add_argument("--evaluate_on_weka", action="store_true", help="Evaluate OE eval on Beaker.")
parser.add_argument("--oe_eval_tasks", type=str, default=None, help="Evaluate OE eval on Beaker.")
args = parser.parse_args()


workspace = args.workspace
model_type = "vanilla_lm" if not args.is_tuned else "tuned_lm"

with open("configs/beaker_configs/default_eval.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

cluster = args.cluster
if cluster[0] == "all":
    cluster = []  # empty list means all clusters
d1['tasks'][0]['constraints']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = args.priority
d1['tasks'][0]['context']['preemptible'] = args.preemptible
d1['tasks'][0]['resources']['gpuCount'] = 1

# remove nfs if asked or jupiter in cluster list.
nfs_available = False
weka_available = False
if all(c in NFS_CLUSTERS for c in cluster):
    d1['tasks'][0]['datasets'].append({
        'mountPath': "/net/nfs.cirrascale",
        "source": {
            "hostPath": "/net/nfs.cirrascale"
        }
    })
    nfs_available = True
elif all(c in WEKA_CLUSTERS for c in cluster):
    d1['tasks'][0]['datasets'].append({
        'mountPath': "/weka/oe-adapt-default",
        "source": {
            "weka": "oe-adapt-default"
        }
    })
    weka_available = True


# Use a different image if requested.
if args.beaker_image is not None:
    d1['tasks'][0]['image']['beaker'] = args.beaker_image

# modify here, or use "--experiments", for different set of experiments
experiment_groups_default = [
    "mmlu_0shot",
    "mmlu_5shot",
    "gsm_direct",
    "gsm_cot",
    "MATH_cot",
    "bbh_direct",
    "bbh_cot",
    "tydiqa_goldp_1shot",
    "tydiqa_no_context_1shot",
    "codex_eval_temp_0.1",
    "codex_eval_temp_0.8",
    "codex_evalplus_temp_0.1",
    "codex_evalplus_temp_0.8",
    "mbpp_evalplus_temp_0.1",
    "mbpp_evalplus_temp_0.8",
    "ifeval",
    "truthfulqa",
    "toxigen",
    "xstest",
    "alpaca_eval",
    "alpaca_eval_2",
]
experiment_groups = args.experiments or experiment_groups_default

# format: model name, their beaker id, checkpoint subfolder, tuned or base.
# or: name, path, None, tuned or base
model_info = (args.model_name, args.location, args.beaker_subfolder, model_type)

# --------------- experiments about number of supervision tasks -------------------------

# for experiment_group, model_info in itertools.product(experiment_groups, models):

d = copy.deepcopy(d1)
model_name = model_info[0] + f"_{model_info[2]}" if model_info[2] is not None else model_info[0]

# Create tasks for each evaluation.
eval_task_specs = []

for experiment_group in experiment_groups:
    print(f"Submitting {experiment_group} for model: {model_info[0]}")
    task_spec = copy.deepcopy(d1["tasks"][0])

    name = f"open_instruct_eval_{experiment_group}_{model_name}_{today}"
    task_spec['name'] = name

    if experiment_group == "mmlu_0shot":
        task_spec['arguments'][0] = '''
            python -m eval.mmlu.run_eval \
            --ntrain 0 \
            --data_dir /data/mmlu/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "mmlu_5shot":
        task_spec['arguments'][0] = '''
            python -m eval.mmlu.run_eval \
            --ntrain 5 \
            --data_dir /data/mmlu/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "bbh_direct":
        task_spec['arguments'][0] = '''
            python -m eval.bbh.run_eval \
            --data_dir /data/bbh \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --max_num_examples_per_task 40 \
            --no_cot \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "bbh_cot":
        task_spec['arguments'][0] = '''
            python -m eval.bbh.run_eval \
            --data_dir /data/bbh \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --max_num_examples_per_task 40 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "gsm_direct":
        task_spec['arguments'][0] = '''
            python -m eval.gsm.run_eval \
            --data_dir /data/gsm/ \
            --max_num_examples 200 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --n_shot 8 \
            --no_cot \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
        if args.gsm_stop_at_double_newline:
            # We need to final backslash in the command above so that there isn't a
            # newline between this argument and the prior part of the command.
            task_spec['arguments'][0] += " --stop_at_double_newline"
    elif experiment_group == "gsm_cot":
        task_spec['arguments'][0] = '''
            python -m eval.gsm.run_eval \
            --data_dir /data/gsm/ \
            --max_num_examples 200 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --n_shot 8 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        ''' 
        if args.gsm_stop_at_double_newline:
            task_spec['arguments'][0] += " --stop_at_double_newline"
    elif experiment_group == "MATH_cot":
        task_spec['arguments'][0] = '''
            python -m eval.MATH.run_eval \
            --data_dir /data/MATH/ \
            --max_num_examples 200 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --n_shot 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "tydiqa_goldp_1shot":
        task_spec["arguments"][0] = '''
            python -m eval.tydiqa.run_eval \
            --data_dir /data/tydiqa/ \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "tydiqa_no_context_1shot":
        task_spec["arguments"][0] = '''
            python -m eval.tydiqa.run_eval \
            --data_dir /data/tydiqa/ \
            --no_context \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "codex_eval_temp_0.1":
        task_spec['arguments'][0] = '''
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.1 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
        '''
    elif experiment_group == "codex_eval_temp_0.8":
        task_spec['arguments'][0] = '''
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.8 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
        '''
    elif experiment_group == "codex_evalplus_temp_0.1":
        task_spec['arguments'][0] = '''
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEvalPlus-OriginFmt.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.1 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "codex_evalplus_temp_0.8":
        task_spec['arguments'][0] = '''
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEvalPlus-OriginFmt.jsonl.gz \
            --data_file_hep data/eval/codex_humaneval/humanevalpack.jsonl  \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.8 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "mbpp_evalplus_temp_0.1":
        task_spec['arguments'][0] = '''
            HF_ALLOW_CODE_EVAL=1 python -m eval.mbpp.run_eval \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.1 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "mbpp_evalplus_temp_0.8":
        task_spec['arguments'][0] = '''
            HF_ALLOW_CODE_EVAL=1 python -m eval.mbpp.run_eval \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.8 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
        '''
    elif experiment_group == "ifeval":
        task_spec['arguments'][0] = '''
            python -m eval.ifeval.run_eval \
                --data_dir /data/ifeval/ \
                --save_dir /output/ \
                --model_name_or_path /model \
                --tokenizer_name_or_path /model \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
                --use_vllm \
        '''
    elif experiment_group == "truthfulqa":
        task_spec['arguments'][0] = '''
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
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "toxigen":
        task_spec['arguments'][0] = '''
        python -m eval.toxigen.run_eval \
            --data_dir /data/toxigen/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 32 \
            --use_vllm \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "xstest":
        task_spec['arguments'][0] = '''
        python -m eval.xstest.run_eval \
            --data_dir /data/xstest/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 32 \
            --use_vllm \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "alpaca_eval":
        task_spec['arguments'][0] = '''
        IS_ALPACA_EVAL_2=False python -m eval.alpaca_farm.run_eval \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --save_dir /output/ \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
    elif experiment_group == "alpaca_eval_2":
        task_spec['arguments'][0] = '''
        IS_ALPACA_EVAL_2=True python -m eval.alpaca_farm.run_eval \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --save_dir /output/ \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        '''
        # OLMo models can only output 2048 new tokens at most; default is 8192.
        if "olmo" in model_info[0] or args.olmo:
            task_spec['arguments'][0] += " --max_new_tokens 4096" # nol increased hardcode to 4096

    else:
        raise ValueError("experiment_group not supported")

    if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--model_name_or_path /model", f"--model_name_or_path {model_info[1]} --hf_revision {args.hf_revision}")]
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--tokenizer_name_or_path /model", f"--tokenizer_name_or_path {model_info[1]}")]
    elif model_info[1].startswith("/"):  # if it's a local model, load it from the local directory
        assert nfs_available or weka_available, "NFS / Weka is required for path-based models."  # to be safe.
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--model_name_or_path /model", f"--model_name_or_path {model_info[1]}")]
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--tokenizer_name_or_path /model", f"--tokenizer_name_or_path {model_info[1]}")]
    else:  # if it's a beaker model, mount the beaker dataset to `/model`
        task_spec['datasets'][1]['source']['beaker'] = model_info[1]

    # if a specific checkpoint is specified, load model from that checkpoint
    if model_info[2] is not None:
        # extract existing model path
        model_name_or_path = re.search("--model_name_or_path (\S+)", task_spec['arguments'][0]).group(1)
        # replace the model path with the checkpoint subfolder.
        task_spec['arguments'] = [task_spec['arguments'][0].replace(model_name_or_path, model_name_or_path+"/"+model_info[2], 1)]
        # NOTE: We don't change the tokenizer subfolder, because by default the
        # tokenizer is only saved to the top-level output dir. That's why we call
        # `str.replace(..., 1)` above; this only replaces the first match.

    # for vanilla_lm, remove the chat formatting function
    if model_info[3] == "vanilla_lm":
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--use_chat_format", "")]

    # Adjust batch size and gpus.
    task_spec = adjust_batch_size(
        task_spec=task_spec,
        model_name=model_info[0],
        batch_size_reduction=args.batch_size_reduction,
    )
    task_spec = adjust_gpus(
        task_spec=task_spec,
        experiment_group=experiment_group,
        model_name=model_info[0],
        gpu_multiplier=args.gpu_multiplier,
    )

    # if using huggingface tokenizer template, replace the chat formatting function with hf tokenizer one
    # otherwise, try to guess what template to use based on model name
    if args.use_hf_tokenizer_template:
        task_spec['arguments'] = [task_spec['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template")
        ]
    elif "llama2-chat" in model_info[0]:
        task_spec['arguments'] = [task_spec['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
        ]
    elif "code_llama_instruct" in model_info[0]:
        task_spec['arguments'] = [task_spec['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
        ]
    elif "zephyr" in model_info[0]:
        task_spec['arguments'] = [task_spec['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_zephyr_chat_format")
        ]
    elif "xwin" in model_info[0]:
        task_spec['arguments'] = [task_spec['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_xwin_chat_format")
        ]
    elif "olmo" in model_info[0] or args.olmo:
        task_spec['arguments'] = [task_spec['arguments'][0].replace(
            "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
            "--chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format")
        ]

    if any([x in model_info[0] for x in ["opt", "pythia", "falcon", "olmoe"]]):
        if "--use_vllm" in task_spec['arguments'][0]:
            print(f"Removing --use_vllm for {model_info[0]}")
            task_spec['arguments'] = [task_spec['arguments'][0].replace("--use_vllm", "")] 

    # Add additional stop sequences if needed.
    # mainly for llama-3-instruct eot.
    tasks_without_addition_stop = ["mmlu_0shot", "mmlu_5shot", "truthfulqa"]
    if args.add_stop_sequence and experiment_group not in tasks_without_addition_stop:
        task_spec['arguments'] = [task_spec['arguments'][0] + " --additional_stop_sequence " + " ".join(args.add_stop_sequence)]

    # add HF hub upload if specified
    if args.upload_to_hf:
        if args.hf_upload_experiments is None or len(args.hf_upload_experiments) == 0:
            # by default, we dont upload oi-evals, only safety and oe-evals.
            args.hf_upload_experiments = []
        if experiment_group not in args.hf_upload_experiments:
            print(f"Skipping HF upload for {experiment_group}")
        else:
            hf_dataset = args.upload_to_hf
            # to match the way oe-eval script works.
            # if we prepended hf- to the model name, remove it.
            if model_name.startswith("hf-"):
                model_name = model_name[3:]
            task_spec['arguments'] = [task_spec['arguments'][0] + f" --upload_to_hf {hf_dataset} --hf_upload_name results/{model_name}"]

    eval_task_specs.append(task_spec)


# Create an experiment that runs all the eval tasks.

if not args.skip_oi_evals:
    experiment_name = f"open_instruct_eval_{model_name}_{today}" 
    d["description"] = experiment_name
    d["tasks"] = eval_task_specs
    # if configs/beaker_configs/auto_created doesn't exist, create it with os
    if not os.path.exists("configs/beaker_configs/auto_created"):
        os.makedirs("configs/beaker_configs/auto_created")
    fn = "configs/beaker_configs/auto_created/{}.yaml".format(experiment_name)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as file:
        yaml.dump(d, file, default_flow_style=True)

    cmd = "beaker experiment create {} --workspace ai2/{}".format(fn, workspace)
    subprocess.Popen(cmd, shell=True)

if args.run_oe_eval_experiments or args.oe_eval_unseen_evals:
    # if so, run oe-eval. We assume it is cloned in the top-level repo directory.
    oe_eval_cmd = f"scripts/eval/oe-eval.sh --model-name {model_name}"
    if args.upload_to_hf:
        oe_eval_cmd += " --hf-upload"
    ## model location munging: if beaker, use beaker://. If hf, just name
    if model_info[0].startswith("hf-"):
        oe_eval_cmd += f" --model-location {model_info[1]}"
    elif model_info[1].startswith("/"):
        oe_eval_cmd += f" --model-location {model_info[1]}"
    else:
        oe_eval_cmd += f" --model-location beaker://{model_info[1]}"
    if args.hf_revision:
        oe_eval_cmd += f" --revision {args.hf_revision}"
    if args.evaluate_on_weka:
        oe_eval_cmd += " --evaluate_on_weka"
    if args.oe_eval_tasks:
        oe_eval_cmd += f" --tasks {args.oe_eval_tasks}"
    # add string with number of gpus
    num_gpus = task_spec['resources']['gpuCount']
    # if num_gpus > 1, double it again for oe-eval configs
    # open_instruct GPT adjustment wasn't quite enough
    # adjusted here so the GPU configs in open-instruct eval are not impacted by the change
    # tested reasonably extensively with 70B
    if num_gpus > 1:
        num_gpus *= 2
    oe_eval_cmd += f" --num_gpus {num_gpus}"
    if args.oe_eval_max_length:
        oe_eval_cmd += f" --max-length {args.oe_eval_max_length}"
    if args.oe_eval_unseen_evals:
        oe_eval_cmd += " --unseen-evals"
    # add priority
    oe_eval_cmd += f" --priority {args.priority}"
    print(f"Running OE eval with command: {oe_eval_cmd}")
    subprocess.Popen(oe_eval_cmd, shell=True)

# create an experiment that runs the safety eval tasks
if args.run_safety_evaluations:
    # just take the original spec we had, modify it for safety eval.
    experiment_name = f"oi_safety_{model_name}"
    experiment_name = experiment_name.replace('β', '').replace(r"{", "").replace(r"}", "") # hack: remove characters beaker doesn't like
    d["description"] = experiment_name
    # specific image for safety eval
    d["tasks"][0]["image"]["beaker"] = "hamishivi/open-safety"
    if args.use_alternate_safety_image:
        d["tasks"][0]["image"]["beaker"] = args.use_alternate_safety_image
    d["tasks"] = [d["tasks"][0]]
    task_spec = d["tasks"][0]
    task_spec["name"] = experiment_name
    task_spec["arguments"][0] = f'''
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=. python evaluation/run_all_generation_benchmarks.py \
    --model_name_or_path /model \
    --model_input_template_path_or_name hf \
    --report_output_path /output/metrics.json \
    --save_individual_results_path /output/all.json \
'''
    # some copied logic
    if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--model_name_or_path /model", f"--model_name_or_path {model_info[1]} --hf_revision {args.hf_revision}")]
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--tokenizer_name_or_path /model", f"--tokenizer_name_or_path {model_info[1]}")]
    elif model_info[1].startswith("/"):  # if it's a local model, load it from the local directory
        assert nfs_available or weka_available, "NFS / Weka is required for path-based models."  # to be safe.
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
        task_spec['arguments'] = [task_spec['arguments'][0].replace("--tokenizer_name_or_path /model", "--tokenizer_name_or_path "+model_info[1])]
    else:  # if it's a beaker model, mount the beaker dataset to `/model`
        task_spec['datasets'][1]['source']['beaker'] = model_info[1]

    task_spec = adjust_gpus(
        task_spec=task_spec,
        experiment_group="safety_eval",
        model_name=model_info[0],
        gpu_multiplier=args.gpu_multiplier,
    )

    # add gpu information.
    # we just assume you want to use all the gpus for one task at a time
    if "70B" in model_info[0]:
        task_spec['resources']['gpuCount'] = 8
    num_gpus = task_spec['resources']['gpuCount']
    task_spec["arguments"][0]+= f" --min_gpus_per_task {num_gpus}"

    if args.upload_to_hf:
        hf_dataset = args.upload_to_hf
        # to match the way oe-eval script works.
        # if we prepended hf- to the model name, remove it.
        # if model_name.startswith("hf-"):
        #     model_name = model_name[3:]
        # Above is no longer the case, oe-eval includes hf- again
        task_spec['arguments'] = [task_spec['arguments'][0] + f" --upload_to_hf {hf_dataset} --hf_upload_name results/{model_name}"]

    d["tasks"] = [task_spec]
    if not os.path.exists("configs/beaker_configs/auto_created"):
        os.makedirs("configs/beaker_configs/auto_created")
    fn = "configs/beaker_configs/auto_created/{}.yaml".format(experiment_name)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as file:
        yaml.dump(d, file, default_flow_style=True)

    cmd = "beaker experiment create {} --workspace ai2/{}".format(fn, workspace)
    subprocess.Popen(cmd, shell=True)
