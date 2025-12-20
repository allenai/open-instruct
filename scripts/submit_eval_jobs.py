import argparse
import copy
import os
import re
import subprocess
from datetime import date

import yaml

from open_instruct import utils

########################################

# Helper functions.


def adjust_batch_size(task_spec, model_name, batch_size_reduction):
    "Adjust batch size using heuristics that are good for A100-size GPUs."
    reduce_by_2 = ["13B"]
    reduce_by_4 = ["30B", "32B", "34B", "40B", "65B", "70B", "70b", "72B", "72b"]
    # If not given, choose a value based on the model name.
    if batch_size_reduction is None:
        if any([pattern in model_name for pattern in reduce_by_2]):
            batch_size_reduction = 2
        elif any([pattern in model_name for pattern in reduce_by_4]):
            batch_size_reduction = 4
        else:
            batch_size_reduction = 1

    # Reduce accordingly.
    if "--eval_batch_size" in task_spec["arguments"][0]:
        original_batch_size = re.search(r"--eval_batch_size (\d+)", task_spec["arguments"][0]).group(1)
        new_batch_size = max(1, int(original_batch_size) // batch_size_reduction)
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                f"--eval_batch_size {original_batch_size}", f"--eval_batch_size {new_batch_size}"
            )
        ]

    return task_spec


def adjust_gpus(task_spec, experiment_group, model_name, gpu_multiplier):
    "Adjust GPU count using heuristics that are good for A100-size GPUs."
    medium = ["30B", "34B"]
    large = ["32B", "40B", "65B", "70B", "70b", "72B", "72b"]
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
        task_spec["resources"]["gpuCount"] = codex_multiplier * task_spec["resources"]["gpuCount"]
    else:
        task_spec["resources"]["gpuCount"] = default_multiplier * task_spec["resources"]["gpuCount"]

    return task_spec


########################################
# Launcher

today = date.today().strftime("%m%d%Y")

parser = argparse.ArgumentParser()
# Require fully-qualified workspace or use a fully-qualified default.
parser.add_argument("--workspace", type=str, default="ai2/tulu-3-results")
parser.add_argument("--model_name", type=str, default="hf-opt-7B")
parser.add_argument("--hf_revision", type=str, default=None)
parser.add_argument("--location", type=str, default=None)
parser.add_argument(
    "--beaker_image", type=str, default="oe-eval-beaker/oe_eval_auto", help="If given, use this Beaker image."
)
# image refernece: https://github.com/allenai/oe-eval-internal/blob/493660aca07d05384c6bd1860c4180860ccc7d53/oe_eval_internal/utilities/launch_utils.py#L143
# image: https://legacy.beaker.org/im/01JRZWRN4FSGK7FWKV1DRPP1R1/details
parser.add_argument("--beaker_subfolder", type=str, default=None)
parser.add_argument("--cluster", nargs="+", default=["ai2/jupiter", "ai2/saturn", "ai2/ceres", "ai2/neptune"])
parser.add_argument("--is_tuned", action="store_true")
parser.add_argument("--use_hf_tokenizer_template", action="store_true")
parser.add_argument("--priority", type=str, default="low")
parser.add_argument(
    "--preemptible", action="store_true", default=False, help="for using preemtipble jobs (required on some instances)"
)
parser.add_argument(
    "--olmo",
    action="store_true",
    help="Pass this flag if evaluating an OLMo model and `olmo` isn't in the model name.",
)
parser.add_argument(
    "--experiments",
    type=str,
    nargs="+",
    default=None,
    help="Experiments to run, e.g., '--experiments mmlu_5shot gsm_cot'",
)
parser.add_argument("--batch_size_reduction", type=int, default=None, help="Reduce batch size by this factor.")
parser.add_argument("--gpu_multiplier", type=int, default=None, help="Multiply the number of GPUs by this factor.")
parser.add_argument(
    "--gsm_stop_at_double_newline", action="store_true", help="Stop GSM generation at the first double newline."
)
parser.add_argument("--no-nfs", action="store_true", help="Don't mount the NFS.")
parser.add_argument(
    "--add_stop_sequence",
    type=str,
    nargs="+",
    default=[],
    help="Additional stop sequences to use when generating completions.",
)  # e.g. \"<|eot_id|>\" for llama 3
parser.add_argument(
    "--upload_to_hf",
    type=str,
    default=None,
    help="If given, upload the eval results to the Hugging Face model hub. Provide the HF dataset and path in form <hf dataset>//<hf path>.",
)
parser.add_argument(
    "--hf_upload_experiments",
    type=str,
    nargs="*",
    default=None,
    help="Upload given experiment to the Hugging Face model hub.",
)
parser.add_argument("--run_oe_eval_experiments", action="store_true", help="Run the OE eval tool and experiments too.")
parser.add_argument("--skip_oi_evals", action="store_true", help="Don't run open instruct evals.")
parser.add_argument("--oe_eval_max_length", type=int, default=4096, help="Max length for OE eval.")
parser.add_argument(
    "--oe_eval_task_suite",
    type=str,
    default="NEXT_MODEL_DEV",
    help="Task suite for OE eval: NEXT_MODEL_DEV, NEXT_MODEL_UNSEEN, TULU_3_DEV, TULU_3_UNSEEN, SAFETY_EVAL, SAFETY_EVAL_REASONING (default: NEXT_MODEL_DEV)",
)
parser.add_argument("--evaluate_on_weka", action="store_true", help="Evaluate OE eval on Beaker.")
# NOTE: evaluate on weka is expected to be on by default. If not, the evals will run on the google augusta cluster.
# TODO: fix this logic at a future date

parser.add_argument("--oe_eval_tasks", type=str, default=None, help="Evaluate OE eval on Beaker.")
parser.add_argument("--step", type=int, default=None, help="Step number for postgresql logging.")
parser.add_argument("--run_id", type=str, default=None, help="A unique run ID for postgresql logging.")
parser.add_argument("--wandb_run_path", type=str, default=None, help="A unique run ID for postgresql logging.")
parser.add_argument(
    "--oe_eval_stop_sequences", type=str, default=None, help="Comma-separated list of stop sequences for OE eval."
)
parser.add_argument(
    "--process_output",
    type=str,
    default="r1_style",
    help="Process output type for OE eval (e.g., 'r1_style'). Defaults to 'r1_style', which should work for most of our models, including non-thinking ones.",
)
args = parser.parse_args()


workspace = args.workspace
# Do not silently alter the workspace; require fully-qualified form.
if len(workspace.split("/")) != 2 or not all(workspace.split("/")):
    raise ValueError(
        f"--workspace must be fully qualified as '<org>/<workspace>' (e.g., 'ai2/oe-adapt-general'). Received: '{workspace}'"
    )
model_type = "vanilla_lm" if not args.is_tuned else "tuned_lm"

with open("configs/beaker_configs/default_eval.yaml") as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

cluster = args.cluster
if cluster[0] == "all":
    cluster = []  # empty list means all clusters
d1["tasks"][0]["constraints"]["cluster"] = cluster
d1["tasks"][0]["context"]["priority"] = args.priority
d1["tasks"][0]["context"]["preemptible"] = args.preemptible
d1["tasks"][0]["resources"]["gpuCount"] = 1

# remove nfs if asked or jupiter in cluster list.
weka_available = False
if all(c in utils.WEKA_CLUSTERS for c in cluster):
    d1["tasks"][0]["datasets"].append({"mountPath": "/weka/oe-adapt-default", "source": {"weka": "oe-adapt-default"}})
    d1["tasks"][0]["datasets"].append(
        {"mountPath": "/weka/oe-training-default", "source": {"weka": "oe-training-default"}}
    )
    weka_available = True


# NOTE: Remove in the future, not sure if this effects oe-eval jobs
# Use a different image if requested.
if args.beaker_image is not None:
    d1["tasks"][0]["image"]["beaker"] = args.beaker_image

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
    if not args.skip_oi_evals:
        print(f"Submitting {experiment_group} for model: {model_info[0]}")
    task_spec = copy.deepcopy(d1["tasks"][0])

    name = f"open_instruct_eval_{experiment_group}_{model_name}_{today}"
    task_spec["name"] = name

    if experiment_group == "mmlu_0shot":
        task_spec["arguments"][0] = """
            python -m eval.mmlu.run_eval \
            --ntrain 0 \
            --data_dir /data/mmlu/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        """
    elif experiment_group == "mmlu_5shot":
        task_spec["arguments"][0] = """
            python -m eval.mmlu.run_eval \
            --ntrain 5 \
            --data_dir /data/mmlu/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 4 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        """
    elif experiment_group == "bbh_direct":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "bbh_cot":
        task_spec["arguments"][0] = """
            python -m eval.bbh.run_eval \
            --data_dir /data/bbh \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --max_num_examples_per_task 40 \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        """
    elif experiment_group == "gsm_direct":
        task_spec["arguments"][0] = """
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
        """
        if args.gsm_stop_at_double_newline:
            # We need to final backslash in the command above so that there isn't a
            # newline between this argument and the prior part of the command.
            task_spec["arguments"][0] += " --stop_at_double_newline"
    elif experiment_group == "gsm_cot":
        task_spec["arguments"][0] = """
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
        """
        if args.gsm_stop_at_double_newline:
            task_spec["arguments"][0] += " --stop_at_double_newline"
    elif experiment_group == "MATH_cot":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "tydiqa_goldp_1shot":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "tydiqa_no_context_1shot":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "codex_eval_temp_0.1":
        task_spec["arguments"][0] = """
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.1 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
        """
    elif experiment_group == "codex_eval_temp_0.8":
        task_spec["arguments"][0] = """
            python -m eval.codex_humaneval.run_eval \
            --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
            --eval_pass_at_ks 1 5 10 20 \
            --unbiased_sampling_size_n 20 \
            --temperature 0.8 \
            --save_dir /output/ \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
        """
    elif experiment_group == "codex_evalplus_temp_0.1":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "codex_evalplus_temp_0.8":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "mbpp_evalplus_temp_0.1":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "mbpp_evalplus_temp_0.8":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "ifeval":
        task_spec["arguments"][0] = """
            python -m eval.ifeval.run_eval \
                --data_dir /data/ifeval/ \
                --save_dir /output/ \
                --model_name_or_path /model \
                --tokenizer_name_or_path /model \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
                --use_vllm \
        """
    elif experiment_group == "truthfulqa":
        task_spec["arguments"][0] = """
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
        """
    elif experiment_group == "toxigen":
        task_spec["arguments"][0] = """
        python -m eval.toxigen.run_eval \
            --data_dir /data/toxigen/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 32 \
            --use_vllm \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        """
    elif experiment_group == "xstest":
        task_spec["arguments"][0] = """
        python -m eval.xstest.run_eval \
            --data_dir /data/xstest/ \
            --save_dir /output/ \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --eval_batch_size 32 \
            --use_vllm \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        """
    elif experiment_group == "alpaca_eval":
        task_spec["arguments"][0] = """
        IS_ALPACA_EVAL_2=False python -m eval.alpaca_farm.run_eval \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --save_dir /output/ \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        """
    elif experiment_group == "alpaca_eval_2":
        task_spec["arguments"][0] = """
        IS_ALPACA_EVAL_2=True python -m eval.alpaca_farm.run_eval \
            --use_vllm \
            --model_name_or_path /model \
            --tokenizer_name_or_path /model \
            --save_dir /output/ \
            --use_chat_format \
            --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
        """
        # OLMo models can only output 2048 new tokens at most; default is 8192.
        if "olmo" in model_info[0] or args.olmo:
            task_spec["arguments"][0] += " --max_new_tokens 4096"  # nol increased hardcode to 4096

    else:
        raise ValueError("experiment_group not supported")

    if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                "--model_name_or_path /model", f"--model_name_or_path {model_info[1]} --hf_revision {args.hf_revision}"
            )
        ]
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                "--tokenizer_name_or_path /model", f"--tokenizer_name_or_path {model_info[1]}"
            )
        ]
    elif model_info[1].startswith("/"):  # if it's a local model, load it from the local directory
        assert weka_available, "NFS / Weka is required for path-based models."  # to be safe.
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace("--model_name_or_path /model", f"--model_name_or_path {model_info[1]}")
        ]
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                "--tokenizer_name_or_path /model", f"--tokenizer_name_or_path {model_info[1]}"
            )
        ]
    else:  # if it's a beaker model, mount the beaker dataset to `/model`
        task_spec["datasets"][1]["source"]["beaker"] = model_info[1]

    # if a specific checkpoint is specified, load model from that checkpoint
    if model_info[2] is not None:
        # extract existing model path
        model_name_or_path = re.search(r"--model_name_or_path (\S+)", task_spec["arguments"][0]).group(1)
        # replace the model path with the checkpoint subfolder.
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(model_name_or_path, model_name_or_path + "/" + model_info[2], 1)
        ]
        # NOTE: We don't change the tokenizer subfolder, because by default the
        # tokenizer is only saved to the top-level output dir. That's why we call
        # `str.replace(..., 1)` above; this only replaces the first match.

    # for vanilla_lm, remove the chat formatting function
    if model_info[3] == "vanilla_lm":
        task_spec["arguments"] = [task_spec["arguments"][0].replace("--use_chat_format", "")]

    # Adjust batch size and gpus.
    task_spec = adjust_batch_size(
        task_spec=task_spec, model_name=model_info[0], batch_size_reduction=args.batch_size_reduction
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
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
                "--chat_formatting_function eval.templates.create_prompt_with_huggingface_tokenizer_template",
            )
        ]
    elif "llama2-chat" in model_info[0] or "code_llama_instruct" in model_info[0]:
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
                "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format",
            )
        ]
    elif "zephyr" in model_info[0]:
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
                "--chat_formatting_function eval.templates.create_prompt_with_zephyr_chat_format",
            )
        ]
    elif "xwin" in model_info[0]:
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
                "--chat_formatting_function eval.templates.create_prompt_with_xwin_chat_format",
            )
        ]
    elif "olmo" in model_info[0] or args.olmo:
        task_spec["arguments"] = [
            task_spec["arguments"][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format",
                "--chat_formatting_function eval.templates.create_prompt_with_olmo_chat_format",
            )
        ]

    if any([x in model_info[0] for x in ["opt", "pythia", "falcon", "olmoe"]]):
        if "--use_vllm" in task_spec["arguments"][0]:
            if not args.skip_oi_evals:
                print(f"Removing --use_vllm for {model_info[0]}")
            task_spec["arguments"] = [task_spec["arguments"][0].replace("--use_vllm", "")]

    # Add additional stop sequences if needed.
    # mainly for llama-3-instruct eot.
    tasks_without_addition_stop = ["mmlu_0shot", "mmlu_5shot", "truthfulqa"]
    if args.add_stop_sequence and experiment_group not in tasks_without_addition_stop:
        task_spec["arguments"] = [
            task_spec["arguments"][0] + " --additional_stop_sequence " + " ".join(args.add_stop_sequence)
        ]

    # add HF hub upload if specified
    if args.upload_to_hf:
        if args.hf_upload_experiments is None or len(args.hf_upload_experiments) == 0:
            # by default, we dont upload oi-evals, only safety and oe-evals.
            args.hf_upload_experiments = []
        if experiment_group not in args.hf_upload_experiments:
            if not args.skip_oi_evals:
                print(f"Skipping HF upload for {experiment_group}")
        else:
            hf_dataset = args.upload_to_hf
            # to match the way oe-eval script works.
            # if we prepended hf- to the model name, remove it.
            if model_name.startswith("hf-"):
                model_name = model_name[3:]
            task_spec["arguments"] = [
                task_spec["arguments"][0] + f" --upload_to_hf {hf_dataset} --hf_upload_name results/{model_name}"
            ]

    eval_task_specs.append(task_spec)


# Create an experiment that runs all the eval tasks.

model_name = model_name[:100]  # beaker doesn't like names longer than 128 characters, here we save for some headroom.
if not args.skip_oi_evals:
    experiment_name = f"open_instruct_eval_{model_name}_{today}"
    d["description"] = experiment_name
    d["tasks"] = eval_task_specs
    # if configs/beaker_configs/auto_created doesn't exist, create it with os
    if not os.path.exists("configs/beaker_configs/auto_created"):
        os.makedirs("configs/beaker_configs/auto_created")
    fn = f"configs/beaker_configs/auto_created/{experiment_name}.yaml"
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as file:
        yaml.dump(d, file, default_flow_style=True)

    # Do not prefix or rewrite the workspace; use exactly what the user provided.
    cmd = f"beaker experiment create {fn} --workspace {workspace}"
    subprocess.Popen(cmd, shell=True)

if args.run_oe_eval_experiments:
    # if so, run oe-eval. We assume it is cloned in the top-level repo directory.
    oe_eval_cmd = f"scripts/eval/oe-eval.sh --model-name {model_name}"
    if args.upload_to_hf:
        oe_eval_cmd += f" --upload_to_hf {args.upload_to_hf}"
    ## model location munging: if beaker, use beaker://. If hf, just name
    if model_info[0].startswith("hf-") or model_info[1].startswith("/") or model_info[1].startswith("gs://"):
        oe_eval_cmd += f" --model-location {model_info[1]}"
    else:
        oe_eval_cmd += f" --model-location beaker://{model_info[1]}"
    if args.hf_revision:
        oe_eval_cmd += f" --revision {args.hf_revision}"
    # Pass through the beaker workspace explicitly to avoid any hardcoded defaults.
    if workspace:
        oe_eval_cmd += f" --beaker-workspace {workspace}"
    if args.evaluate_on_weka:
        oe_eval_cmd += " --evaluate_on_weka"
    if args.oe_eval_tasks:
        oe_eval_cmd += f" --tasks {args.oe_eval_tasks}"
    if args.run_id:
        oe_eval_cmd += f" --run-id {args.run_id}"
    if args.step:
        oe_eval_cmd += f" --step {args.step}"
    if args.wandb_run_path:
        oe_eval_cmd += f" --wandb-run-path {args.wandb_run_path}"
    # add string with number of gpus
    num_gpus = task_spec["resources"]["gpuCount"]
    # if num_gpus > 1, double it again for oe-eval configs
    # open_instruct GPT adjustment wasn't quite enough
    # adjusted here so the GPU configs in open-instruct eval are not impacted by the change
    # tested reasonably extensively with 70B
    if num_gpus > 1:
        num_gpus *= 2
    # double GPUs for reasoning models
    if args.oe_eval_task_suite == "SAFETY_EVAL_REASONING":
        num_gpus *= 2
    oe_eval_cmd += f" --num_gpus {num_gpus}"

    if args.oe_eval_max_length:
        oe_eval_cmd += f" --max-length {args.oe_eval_max_length}"
    # Add task suite parameter
    if args.oe_eval_task_suite:
        oe_eval_cmd += f" --task-suite {args.oe_eval_task_suite}"
    # add priority
    oe_eval_cmd += f" --priority {args.priority}"

    # Add stop sequences if provided
    if args.oe_eval_stop_sequences:
        oe_eval_cmd += f" --stop-sequences '{args.oe_eval_stop_sequences}'"

    # Add process output if provided
    if args.process_output:
        oe_eval_cmd += f" --process-output {args.process_output}"

    # Add beaker image from existing argument
    if args.beaker_image:
        oe_eval_cmd += f" --beaker-image {args.beaker_image}"

    # Add cluster parameter - use the existing cluster argument
    # Join the list with commas since oe-eval.sh expects a comma-separated string
    if args.cluster and len(args.cluster) > 0:
        cluster_str = ",".join(args.cluster)
        oe_eval_cmd += f" --cluster '{cluster_str}'"

    print(f"Running OE eval with command: {oe_eval_cmd}")
    subprocess.Popen(oe_eval_cmd, shell=True)
