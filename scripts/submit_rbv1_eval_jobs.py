# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import subprocess
from datetime import date

import yaml

# Create argparse, for store_true variables of eval_on_pref_sets and eval_on_bon
# String image for Beaker image
# Bool default true for upload_to_hub
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--eval_on_pref_sets", action="store_true", default=False, help="Evaluate on preference sets rather than core set"
)
argparser.add_argument("--eval_on_bon", action="store_true", default=False, help="Evaluate on BON preference sets")
argparser.add_argument("--image", type=str, default="nathanl/rewardbench_auto", help="Beaker image to use")
argparser.add_argument("--cluster", type=str, default="ai2/saturn-cirrascale", help="Beaker cluster to use")
argparser.add_argument("--priority", type=str, default="normal", help="Priority of the job")
argparser.add_argument("--upload_to_hub", action="store_false", default=True, help="Upload to results to HF hub")
argparser.add_argument("--model", type=str, default=None, help="Specific model to evaluate if not sweep")
argparser.add_argument("--revision", type=str, default=None, help="Specific model to evaluate if not sweep")
argparser.add_argument(
    "--ref_free", action="store_true", default=False, help="If true, runs DPO models without reference"
)
argparser.add_argument(
    "--eval_dpo_only", action="store_true", default=False, help="If true, only evaluates DPO models"
)
argparser.add_argument("--eval_rm_only", action="store_true", default=False, help="If true, only evaluates RM models")
args = argparser.parse_args()

# assert that only one of eval_dpo_only and eval_rm_only is True at a time
assert not (args.eval_dpo_only and args.eval_rm_only), "Only one of eval_dpo_only and eval_rm_only can be True"

today = date.today().strftime("%m%d%Y")

with open("scripts/configs/beaker_eval.yaml", "r") as f:
    d1 = yaml.load(f.read(), Loader=yaml.FullLoader)

cluster = args.cluster

image = args.image
num_gpus = 1
upload_to_hub = args.upload_to_hub
eval_on_pref_sets = args.eval_on_pref_sets
eval_on_bon = args.eval_on_bon

if eval_on_bon:
    with open("scripts/configs/eval_bon_configs.yaml", "r") as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
else:
    with open("scripts/configs/eval_configs.yaml", "r") as f:
        configs = yaml.load(f.read(), Loader=yaml.FullLoader)
print(configs)


# assert only one of eval_on_pref_sets and eval_on_bon is True
assert not (eval_on_pref_sets and eval_on_bon), "Only one of eval_on_pref_sets and eval_on_bon can be True"

d1["tasks"][0]["image"]["beaker"] = image
d1["tasks"][0]["context"]["cluster"] = cluster
d1["tasks"][0]["context"]["priority"] = args.priority
d1["tasks"][0]["resources"]["gpuCount"] = num_gpus

# get model from config keys
models_to_evaluate = list(configs.keys())

if args.model is not None:
    if args.model in models_to_evaluate:
        models_to_evaluate = [args.model]
    else:
        raise ValueError(f"Model {args.model} not found in configs")

for model in models_to_evaluate:
    model_config = configs[model]
    eval_dpo = model_config["dpo"]

    # check if generative in model_config
    if "generative" in model_config:
        if model_config["generative"]:
            eval_gen = True
    else:
        eval_gen = False

    # check if bfloat16
    if "torch_dtype" in model_config:
        if model_config["torch_dtype"] == "torch.bfloat16" or model_config["torch_dtype"] == "bfloat16":
            eval_bfloat16 = True
    else:
        eval_bfloat16 = False

    # ignore models depending on eval_dpo_only and eval_rm_only
    if args.eval_dpo_only:
        if not eval_dpo:
            continue
    elif args.eval_rm_only:
        if eval_dpo:
            continue

    if eval_on_bon:
        experiment_group = "rewardebench-bon"
        script = "run_v2.py"
    elif eval_dpo:
        experiment_group = "rewardebench-dpo"
        script = "run_dpo.py"
    elif eval_gen:
        experiment_group = "rewardebench-gen"
        script = "run_generative.py"
    else:
        experiment_group = "rewardebench-seq"
        script = "run_rm.py"

    # log experiment name
    if eval_on_pref_sets:
        experiment_group += "-pref-sets"

    print(f"Submitting evaluation for model: {model} on {experiment_group}")
    d = copy.deepcopy(d1)

    name = f"rewardbench_eval_for_{model}_on_{experiment_group}".replace("/", "-")
    d["description"] = name
    d["tasks"][0]["name"] = name

    if "num_gpus" in model_config:
        d["tasks"][0]["resources"]["gpuCount"] = model_config["num_gpus"]

    if not eval_gen:
        d["tasks"][0]["arguments"][0] = (
            f"python scripts/{script}"
            f" --model {model}"
            f" --tokenizer {model_config['tokenizer']}"
            f" --revision {args.revision}"
            f" --batch_size {model_config['batch_size']}"
        )
    else:
        d["tasks"][0]["arguments"][0] = (
            f"python scripts/{script}" f" --model {model}" f" --num_gpus {model_config['num_gpus']}"
        )
    if model_config["chat_template"] is not None:
        d["tasks"][0]["arguments"][0] += f" --chat_template {model_config['chat_template']}"
    if model_config["trust_remote_code"]:
        d["tasks"][0]["arguments"][0] += " --trust_remote_code"
    if not upload_to_hub:
        d["tasks"][0]["arguments"][0] += " --do_not_save"
    if eval_on_pref_sets:
        d["tasks"][0]["arguments"][0] += " --pref_sets"
    if eval_bfloat16:
        d["tasks"][0]["arguments"][0] += " --torch_dtype=bfloat16"
    if model_config["quantized"] is not None:
        if not model_config["quantized"] and not eval_dpo:
            d["tasks"][0]["arguments"][0] += " --not_quantized"

    # for run_rm only, for now, and gemma-2-27b RMs
    if "attention_implementation" in model_config:
        d["tasks"][0]["arguments"][0] += f" --attn_implementation {model_config['attention_implementation']}"

    if "ref_model" in model_config:
        if not args.ref_free:  # if passed, ignore logic in eval configs
            d["tasks"][0]["arguments"][0] += f" --ref_model {model_config['ref_model']}"
    if "max_length" in model_config:  # for `mightbe/Better-PairRM`, but could come up in the future
        d["tasks"][0]["arguments"][0] += f" --max_length {model_config['max_length']}"

    # use os to check if beaker_configs/auto_created exists
    if not os.path.exists("beaker_configs/auto_created"):
        os.makedirs("beaker_configs/auto_created")

    fn = "beaker_configs/auto_created/{}.yaml".format(name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/reward-bench-v2".format(fn)
    subprocess.Popen(cmd, shell=True)
