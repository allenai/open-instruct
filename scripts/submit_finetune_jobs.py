import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_finetune.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/general-cirrascale"
# cluster = "ai2/yizhongw-a100-80gb"
cluster = "ai2/allennlp-cirrascale"
num_gpus = 4
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "high"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

# modify here for different set of experiments
experiment_group = "dataset_comparison"
wandb_project = "open_instruct"
wandb_api_key = "Your Wandb API Key"


# ----------------------- dataset comparison -----------------------
if experiment_group == "dataset_comparison":
    datasets = [
        "baize",
        "code_alpaca",
        "cot",
        "dolly",
        "flan_v2",
        "gpt4_alpaca",
        "oasst1",
        "sharegpt",
        "stanford_alpaca",
        "super_ni",
        "self_instruct",
        "unnatural_instructions",
        "combined",
    ]
    model_size = "7B"

    for dataset in datasets:
        d = copy.deepcopy(d1)

        # name and description
        exp_name = f"open_instruct_finetune_{model_size}_{dataset}_{today}"
        d['description'] = exp_name
        d['tasks'][0]['name'] = exp_name

        # model specific
        for mount_dataset in d['tasks'][0]['datasets']:
            if mount_dataset["mountPath"] == "/hf_llama_models":
                mount_dataset["source"]["beaker"] = f"Yizhongw03/hf_llama_model_{model_size}"
        if model_size == "7B":
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--per_device_train_batch_size 2", 
                "--per_device_train_batch_size 2"
            )
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--gradient_accumulation_steps 16",
                f"--gradient_accumulation_steps {128 // 2 // num_gpus}"
            )
        elif model_size == "13B":
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--per_device_train_batch_size 2", 
                "--per_device_train_batch_size 2"
            )
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--gradient_accumulation_steps 16",
                f"--gradient_accumulation_steps {128 // 2 // num_gpus}"
            )
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf",
                "--deepspeed_config_file ds_configs/stage3_offloading_accelerate.conf",
            )
        else:
            raise NotImplementedError


        # dataset specific
        if dataset == "combined":
            combining_datasets = [
                "super_ni",
                "sharegpt",
                "oasst1",
                "dolly",
                "cot",
                "code_alpaca",
            ]
            combining_bash_command = "cat " + " ".join([f"/data/{d}/{d}_data.jsonl" for d in combining_datasets]) + " > /output/combined_data.jsonl"
            d["tasks"][0]["arguments"][0] = combining_bash_command + " && " + d["tasks"][0]["arguments"][0]

            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--train_file /data/alpaca_data_original_template.jsonl", 
                f"--train_file /output/combined_data.jsonl"
            )
        else:
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
                "--train_file /data/alpaca_data_original_template.jsonl", 
                f"--train_file /data/{dataset}/{dataset}_data.jsonl"
            )

        # wandb specific
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--report_to tensorboard",
            "--report_to wandb"
        )
        for env in d['tasks'][0]['envVars']:
            if env['name'] == "WANDB_DISABLED":
                env['value'] = False
            if env['name'] == "WANDB_PROJECT":
                env['value'] = wandb_project
        d['tasks'][0]['envVars'].append({
            'name': 'WANDB_API_KEY', 'value': wandb_api_key
        })
        d['tasks'][0]['envVars'].append({
            'name': 'WANDB_NAME', 'value': exp_name
        })
        d['tasks'][0]['envVars'].append({
            'name': 'WANDB_RUN_GROUP', 'value': experiment_group
        })
        # print(d)

        fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
        subprocess.Popen(cmd, shell=True)
