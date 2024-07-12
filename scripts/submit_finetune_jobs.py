import copy
import subprocess
import yaml
from datetime import date
import argparse
import os 

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser(description="Run experiment with Beaker config")
    parser.add_argument("--beaker_config", default="configs/beaker_configs/default_finetune.yaml", 
                        help="Path to the default Beaker config file")
    parser.add_argument("--additional_config", default=None, 
                        help="Path to an additional config file to override default settings")
    parser.add_argument("--wandb_api_key", required=True, help="Weights & Biases API key")
    parser.add_argument("--cluster", type=str, default="ai2/allennlp-cirrascale", help="Beaker cluster to use")
    parser.add_argument("--priority", type=str, default="normal", help="Priority of the job")
    parser.add_argument("--preemptible", type=bool, default=True, help="Whether to use preemptible instances")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    args = parser.parse_args()

    today = date.today().strftime("%m%d%Y")
    with open(args.beaker_config, 'r') as f:
        default_yaml = f.read()
    d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

    d1['tasks'][0]['context']['cluster'] = args.cluster
    d1['tasks'][0]['context']['priority'] = args.priority
    d1['tasks'][0]['context']['preemptible'] = args.preemptible # True requried for Jupiter/Pluto
    d1['tasks'][0]['resources']['gpuCount'] = args.num_gpus

    # modify here for different set of experiments
    experiment_group = "dataset_comparison"
    wandb_project = "open_instruct"
    if args.wandb_api_key:
        wandb_api_key = args.wandb_api_key
    else:
        wandb_api_key = os.environ.get("WANDB_API_KEY")


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
                    "--deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf",
                    "--deepspeed_config_file configs/ds_configs/stage3_offloading_accelerate.conf",
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

            fn = "configs/beaker_configs/auto_created/{}.yaml".format(exp_name)
            file = open(fn, "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            cmd = "beaker experiment create {} --workspace ai2/yizhong_default".format(fn)
            subprocess.Popen(cmd, shell=True)


if __name__ == "__main__":
    main()