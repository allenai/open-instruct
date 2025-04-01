import argparse
import copy
import os
import subprocess
from datetime import date, datetime
import uuid

# Base Beaker configuration as a Python dictionary
BASE_CONFIG = {
    "version": "v2",
    "budget": "ai2/oe-adapt",
    "description": "Best of N ranking experiment",
    "tasks": [{
        "envVars": [{"name": "HF_TOKEN", "secret": "HF_TOKEN"}, {"name": "CUDA_DEVICE_ORDER", "value":"PCI_BUS_ID"}],
        "command": ["/bin/sh", "-c"],
        "name": "bon_ranking",
        "image": {
            "beaker": "your-org/bon-ranking"
        },
        "constraints": {
            "cluster": ["ai2/jupiter-cirrascale-2"]
        },
        "context": {
            "priority": "normal"
        },
        "datasets":[{
            "mountPath": "/weka/oe-adapt-default",
            "source": {"weka":"oe-adapt-default"}
        }],
        "resources": {
            "gpuCount": 1
        },
        "arguments": [
            "python run_ties.py"
        ]
    }]
}

def parse_args():
    parser = argparse.ArgumentParser()
    # Beaker-specific arguments
    parser.add_argument("--image", type=str, default="nathanl/rewardbench_auto", help="Beaker image to use")
    parser.add_argument("--cluster", nargs='+', default=["ai2/jupiter-cirrascale-2","ai2/saturn-cirrascale","ai2/neptune-cirrascale"], help="Beaker cluster to use")
    parser.add_argument("--priority", type=str, default="normal", help="Priority of the job")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--workspace", type=str, default="ai2/reward-bench-v2", help="Beaker workspace")
    parser.add_argument("--mount", type=str, default="/weka/oe-adapt-default/", help="Mount")
    parser.add_argument("--source", type=str, default="oe-adapt-default", help="Source")
    
    # Required experiment parameters
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    
    # Optional experiment parameters
    parser.add_argument("--revision", type=str, default=None, help="model revision")
    parser.add_argument("--best_of", type=int, default=4, help="Number of best of n to select from")
    parser.add_argument("--tokenizer", type=str, help="Path to non-matching tokenizer to model")
    parser.add_argument("--chat_template", type=str, help="Path to chat template")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--do_not_save", action="store_true", help="Do not save results to hub (for debugging)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Directly load model instead of pipeline")
    parser.add_argument("--debug", action="store_true", help="Run on common preference sets instead of custom eval set")

    return parser.parse_args()

def create_experiment_name(args):
    model_name = args.model.split("/")[-1]
    dataset_name = args.dataset.split("/")[-1].split('.jsonl')[0]
    today = date.today().strftime("%m%d%Y")
    unique_id = str(uuid.uuid4())[:8]
    
    return f"bon_ranking_{dataset_name}_{model_name}_{unique_id}_{today}".replace("/", "-")[:128]

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs("beaker_configs/bon_experiments", exist_ok=True)
    
    # Create experiment config
    config = copy.deepcopy(BASE_CONFIG)
    
    # Set experiment name and description
    name = create_experiment_name(args)
    config["description"] = name
    config["tasks"][0]["name"] = name
    
    # Configure cluster and resources
    config["tasks"][0]["image"]["beaker"] = args.image
    # Add both mounts - one for data and one for code
    config["tasks"][0]["datasets"] = [
        {
            "mountPath": "/stage/reward-bench",
            "source": {"weka": "oe-adapt-default"},
            "subPath": "saumyam/cloned/reward-bench"  # This points to your directory
        }
    ]
    config["tasks"][0]["constraints"]["cluster"] = args.cluster
    config["tasks"][0]["context"]["priority"] = args.priority
    config["tasks"][0]["resources"]["gpuCount"] = args.num_gpus

    
    # Build base command with required parameters
    cmd_parts = [
        "cd /stage/reward-bench && python scripts/run_ties.py",
        f"--dataset {args.dataset}",
        f"--model {args.model}"
    ]
    
    # Optional parameters mapping
    optional_params = {
        "revision": "--revision",
        "best_of": "--best_of",
        "tokenizer": "--tokenizer",
        "chat_template": "--chat_template",
        "batch_size": "--batch_size"
    }
    
    # Add optional parameters if specified
    for param_name, cmd_arg in optional_params.items():
        value = getattr(args, param_name)
        if value is not None:
            if isinstance(value, str) and any(char.isspace() for char in value):
                cmd_parts.append(f"{cmd_arg} '{value}'")
            else:
                cmd_parts.append(f"{cmd_arg} {value}")
    
    # Add flags if they're True
    flag_params = [
        "do_not_save",
        "trust_remote_code",
        "debug"
    ]
    
    for flag in flag_params:
        if getattr(args, flag):
            cmd_parts.append(f"--{flag}")
    
    # Join command parts
    config["tasks"][0]["arguments"][0] = " ".join(cmd_parts)
    
    # Write config file
    config_path = f"beaker_configs/bon_experiments/{name}.yaml"
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    # Submit to Beaker
    print(f"Submitting experiment: {name}")
    beaker_cmd = f"beaker experiment create {config_path} --workspace {args.workspace}"
    subprocess.Popen(beaker_cmd, shell=True)

if __name__ == "__main__":
    main()