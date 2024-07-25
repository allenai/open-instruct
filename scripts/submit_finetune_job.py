import copy
import subprocess
import yaml
from datetime import datetime
import argparse
import re 
import shlex

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def main():
    parser = argparse.ArgumentParser(description="Run experiment with Beaker config")
    parser.add_argument("--default_beaker_config", default="configs/beaker_configs/default_finetune.yaml", 
                        help="Path to the default Beaker config file")
    parser.add_argument("--config", default=None, 
                        help="Path to an additional config file to override default settings")
    # parser.add_argument("--wandb_api_key", required=False, help="Weights & Biases API key")
    parser.add_argument("--cluster", type=str, default="ai2/allennlp-cirrascale", help="Beaker cluster to use")
    parser.add_argument("--priority", type=str, default="high", help="Priority of the job")
    parser.add_argument("--preemptible", type=bool, default=True, help="Whether to use preemptible instances")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--image", type=str, default="nathanl/open_instruct_auto", help="Beaker image to use.")
    parser.add_argument("--workspace", type=str, default="ai2/tulu-2-improvements", help="Beaker workspace to use.")
    # allow unknown args from CLI, use this to modify loaded config in bash scripts for sweeping
    # Note, can only override args in --config passed (not default FlatArguments class in open_instruct/utils.py)
    
    # Use parse_known_args instead of parse_args
    args, unknown = parser.parse_known_args()

    # Process unknown arguments
    # must be of the form --{arg} {value}
    unknown_args = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith('--'):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i+1].startswith('--'):
                value = unknown[i+1]
                i += 2
            else:
                value = None
                i += 1
            unknown_args[key] = value
        else:
            i += 1

    # Print known arguments
    train_config = load_yaml(args.config)
    print("Config:", train_config)

    # Print unknown arguments
    print("Unknown arguments:", unknown_args)

        
    now = datetime.now().strftime("%m%d%Y%H%M%S")
    with open(args.default_beaker_config, 'r') as f:
        default_yaml = f.read()
    d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

    d1["tasks"][0]["image"]["beaker"] = args.image
    d1['tasks'][0]['context']['cluster'] = args.cluster
    d1['tasks'][0]['context']['priority'] = args.priority
    d1['tasks'][0]['context']['preemptible'] = args.preemptible # True requried for Jupiter/Pluto
    d1['tasks'][0]['resources']['gpuCount'] = args.num_gpus

    # modify here for different set of experiments
    experiment_group = "dataset_comparison"
    wandb_project = "open_instruct"
    # if args.wandb_api_key:
    #     wandb_api_key = args.wandb_api_key
    # else:
    #     wandb_api_key = os.environ.get("WANDB_API_KEY")

    # if config is passed, load and merge that
    def override_and_reconstruct_command(original_command, train_config, unknown_args):
        def parse_args(args):
            cmd_dict = {}
            i = 0
            while i < len(args):
                if args[i].startswith('--'):
                    key = args[i][2:]
                    if i + 1 < len(args) and not args[i+1].startswith('--'):
                        cmd_dict[key] = args[i+1]
                        i += 2
                    else:
                        cmd_dict[key] = True
                        i += 1
                else:
                    i += 1
            return cmd_dict
        # Split the original command into a list
        cmd_parts = shlex.split(original_command)

        # Find the index of open_instruct/finetune.py
        script_index = cmd_parts.index('open_instruct/finetune.py')

        # Separate the command into pre-script and post-script parts
        pre_script = cmd_parts[:3]  # 'accelerate launch'
        pre_script_args = cmd_parts[3:script_index]
        post_script_args = cmd_parts[script_index+1:]

        # Parse arguments
        pre_dict = parse_args(pre_script_args)
        post_dict = parse_args(post_script_args)

        # Combine dictionaries and apply overrides
        cmd_dict = {**post_dict}
        cmd_dict.update(train_config)
        cmd_dict.update(unknown_args)
        
        # Reconstruct the command string
        new_cmd_parts = pre_script
        # add pre python args
        for key, value in pre_dict.items():
            new_cmd_parts.append(f'--{key}')
            if value is not True:
                new_cmd_parts.append(str(value))
        # add python job + post args
        new_cmd_parts.append('open_instruct/finetune.py')
        for key, value in cmd_dict.items():
            new_cmd_parts.append(f'--{key}')
            # if string in [], expand args
            if isinstance(value, list):
                for v in value:
                    new_cmd_parts.append(str(v))
            elif value is not True:
                new_cmd_parts.append(str(value))

        return ' '.join(new_cmd_parts)

    new_arguments = override_and_reconstruct_command(d1['tasks'][0]['arguments'][0], train_config, unknown_args)
    
    # place --num_processes with args.num_gpus
    # will be --num_processes {N} before
    new_arguments = re.sub(r'--num_processes \d+', f'--num_processes {args.num_gpus}', new_arguments)

    model_name = get_model_name(new_arguments)
    # if model name has /, replace with _
    model_name = model_name.replace("/", "_")
    # try given config only has one
    dataset_name, dataset_mixer, train_file = check_dataset_selection(new_arguments)
    print(f"Dataset selection is valid.")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset mixer: {dataset_mixer}")
    print(f"Train file: {train_file}")

    d = copy.deepcopy(d1)
    d['tasks'][0]['arguments'][0] = new_arguments

    # name and description
    exp_name = f"open_instruct_finetune_{model_name}_{now}"
    d['description'] = exp_name
    d['tasks'][0]['name'] = exp_name

    # WANDB settings
    for env in d['tasks'][0]['envVars']:
        if env['name'] == "WANDB_DISABLED":
            env['value'] = False
        if env['name'] == "WANDB_PROJECT":
            env['value'] = wandb_project
    # d['tasks'][0]['envVars'].append({
    #     'name': 'WANDB_API_KEY', 'value': wandb_api_key
    # })
    d['tasks'][0]['envVars'].append({
        'name': 'WANDB_NAME', 'value': exp_name
    })
    d['tasks'][0]['envVars'].append({
        'name': 'WANDB_RUN_GROUP', 'value': experiment_group
    })

    # optionally, print to debug config
    # print(d)

    fn = "configs/beaker_configs/auto_created/{}.yaml".format(exp_name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = f"beaker experiment create {fn} --workspace {args.workspace}"
    subprocess.Popen(cmd, shell=True)

def check_dataset_selection(command_string):
    parts = shlex.split(command_string)
    dataset_name = None
    dataset_mixer = None
    train_file = None

    for i, part in enumerate(parts):
        if part == '--dataset_name' and i + 1 < len(parts):
            dataset_name = parts[i + 1]
        elif part == '--dataset_mixer' and i + 1 < len(parts):
            dataset_mixer = parts[i + 1]
        elif part == '--train_file' and i + 1 < len(parts):
            train_file = parts[i + 1]

    if ((dataset_name is not None and dataset_mixer is not None) or
        (dataset_name is not None and train_file is not None) or
        (dataset_mixer is not None and train_file is not None)):
        raise ValueError("Cannot provide two dataset selection mechanisms.")

    return dataset_name, dataset_mixer, train_file

def get_model_name(command_string):
    parts = shlex.split(command_string)
    for i, part in enumerate(parts):
        if part == '--model_name_or_path':
            if i + 1 < len(parts):
                return parts[i + 1]
    return None  # Return None if model name is not found


if __name__ == "__main__":
    main()