import copy
import subprocess
import yaml
import re
import itertools
from datetime import date
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="tulu-3-dev")
    parser.add_argument("--beaker_image", type=str, default="nathanl/open_instruct_auto", help="If given, use this Beaker image.")
    parser.add_argument("--beaker_config", type=str, default="configs/beaker_configs/default_merge.yaml")
    parser.add_argument("--merge_config", type=str, default="configs/merge_configs/example_linear_merge_config.yaml")
    parser.add_argument("--cluster", nargs='+', default=["ai2/neptune-cirrascale", "ai2/saturn-cirrascale", "ai2/jupiter-cirrascale-2"])
    parser.add_argument("--priority", type=str, default="high")
    parser.add_argument("--preemptible", action="store_true", default=True, help="for using preemtipble jobs (required on some instances)")
    parser.add_argument("--output_dir", type=str, default="/output")
    args = parser.parse_args()

    with open(args.merge_config, 'r') as f:
        default_yaml = f.read()
    mergeConfig = yaml.load(default_yaml, Loader=yaml.FullLoader)

    # TODO: support SLERP
    assert mergeConfig["merge_method"] in ["linear", "task_arithmetic"], f"merging method {mergeConfig['merge_method']} not supported"

    with open(f"configs/merge_configs/base_configs/default_{mergeConfig['merge_method']}_merge.yaml", 'r') as f:
        merge_yaml = f.read()
    baseConfig = yaml.load(merge_yaml, Loader=yaml.FullLoader)

    baseConfig["normalize"] = mergeConfig["normalize"]
    baseConfig["models"] = []

    if mergeConfig["merge_method"] == "task_arithmetic":
        baseConfig["models"].append({
            "model": mergeConfig["base_model"]
        })
        baseConfig["base_model"] = mergeConfig["base_model"]

    beakerDatasets = []
    wekaBuckets = set()
    for elem in mergeConfig["models"]:
    #   - model: /model-one
    #     parameters:
    #       weight: 1.0
        
    #   - name: name
    #     location: beaker
    #     path: jacobm/beaker-dataset
    #     weight: 0.5
        if elem["location"] == "beaker":
            model_data = {
                "model": f"/{elem['name']}",
                "parameters": {"weight": float(elem["weight"])}
            }
            if mergeConfig["merge_method"] == "task_arithmetic":
                model_data["parameters"]["normalize"] = mergeConfig["normalize"]
            # beakerConfig['datasets'][1]['source']['beaker'] = model_info[1]
    #   - mountPath: /hf_llama_models
    #     source:
    #       beaker: Yizhongw03/hf_llama_model_7B
            beakerDatasets.append({
                "mountPath": f"/{elem['name']}",
                "source": {"beaker": elem["path"]}
            })
            # mount datasets
        elif elem["location"] in ["huggingface", "nfs"]:
            model_data = {
                "model": elem['path'],
                "parameters": {"weight": float(elem["weight"])}
            }
            if mergeConfig["merge_method"] == "task_arithmetic":
                model_data["parameters"]["normalize"] = mergeConfig["normalize"]
        elif elem["location"] == "weka": # verify the only available cluster(s) have weka
            if elem["wekaBucket"] not in wekaBuckets:
                beakerDatasets.append({
                    "mountPath": f"/{elem['wekaBucket']}",
                    "source": {"weka": elem["wekaBucket"]}
                })
                wekaBuckets.add(elem["wekaBucket"])
            model_data = {
                "model": elem["path"],
                "parameters": {"weight": float(elem["weight"])}
            }
            if mergeConfig["merge_method"] == "task_arithmetic":
                model_data["parameters"]["normalize"] = mergeConfig["normalize"]
        else:
            print(f"Unsupported location: {elem['location']}")
        baseConfig["models"].append(model_data)

    with open(args.beaker_config, 'r') as f:
        beaker_yaml = f.read()
    beakerConfig = yaml.load(beaker_yaml, Loader=yaml.FullLoader)

    beakerConfig['tasks'][0]['image']['beaker'] = args.beaker_image
    # TODO: fix these
    beakerConfig['tasks'][0]['constraints']['cluster'] = args.cluster
    beakerConfig['tasks'][0]['context']['priority'] = args.priority
    beakerConfig['tasks'][0]['context']['preemptible'] = args.preemptible # True required for Jupiter/Pluto

    print(beakerConfig)
    
    if len(beakerDatasets) > 0:
        beakerConfig["tasks"][0]["datasets"] = beakerDatasets
    base_command = beakerConfig["tasks"][0]["arguments"][0].replace("{OUTPUT_DIR}", args.output_dir)
    beakerConfig["tasks"][0]["arguments"][0] = base_command.replace("{RAW_CONFIG}", f'"{str(baseConfig)}"')

    experiment_name = f"open_instruct_merge_models" 
    beakerConfig["description"] = experiment_name
    # if configs/beaker_configs/auto_created doesn't exist, create it with os
    if not os.path.exists("configs/beaker_configs/auto_created"):
        os.makedirs("configs/beaker_configs/auto_created")
    fn = "configs/beaker_configs/auto_created/{}.yaml".format(experiment_name)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, "w") as file:
        yaml.dump(beakerConfig, file, default_flow_style=True)    

    cmd = "beaker experiment create {} --workspace ai2/{}".format(fn, args.workspace)
    subprocess.Popen(cmd, shell=True)

if __name__ == "__main__":
    main()