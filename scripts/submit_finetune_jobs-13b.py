import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_finetune_13b.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/general-cirrascale"
# cluster = "ai2/yizhongw-a100-80gb"
# cluster = "ai2/pluto-cirrascale"
cluster = "ai2/jupiter-cirrascale"
num_gpus = 8
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "preemptible"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

# ----------------------- experiments -----------------------

domain_info = [
    # ("tulu", "tulu_all_no_science_no_safety.jsonl", False),
    ("tulu", "tulu_all_no_science_no_safety_no_coding.jsonl", False),

    # ("safety", "tulu_all_no_science_no_safety-safety_100.jsonl", False),
    # ("science", "tulu_all_no_science_no_safety-science_2500.jsonl", False),
    # ("coding", "tulu_all_no_science_no_safety-coding_100.jsonl", False),

    ("safety", "tulu_all_no_science_no_safety_no_coding-safety_100.jsonl", False),
    ("science", "tulu_all_no_science_no_safety_no_coding-science_2500.jsonl", False),
    ("coding", "tulu_all_no_science_no_safety_no_coding-coding_100.jsonl", False),

    # TODO: implement continued ft per domain
]
model_size = "13B"

for domain, dataset, continued_ft in domain_info:
    d = copy.deepcopy(d1)

    # name and description
    if continued_ft:
        base_model = "tulu_2_13b"
    else:
        base_model = "llama_2_13b"
    exp_name = f"finetune-{base_model}-{dataset.replace('.jsonl', '').replace('_eval_no', '')}"
    d['description'] = exp_name
    d['tasks'][0]['name'] = exp_name

    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_v2_mix.jsonl", 
        f"--train_file /net/weka/reviz/jacobm/modular_adaptation/training_data/consistent_mix/{dataset}", 
    )

    if continued_ft:
        print("continued finetuning is not implemented!")
        print(f"inputs: {domain}, {dataset}, {continued_ft}")
        quit()

    # model location
    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/13B/", 
        "--model_name_or_path /net/weka/reviz/jacobm/modular_adaptation/checkpoints/llama_2_13b/", 
    )
    # tokenizer location
    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/13B/", 
        "--tokenizer_name /net/weka/reviz/jacobm/modular_adaptation/checkpoints/llama_2_13b/", 
    )
         

    fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/modular-adaptation-coding".format(fn)
    subprocess.Popen(cmd, shell=True)
