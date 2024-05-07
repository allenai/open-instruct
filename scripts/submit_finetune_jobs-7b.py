import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_finetune_7b.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

model_size = "7B"
upload_to_beaker = True

# cluster = "ai2/general-cirrascale"
# cluster = "ai2/yizhongw-a100-80gb"
# cluster = "ai2/pluto-cirrascale"
cluster = "ai2/jupiter-cirrascale"
num_gpus = 8
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "preemptible"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

# ----------------------- experiments -----------------------

# tulu_all_no_science_no_safety.jsonl
# tulu_all_no_science_no_safety_no_coding.jsonl
# tulu_all_no_science_no_safety-coding_100.jsonl
# tulu_all_no_science_no_safety-safety_100.jsonl
# tulu_all_no_science_no_safety_no_coding-coding_100.jsonl
# tulu_all_no_science_no_safety_no_coding-safety_100.jsonl
# tulu_all_no_science_no_safety_no_coding-science_2500.jsonl
# tulu_all_no_science_no_safety-science_2500.jsonl

domain_info = [
    # ("tulu", "tulu_all_no_science_no_safety.jsonl", False),
    # ("tulu", "tulu_all_no_science_no_safety_no_coding.jsonl", False),

    # ("safety", "tulu_all_no_science_no_safety-safety_100.jsonl", False),
    # ("science", "tulu_all_no_science_no_safety-science_2500.jsonl", False),
    # ("coding", "tulu_all_no_science_no_safety-coding_100.jsonl", False),

    # ("safety", "tulu_all_no_science_no_safety_no_coding-safety_100.jsonl", False),
    # ("science", "tulu_all_no_science_no_safety_no_coding-science_2500.jsonl", False),
    # ("coding", "tulu_all_no_science_no_safety_no_coding-coding_100.jsonl", False),

    # ("tulu", "tulu_mix_v2.jsonl", False),

    # # TODO: implement continued ft per domain
    ("science", "tulu_none-science_2500.jsonl", True),
    ("safety", "tulu_none-safety_100.jsonl", True),
    ("coding", "tulu_none-coding_50.jsonl", True),
    ("coding", "tulu_none-coding_100.jsonl", True),
    
    ("science", "tulu_match_no_science_no_safety-science_2500.jsonl", True),
    ("safety", "tulu_match_no_science_no_safety-safety_100.jsonl", True),
    # ("coding", "tulu_match_no_science_no_safety-coding_50.jsonl", True), # haven't made this file yet
    ("coding", "tulu_match_no_science_no_safety-coding_100.jsonl", True),
]

for domain, dataset, continued_ft in domain_info:
    d = copy.deepcopy(d1)

    # name and description
    if continued_ft:
        base_model = "tulu_2_7b_no_science_no_safety"
    else:
        base_model = "llama_2_7b"
    exp_name = f"finetune-{base_model}-{dataset.replace('.jsonl', '')}-4k"
    d['description'] = exp_name
    d['tasks'][0]['name'] = exp_name

    # d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
    #     "--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_v2_mix.jsonl", 
    #     f"--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/{dataset}"
    # )

    # model location
    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--model_name_or_path /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/", 
        "--model_name_or_path /net/weka/reviz/jacobm/modular_adaptation/checkpoints/llama_2_7b/", 
    )
    # tokenizer location
    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--tokenizer_name /net/nfs.cirrascale/allennlp/yizhongw/hf_llama2_models/7B/", 
        "--tokenizer_name /net/weka/reviz/jacobm/modular_adaptation/checkpoints/llama_2_7b/", 
    )
    # train data location
    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_v2_mix.jsonl", 
        f"--train_file /net/weka/reviz/jacobm/modular_adaptation/training_data/consistent_mix/{dataset}", 
    )
    
    if continued_ft:
        # model location
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--model_name_or_path /net/weka/reviz/jacobm/modular_adaptation/checkpoints/llama_2_7b/", 
            "--model_name_or_path /net/weka/reviz/jacobm/modular_adaptation/checkpoints/llama_2_7b-tulu_all_no_science_no_safety/",
        )
        # tokenizer location
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--tokenizer_name /net/weka/reviz/jacobm/modular_adaptation/checkpoints/llama_2_7b/", 
            "--tokenizer_name /net/weka/reviz/jacobm/modular_adaptation/checkpoints/llama_2_7b-tulu_all_no_science_no_safety/",
        )
        

         

    fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/modular_adaptation".format(fn)
    subprocess.Popen(cmd, shell=True)
