import copy
import subprocess
import yaml
import random
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_finetune_7b.yaml", 'r') as f:
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

# tulu_mix_v2.jsonl
# tulu_none-science_2500.jsonl
# tulu_none-safety_100.jsonl
# tulu_none-coding_50.jsonl
# tulu_none-coding_100.jsonl
# tulu_mix_v2-science_2500.jsonl
# tulu_mix_v2-safety_100.jsonl
# tulu_mix_v2-coding_50.jsonl
# tulu_mix_v2-coding_100.jsonl 
# tulu_mix_v2_match-science_2500.jsonl
# tulu_mix_v2_match-safety_100.jsonl
# tulu_mix_v2_match-coding_50.jsonl
# tulu_mix_v2_match-coding_100.jsonl

domain_info = [
    # ("tulu", "tulu_mix_v2.jsonl", False),

    # TODO: fix context length/etc below
    ("science", "tulu_none-science_2500.jsonl", False),
    ("science", "tulu_mix_v2-science_2500.jsonl", False),

    ("safety", "tulu_none-safety_100.jsonl", False),
    ("safety", "tulu_mix_v2-safety_100.jsonl", False),

    ("coding", "tulu_none-coding_50.jsonl", False),
    ("coding", "tulu_mix_v2-coding_50.jsonl", False),

    ("coding", "tulu_none-coding_100.jsonl", False),
    ("coding", "tulu_mix_v2-coding_100.jsonl", False),

    # # TODO: implement continued ft per domain
    # ("science", "tulu_none-science_2500.jsonl", True),
    # ("science", "tulu_mix_v2_match-science_2500.jsonl", True),

    # ("safety", "tulu_none-safety_100.jsonl", True),
    # ("safety", "tulu_mix_v2_match-safety_100.jsonl", True),

    # ("coding", "tulu_none-coding_50.jsonl", True),
    # ("coding", "tulu_mix_v2_match-coding_50.jsonl", True),

    # ("coding", "tulu_none-coding_100.jsonl", True),
    # ("coding", "tulu_mix_v2_match-coding_100.jsonl", True),
]
model_size = "7B"

for domain, dataset, continued_ft in domain_info:
    d = copy.deepcopy(d1)

    # name and description
    if continued_ft:
        base_model = "tulu_2_7b"
    else:
        base_model = "llama_2_7b"
    exp_name = f"finetune-{base_model}-{dataset.replace('.jsonl', '')}-4k"
    d['description'] = exp_name
    d['tasks'][0]['name'] = exp_name

    d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        "--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_v2_mix.jsonl", 
        f"--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/full_tulu_mixtures/{dataset}"
    )

    if continued_ft:
        print("continued finetuning is not implemented!")
        print(f"inputs: {domain}, {dataset}, {continued_ft}")
        quit()

    if cluster == "ai2/jupiter-cirrascale":
        pass
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
        # save location
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            "--output_dir /output/", 
            f"--output_dir /net/weka/reviz/jacobm/modular_adaptation/checkpoints/full_tulu_mix/{base_model}-{dataset.replace('.jsonl', '')}-4k/", 
        )
        # train data location
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            f"--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/full_tulu_mixtures/{dataset}", 
            f"--train_file /net/weka/reviz/jacobm/modular_adaptation/training_data/full_tulu_mixtures/{dataset}", 
        )
         

    fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/modular_adaptation".format(fn)
    subprocess.Popen(cmd, shell=True)
