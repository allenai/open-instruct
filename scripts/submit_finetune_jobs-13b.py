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
    # TODO: fix context length/etc below
    # ("science", "tulu_all-science_none.jsonl", False),
    # ("science", "tulu_none-science_2500.jsonl", False),
    ("science", "tulu_all-science_2500.jsonl", False),

    # ("safety", "tulu_all-safety_none.jsonl", False),
    # ("safety", "tulu_none-safety_100.jsonl", False),
    ("safety", "tulu_all-safety_100.jsonl", False),

    # ("coding", "tulu_all-coding_none.jsonl", False),
    # ("coding", "tulu_none-coding_100.jsonl", False),
    ("coding", "tulu_all-coding_100.jsonl", False),
    ("coding", "tulu_v2_mix-coding_100.jsonl", False),

    # # TODO: implement continued ft per domain
    # ("science", "tulu_none-science_2500.jsonl", True),
    # ("science", "tulu_match-science_2500.jsonl", True),

    # ("safety", "tulu_none-safety_100.jsonl", True),
    # ("safety", "tulu_match-safety_100.jsonl", True),

    # ("coding", "tulu_none-coding_100.jsonl", True),
    # ("coding", "tulu_match-coding_100.jsonl", True),
    # # TODO: these two are trained on top of the standard tulu 2 retrained
    # ("coding", "tulu_none-coding_100.jsonl", True),
    # ("coding", "tulu_v2_mix_match-coding_100.jsonl", True),
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
        f"--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/{domain}/{dataset}"
    )

    if continued_ft:
        print("continued finetuning is not implemented!")
        print(f"inputs: {domain}, {dataset}, {continued_ft}")
        quit()

    if cluster == "ai2/jupiter-cirrascale":
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
        # # save location
        # d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
        #     "--output_dir /output/", 
        #     f"--output_dir /net/weka/reviz/jacobm/modular_adaptation/checkpoints/{base_model}-{dataset.replace('.jsonl', '').replace('_eval_no', '')}/", 
        # )
        # train data location
        d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].replace(
            f"--train_file /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/{domain}/{dataset}", 
            f"--train_file /net/weka/reviz/jacobm/modular_adaptation/training_data/{domain}/{dataset}", 
        )
         

    fn = "beaker_configs/auto_created/{}.yaml".format(exp_name)
    file = open(fn, "w")
    yaml.dump(d, file, default_flow_style=True)
    file.close()

    cmd = "beaker experiment create {} --workspace ai2/modular-adaptation-coding".format(fn)
    subprocess.Popen(cmd, shell=True)
