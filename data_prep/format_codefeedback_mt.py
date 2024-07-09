from datasets import load_dataset
import json
import random

ds = load_dataset("m-a-p/Code-Feedback")

dataset = load_dataset("allenai/tulu-v2-sft-mixture")
tulu_data = []
for elem in dataset["train"]:
    tulu_data.append(elem)

code_data = []

i = 0
for elem in ds["train"]:
    inst = {
        "dataset": "codefeedback-mt",
        "id": str(i),
        "messages": elem["messages"]
    }
    code_data.append(inst)
    i += 1

random.shuffle(code_data)
with open("/oe-adapt-default/jacobm/tulu-3-dev/data/codefeedback-mt.jsonl", "w") as f_out:
    for elem in code_data:
        print(json.dumps(elem), file=f_out)

all_data = code_data + tulu_data
random.shuffle(all_data)
with open("/oe-adapt-default/jacobm/tulu-3-dev/data/tulu-and-codefeedback-mt.jsonl", "w") as f_out:
    for elem in all_data:
        print(json.dumps(elem), file=f_out)