import json
import random

data = []
with open("/oe-adapt-default/jacobm/tulu-3-dev/data/tulu_v2_mix.jsonl") as f_in:
    for line in f_in.readlines():
        data.append(json.loads(line))

with open("/oe-adapt-default/jacobm/tulu-3-dev/data/codefeedback-single-turn.jsonl") as f_in:
    for line in f_in.readlines():
        data.append(json.loads(line))

with open("/oe-adapt-default/jacobm/tulu-3-dev/data/mathplus-500k.jsonl") as f_in:
    for line in f_in.readlines():
        data.append(json.loads(line))

random.shuffle(data)
with open("/oe-adapt-default/jacobm/tulu-3-dev/data/tulu-and-mathplus_500k-and-codefeedback-single-turn.jsonl", "w") as f_out:
    for elem in data:
        print(json.dumps(elem), file=f_out)