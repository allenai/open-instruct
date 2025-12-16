import random

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils

random_gen = random.Random(42)

ds = load_dataset(
    "jacobmorrison/tulu-3-sft-t3-70b-thoughts", split="train", num_proc=open_instruct_utils.max_num_processes()
)
new_data = []

for sample in ds:
    original_messages = sample["messages"]
    new_messages = []
    for message in original_messages:
        # <think> gets added by the chat template, so we dont add it here
        new_content = message["content"].replace("<|reserved_special_token_246|>", "<think>")
        new_content = new_content.replace("<|reserved_special_token_247|>", "</think><answer>")
        if "<answer>" in new_content:
            new_content = new_content + "</answer>"
        new_messages.append({"role": message["role"], "content": new_content})
    last_turn_no_think = (
        new_messages[-1]["content"].split("</think>")[-1].replace("<answer>", "").replace("</answer>", "")
    )
    new_data.append({"messages": new_messages, "ground_truth": last_turn_no_think, "dataset": "tulu_thinker"})

random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("hamishivi/tulu-3-sft-t3-70b-thinker")
