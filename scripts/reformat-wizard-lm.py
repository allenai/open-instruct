from datasets import load_dataset, Dataset

new_list = []
ds = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k")
for elem in ds:
    new_messages = []
    for turn in elem["conversations"]:
        if turn["from"] == "human":
            role = "user"
        elif turn["from"] == "gpt":
            role = "assistant"
        else:
            print(f"who is this? {turn['from']}")
        new_messages.append({
            "role": role,
            "content": turn["value"]
        })
    new_list.append({
        "messages": new_messages
    })

new_ds = Dataset.from_list(new_list)
new_ds.push_to_hub("ai2-adapt-dev/WizardLM_evol_instruct_V2_196k_reformat")