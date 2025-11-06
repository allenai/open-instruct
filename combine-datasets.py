from datasets import load_dataset, Dataset

new_elems = []

math_ds_id = load_dataset("jacobmorrison/math-prompts-used")

for elem in math_ds_id["train"]:
    prompt = elem["prompt"]
    if prompt.startswith("user: "):
        prompt = prompt[6:].strip()
    new_elems.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "domain": "math",
        "in_distribution": True
    })

code_ds_id = load_dataset("jacobmorrison/code-prompts-used")

for elem in code_ds_id["train"]:
    prompt = elem["prompt"]
    if prompt.startswith("user: "):
        prompt = prompt[6:].strip()
    new_elems.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "domain": "code",
        "in_distribution": True
    })

if_ds_id = load_dataset("jacobmorrison/if-prompts-used")

for elem in if_ds_id["train"]:
    prompt = elem["prompt"]
    if prompt.startswith("user: "):
        prompt = prompt[6:].strip()
    new_elems.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "domain": "ifeval",
        "in_distribution": True
    })

math_ds_ood = load_dataset("jacobmorrison/rlvr_math_ood")

for elem in math_ds_ood["train"]:
    prompt = elem["messages"][0]["content"]
    if prompt.startswith("user: "):
        prompt = prompt[6:].strip()
    new_elems.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "domain": "math",
        "in_distribution": True
    })

code_ds_ood = load_dataset("jacobmorrison/rlvr_code_ood")

for elem in code_ds_ood["train"]:
    prompt = elem["messages"][0]["content"]
    if prompt.startswith("user: "):
        prompt = prompt[6:].strip()
    new_elems.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "domain": "code",
        "in_distribution": True
    })

if_ds_ood = load_dataset("jacobmorrison/rlvr_if_ood")

for elem in if_ds_ood["train"]:
    prompt = elem["messages"][0]["content"]
    if prompt.startswith("user: "):
        prompt = prompt[6:].strip()
    new_elems.append({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "domain": "ifeval",
        "in_distribution": True
    })
