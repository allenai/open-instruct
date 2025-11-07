from datasets import load_dataset, Dataset

new_elems = []

math_ds_id = load_dataset("jacobmorrison/math-prompts-used", split="train[:1000]")

for elem in math_ds_id:
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

code_ds_id = load_dataset("jacobmorrison/code-prompts-used", split="train[:1000]")

for elem in code_ds_id:
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

if_ds_id = load_dataset("jacobmorrison/if-prompts-used", split="train[:1000]")

for elem in if_ds_id:
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

math_ds_ood = load_dataset("jacobmorrison/rlvr_math_ood", split="train[:1000]")

for elem in math_ds_ood:
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
        "in_distribution": False
    })

code_ds_ood = load_dataset("jacobmorrison/rlvr_code_ood", split="train[:1000]")

for elem in code_ds_ood:
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
        "in_distribution": False
    })

if_ds_ood = load_dataset("jacobmorrison/rlvr_if_ood", split="train[:1000]")

for elem in if_ds_ood:
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
        "in_distribution": False
    })

eval_dataset = Dataset.from_list(new_elems)
eval_dataset.push_to_hub("jacobmorrison/social-rl-eval-prompts", split="train[:1000]")