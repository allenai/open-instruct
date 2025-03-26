from datasets import load_dataset, Dataset
import random

random_gen = random.Random(42)

ds = load_dataset("natolambert/GeneralThought-430K-filtered", split="train")
new_data = []

for sample in ds:
    question = sample["question"]
    model_reasoning = sample["model_reasoning"]
    model_answer = sample["model_answer"]
    if not question or not model_reasoning or not model_answer:
        continue
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": f"{model_reasoning}</think><answer>{model_answer}</answer>"},
    ]
    new_data.append({
        "messages": messages,
        "ground_truth": model_answer,
        "dataset": "tulu_thinker"
    })

random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("hamishivi/GeneralThought-430K-filtered-thinker")
