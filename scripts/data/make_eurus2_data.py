from datasets import load_dataset, Dataset
import random
from tqdm import tqdm

random_gen = random.Random(42)

dataset = load_dataset("ai2-adapt-dev/eurus2_ground_truth", split="train")

new_data = []
for sample in tqdm(dataset):
    sampled_length = random_gen.sample(range(100, 16384), 1)[0]
    # add a random length.
    sample['messages'][0]['content'] += f"\nUse no more than {sampled_length} tokens."
    sample['ground_truth'] = sample['ground_truth'] + "<sep>" + str(sampled_length)
    sample['dataset'] = "MATH,max_length"
    new_data.append(sample)

# combine into one dataset and push
random_gen.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("ai2-adapt-dev/eurus2_ground_truth_with_random_max_length")
